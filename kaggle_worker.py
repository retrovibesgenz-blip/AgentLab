"""
AgentLab — Kaggle GPU worker orchestration
==========================================

Backup provider: run the vision model on YOUR Kaggle GPU instead of the Z.AI
cloud. Selected in the app via "Run on your Kaggle GPU".

Lifecycle (matches the app's flow):
  main server starts
    -> is a Kaggle worker already reachable?  (KAGGLE_ENDPOINT_URL, or the
       rendezvous dataset)  --> reuse it.
    -> NO: push the worker kernel via the Kaggle API. It boots, `server.py`
       loads the model, opens a cloudflared tunnel, and publishes its public
       URL to a private Kaggle "link" dataset.
    -> poll that dataset until status == READY and a URL is present.
    -> health-check the tunnel, then hand the base_url to the planner.
  ... app sends inference requests to the tunnel ...
  main server closes -> stop(): POST /shutdown to the tunnel so `server.py`
       exits and the kernel ends.

Uses the Kaggle Python API (KaggleApi) directly — no dependency on the `kaggle`
CLI being on PATH.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

APP_DIR = Path(__file__).resolve().parent
KAGGLE_DIR = APP_DIR / "kaggle_kernel"   # holds server.py + kernel-metadata.json


class KaggleWorker:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.base_url: Optional[str] = None      # https://xxxx.trycloudflare.com/v1
        self.tunnel: Optional[str] = None        # https://xxxx.trycloudflare.com
        self._api_obj = None

    # -- credential helpers ------------------------------------------------
    @staticmethod
    def _real(v: str) -> str:
        """Treat empty or PASTE_… placeholder values as unset."""
        v = (v or "").strip()
        return "" if v.startswith("PASTE_") else v

    @property
    def user(self) -> str:
        return self._real(self.cfg.kaggle_username)

    @property
    def key(self) -> str:
        return self._real(self.cfg.kaggle_key)

    @property
    def endpoint(self) -> str:
        return self._real(self.cfg.kaggle_endpoint_url)

    @property
    def available(self) -> bool:
        return bool(self.user and self.key) or bool(self.endpoint)

    def _api(self):
        """Authenticated KaggleApi (cached). Raises on bad credentials."""
        if self._api_obj is None:
            if self.user:
                os.environ["KAGGLE_USERNAME"] = self.user
            if self.key:
                os.environ["KAGGLE_KEY"] = self.key
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            self._api_obj = api
        return self._api_obj

    async def _health(self, tunnel: str) -> bool:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(tunnel.rstrip("/") + "/health")
                return r.status_code == 200
        except Exception:
            return False

    async def _health_model(self, tunnel: str) -> Optional[str]:
        """Which model the running worker is serving (from /health), or None."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(tunnel.rstrip("/") + "/health")
            if r.status_code == 200:
                return (r.json() or {}).get("model")
        except Exception:
            pass
        return None

    # -- main entry --------------------------------------------------------
    async def ensure_running(self, progress: Callable[[str], None]) -> Optional[str]:
        """Return an OpenAI-compatible base_url for the worker, or None on failure."""
        # 1) Manual / stable URL override — simplest path.
        if self.endpoint:
            tunnel = self.endpoint.rstrip("/")
            tunnel = tunnel[:-3] if tunnel.endswith("/v1") else tunnel
            progress("checking your Kaggle endpoint…")
            for _ in range(20):
                if await self._health(tunnel):
                    self.tunnel, self.base_url = tunnel, tunnel + "/v1"
                    progress("Kaggle GPU ready")
                    return self.base_url
                await asyncio.sleep(3)
            progress("couldn't reach KAGGLE_ENDPOINT_URL — is the worker running?")
            return None

        if not (self.user and self.key):
            progress("Kaggle not configured — add KAGGLE_USERNAME and KAGGLE_KEY to .env")
            return None

        # Authenticate up front so bad keys fail fast with a clear message.
        try:
            await asyncio.to_thread(self._api)
        except Exception as exc:
            progress(f"Kaggle login failed: {exc}")
            return None

        # 2) Reuse a URL already published by a running worker — but only if it's
        #    serving the model we want; otherwise stop it and start a fresh one.
        url = await asyncio.to_thread(self._read_link)
        if url and await self._health(url):
            running = await self._health_model(url)
            if not running or running == self.cfg.kaggle_model:
                self.tunnel, self.base_url = url, url + "/v1"
                progress("reusing running Kaggle GPU worker")
                return self.base_url
            progress(f"worker is running {running}; switching to {self.cfg.kaggle_model}…")
            import httpx
            try:
                async with httpx.AsyncClient(timeout=8) as c:
                    await c.post(url + "/shutdown")
            except Exception:
                pass
            await asyncio.sleep(5)

        # 3) Start the worker kernel.
        progress("starting your Kaggle GPU notebook…")
        ok, detail = await asyncio.to_thread(self._push_kernel)
        if not ok:
            progress(f"failed to start the Kaggle kernel: {detail}")
            return None

        # 4) Wait for the kernel to boot, load the model, and publish its URL.
        progress("Kaggle booting + loading the model (this takes a few minutes)…")
        deadline = time.time() + self.cfg.kaggle_boot_timeout
        ticks = 0
        while time.time() < deadline:
            url = await asyncio.to_thread(self._read_link)
            if url:
                progress("worker published its address — health-checking…")
                for _ in range(24):
                    if await self._health(url):
                        self.tunnel, self.base_url = url, url + "/v1"
                        progress("Kaggle GPU ready")
                        return self.base_url
                    await asyncio.sleep(5)
            ticks += 1
            if ticks % 3 == 0:  # every ~30s, check whether the kernel died/finished
                status = await asyncio.to_thread(self._kernel_status)
                if status:
                    progress(f"kernel status: {status}")
                    if status in ("error", "complete", "cancelAcknowledged", "cancelRequested"):
                        progress("the Kaggle kernel stopped before serving — open it on "
                                 "kaggle.com to read its log (model too big for the GPU, "
                                 "or a load error, are the usual causes)")
                        return None
            await asyncio.sleep(10)
        progress("timed out waiting for the Kaggle worker")
        return None

    async def stop(self) -> None:
        """Tell the worker to shut itself down (ends the kernel)."""
        if not self.tunnel:
            return
        import httpx
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                await c.post(self.tunnel + "/shutdown")
        except Exception:
            pass
        self.base_url = self.tunnel = None

    # -- kaggle plumbing (sync — call via asyncio.to_thread) ---------------
    def _kernel_ref(self) -> str:
        slug = (self._real(self.cfg.kaggle_kernel_slug) or f"{self.user}/agentlab-worker")
        return slug if "/" in slug else f"{self.user}/{slug}"

    def _link_dataset(self) -> str:
        ds = self._real(self.cfg.kaggle_link_dataset) or f"{self.user}/agentlab-endpoint"
        return ds if "/" in ds else f"{self.user}/{ds}"

    def _push_kernel(self) -> tuple[bool, str]:
        """Build a throwaway copy of the kernel (server.py + metadata) with the
        config baked in, and push THAT — so your key never touches a tracked
        file. The temp dir is deleted afterwards."""
        import shutil
        import tempfile
        build = Path(tempfile.mkdtemp(prefix="agentlab_kernel_"))
        try:
            shutil.copy(KAGGLE_DIR / "server.py", build / "server.py")
            self._bake_server_config(build / "server.py")
            (build / "kernel-metadata.json").write_text(self._kernel_metadata_json())
            self._api().kernels_push(str(build))
            print(f"[kaggle] pushed kernel {self._kernel_ref()}")
            return True, "pushed"
        except Exception as exc:
            print(f"[kaggle] push failed: {exc}")
            return False, str(exc)
        finally:
            shutil.rmtree(build, ignore_errors=True)

    def _bake_server_config(self, target: Path) -> None:
        """Fill the AGENTLAB CONFIG block in the build copy with this run's values.
        (A Kaggle kernel can't receive env vars/extra files, so we bake them into
        the uploaded script. The kernel is private and it's your own key.)"""
        import re as _re
        src = target.read_text(encoding="utf-8")
        repl = {
            "MODEL_ID": self.cfg.kaggle_model,
            "LINK_DATASET": self._link_dataset(),
            "KUSER": self.user,
            "KKEY": self.key,
            "HF_TOKEN": self._real(getattr(self.cfg, "hf_token", "")),
        }
        for var, val in repl.items():
            src = _re.sub(rf'(?m)^{var} = ".*?"',
                          f'{var} = {json.dumps(val)}', src, count=1)
        target.write_text(src, encoding="utf-8")

    def _kernel_metadata_json(self) -> str:
        ref = self._kernel_ref()
        return json.dumps({
            "id": ref,
            "title": ref.split("/")[-1],
            "code_file": "server.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }, indent=2)

    def _kernel_status(self) -> Optional[str]:
        """Best-effort kernel run status ('running'|'complete'|'error'|…)."""
        try:
            r = self._api().kernels_status(self._kernel_ref())
            st = getattr(r, "status", r)
            return str(getattr(st, "name", st)).lower()
        except Exception:
            return None

    def _read_link(self) -> Optional[str]:
        """Fetch the tunnel URL the worker published to the rendezvous dataset."""
        ds = self._link_dataset()
        try:
            with tempfile.TemporaryDirectory() as d:
                self._api().dataset_download_files(ds, path=d, unzip=True, quiet=True)
                f = Path(d) / "endpoint.json"
                if not f.exists():
                    return None
                data = json.loads(f.read_text())
                if str(data.get("status", "")).upper() == "READY" and data.get("url"):
                    return str(data["url"]).rstrip("/")
        except Exception:
            return None
        return None
