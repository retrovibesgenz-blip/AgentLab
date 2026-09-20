"""
AgentLab — Kaggle GPU worker (runs INSIDE a Kaggle notebook)
============================================================

This is the `server.py` the app pushes to Kaggle. It:
  1. loads an open vision-language model (default GLM-4.1V-9B) onto the GPU,
  2. serves an OpenAI-compatible /v1/chat/completions endpoint (text + images),
  3. opens a cloudflared quick-tunnel to get a public https URL,
  4. publishes that URL to a private Kaggle "link" dataset so the app finds it,
  5. stays alive until the app calls POST /shutdown.

You said you've already hosted the 9B model on Kaggle — if your loading code
differs, replace the two functions marked `# >>> MODEL <<<` with your working
version. Everything else (tunnel, OpenAI API shape, URL publishing, shutdown)
is generic and app-facing, so keep it.

Set MODEL_ID / LINK_DATASET below (or via env) to match your account.
"""

import base64
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# === AGENTLAB CONFIG (auto-filled into a throwaway copy on push — NOT here) ===
# These stay EMPTY in the repo. The app fills them in a temp build dir at push
# time so your Kaggle key is never written to a tracked file. Safe to commit.
MODEL_ID = "THUDM/GLM-4.1V-9B-Thinking"
LINK_DATASET = ""      # "<kaggle-username>/agentlab-endpoint" — where we publish our URL
KUSER = ""             # your Kaggle username (baked in at push so the kernel can publish)
KKEY = ""              # your Kaggle key
HF_TOKEN = ""          # optional Hugging Face token (faster downloads, no rate limit)
# === END AGENTLAB CONFIG ===

PORT = 8000
_should_exit = threading.Event()


# ---------------------------------------------------------------------------
def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, **kw)


def install_deps():
    # Kaggle already ships torch/transformers/accelerate/pillow/kaggle. Only add
    # what's missing, and DON'T --upgrade (that's what caused the pillow/gradio
    # dependency conflicts). bitsandbytes enables 4-bit loading.
    sh(f"{sys.executable} -m pip install -q fastapi uvicorn bitsandbytes")
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN  # avoids the unauthenticated-HF rate limit
    # cloudflared binary
    if not Path("cloudflared").exists():
        sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
           "cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared")


# ---------------------------------------------------------------------------
# >>> MODEL <<<  (swap this block for your exact working loader if different)
_model = None
_processor = None


def load_model():
    global _model, _processor
    import torch
    # GLM-4.1V / Qwen-VL / Gemma-V etc. are VISION models — they load with
    # AutoModelForImageTextToText, NOT AutoModelForCausalLM (that was the crash:
    # "Unrecognized configuration class Glm4vConfig for AutoModelForCausalLM").
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"loading {MODEL_ID} …", flush=True)
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    common = dict(device_map="auto", trust_remote_code=True)
    # A 9B model in bf16 (~18GB) won't fit a single 16GB Kaggle GPU, so load it
    # in 4-bit (~6GB). Falls back to bf16 on a 2xT4 / bigger GPU.
    try:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_quant_type="nf4")
        _model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=quant, **common)
        print("loaded in 4-bit", flush=True)
    except Exception as exc:
        print("4-bit load failed, trying bf16:", exc, flush=True)
        _model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, **common)
    _model.eval()
    print("model loaded", flush=True)


def run_model(messages, max_tokens=600, temperature=0.2):
    """messages: OpenAI-style list. Returns the assistant text."""
    import torch
    from PIL import Image

    # Convert OpenAI messages -> processor chat messages (text + PIL images).
    conv = []
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        if isinstance(content, str):
            conv.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        parts = []
        for p in content:
            if p.get("type") == "text":
                parts.append({"type": "text", "text": p.get("text", "")})
            elif p.get("type") == "image_url":
                url = p["image_url"]["url"]
                b64 = url.split(",", 1)[1] if url.startswith("data:") else url
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                parts.append({"type": "image", "image": img})
        conv.append({"role": role, "content": parts})

    inputs = _processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=max_tokens,
                              do_sample=temperature > 0, temperature=max(temperature, 0.01))
    gen = out[0][inputs["input_ids"].shape[1]:]
    return _processor.decode(gen, skip_special_tokens=True).strip()
# >>> END MODEL <<<


# ---------------------------------------------------------------------------
def build_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.concurrency import run_in_threadpool

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_ID}

    @app.post("/shutdown")
    def shutdown():
        _should_exit.set()
        return {"stopping": True}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        n = len(body.get("messages", []))
        mt = int(body.get("max_tokens", 512))
        print(f"[chat] request: {n} msgs, max_tokens={mt} — generating…", flush=True)
        t0 = time.time()
        try:
            # Run the blocking GPU generation OFF the event loop so /health and
            # other requests aren't starved (that made it look 'stuck').
            text = await run_in_threadpool(
                run_model, body.get("messages", []), mt,
                float(body.get("temperature", 0.2)))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[chat] ERROR after {time.time()-t0:.1f}s: {exc}", flush=True)
            return JSONResponse(status_code=500, content={"error": str(exc)})
        print(f"[chat] done in {time.time()-t0:.1f}s -> {text[:120]!r}", flush=True)
        return {
            "id": "chatcmpl-agentlab", "object": "chat.completion",
            "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
        }

    return app


def start_tunnel() -> str:
    """Run cloudflared and return the public https URL it prints."""
    proc = subprocess.Popen(
        ["./cloudflared", "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = None
    for line in proc.stdout:  # type: ignore[union-attr]
        print("[cloudflared]", line.strip(), flush=True)
        m = re.search(r"https://[-\w.]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            break
    if not url:
        raise RuntimeError("cloudflared did not return a URL")
    return url


def publish_url(url: str):
    """Publish the tunnel URL to the link dataset so the app finds it. Uses the
    Kaggle API with the baked-in credentials (this is your own private kernel)."""
    print("PUBLIC URL:", url, flush=True)
    if not (LINK_DATASET and KUSER and KKEY):
        # No auto-publish configured — the app can still use KAGGLE_ENDPOINT_URL.
        print("AGENTLAB_ENDPOINT=", url, flush=True)
        return
    os.environ["KAGGLE_USERNAME"] = KUSER
    os.environ["KAGGLE_KEY"] = KKEY
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    slug = LINK_DATASET.split("/")[-1]
    d = Path("/kaggle/working/agentlab_link")
    d.mkdir(parents=True, exist_ok=True)
    (d / "endpoint.json").write_text(json.dumps({"status": "READY", "url": url, "ts": time.time()}))
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": LINK_DATASET, "licenses": [{"name": "CC0-1.0"}]}))
    try:
        api.dataset_create_version(str(d), version_notes="ready", dir_mode="zip")
    except Exception:
        api.dataset_create_new(str(d), dir_mode="zip", public=False)
    print("published endpoint to", LINK_DATASET, flush=True)


def main():
    install_deps()
    load_model()

    import uvicorn
    app = build_app()
    threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning"),
        daemon=True).start()
    time.sleep(4)

    url = start_tunnel()
    print("PUBLIC URL:", url, flush=True)
    publish_url(url)
    print("READY", flush=True)

    # keep the kernel alive until the app asks us to stop
    while not _should_exit.is_set():
        time.sleep(2)
    print("shutdown requested — exiting", flush=True)


if __name__ == "__main__":
    main()
