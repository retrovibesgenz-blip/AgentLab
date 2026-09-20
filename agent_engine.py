"""
AgentLab — Python Agentic Execution Engine
===========================================

Responsibilities
----------------
1. Talk to an OpenAI-compatible LLM (ByNara / NaraRouter / OpenAI) to plan the
   next screen action from a screenshot + goal.
2. Execute OS-level actions (move / click / type / scroll / drag) via pyautogui,
   serialised behind a single physical-input lock so multiple agents never fight
   over the one real mouse.
3. Maintain a *virtual* cursor per agent and stream its interpolated position,
   action state and logs to the Electron overlay over WebSockets.

The frontend renders one distinctly-coloured cursor per agent from this telemetry.

Run:  python agent_engine.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import mss
import pyautogui
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

try:
    import websockets
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'websockets'. Run: pip install -r requirements.txt") from exc


# Silence the cosmetic "mss.mss is deprecated" notice; the call still works.
warnings.filterwarnings("ignore", message=".*mss.mss is deprecated.*")


def _enable_dpi_awareness() -> None:
    """Make this process DPI-aware so screenshots (mss), UI-element rects (UIA),
    and mouse clicks (pyautogui) ALL use the same physical pixels. Without this,
    Windows display scaling (e.g. 125%) makes every click land ~25% off target.
    Must run before mss/pyautogui first read the screen size."""
    if sys.platform != "win32":
        return
    import ctypes
    for attempt in (
        lambda: ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)),  # per-monitor v2
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),  # per-monitor
        lambda: ctypes.windll.user32.SetProcessDPIAware(),       # system aware
    ):
        try:
            attempt()
            return
        except Exception:
            continue


_enable_dpi_awareness()

# Windows consoles default to cp1252, which crashes on any non-ASCII we print
# (model thoughts, emoji, arrows, accents). Force UTF-8 with safe replacement so
# a stray character can never take down the engine.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()


@dataclass
class Config:
    # Z.AI (GLM) is the default provider — you only need ZAI_API_KEY in .env.
    # GEMINI_API_KEY / BYNARA_API_KEY still work as fallbacks (any
    # OpenAI-compatible endpoint does; just set the matching *_BASE_URL).
    api_key: str = (os.getenv("ZAI_API_KEY") or os.getenv("GEMINI_API_KEY")
                    or os.getenv("BYNARA_API_KEY", ""))
    base_url: str = (os.getenv("ZAI_BASE_URL") or os.getenv("GEMINI_BASE_URL")
                     or os.getenv("BYNARA_BASE_URL")
                     or "https://api.z.ai/api/paas/v4/")
    model: str = os.getenv("AGENT_MODEL", "glm-4.6v-flash")
    vision_enabled: bool = os.getenv("VISION_ENABLED", "true").lower() == "true"
    ws_host: str = os.getenv("WS_HOST", "127.0.0.1")
    ws_port: int = int(os.getenv("WS_PORT", "8765"))
    max_steps: int = int(os.getenv("MAX_STEPS", "40"))
    step_delay: float = float(os.getenv("STEP_DELAY", "0.4"))
    move_steps: int = int(os.getenv("MOVE_STEPS", "24"))
    failsafe: bool = os.getenv("FAILSAFE", "true").lower() == "true"
    # "physical" = the agent uses the REAL mouse + keyboard, exactly like a human
    # — works in EVERY app (browsers, Spotify, games, drag, hotkeys). It shares
    # your cursor while running. "background" = its own cursor via Win32 messages,
    # but can't drive Chromium apps / games (limited).
    input_mode: str = os.getenv("INPUT_MODE", "physical").lower()
    # Read real clickable UI elements (Windows UI Automation) and let the model
    # pick one by number. Off by default — the coordinate grid below is used
    # instead. Set USE_UI_ELEMENTS=true to turn element-reading back on.
    use_ui_elements: bool = os.getenv("USE_UI_ELEMENTS", "false").lower() == "true"
    # Overlay a coordinate grid/ruler on the screenshot the AI sees (your real
    # screen is untouched) so raw x/y clicks — used where no element is readable
    # (game canvases, custom UIs, images) — are far more accurate.
    use_grid: bool = os.getenv("USE_GRID", "true").lower() == "true"
    # ElevenLabs voice chat: talk to the agent, and it talks back. Key -> .env.
    el_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    el_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # "George" — free-tier premade
    el_tts_model: str = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
    el_stt_model: str = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")
    # Kaggle GPU backup — run an open vision model on your own Kaggle notebook.
    kaggle_username: str = os.getenv("KAGGLE_USERNAME", "")
    kaggle_key: str = os.getenv("KAGGLE_KEY", "")
    kaggle_model: str = os.getenv("KAGGLE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    kaggle_endpoint_url: str = os.getenv("KAGGLE_ENDPOINT_URL", "")   # manual/stable URL (skips auto-start)
    kaggle_kernel_slug: str = os.getenv("KAGGLE_KERNEL_SLUG", "")     # "<user>/agentlab-worker"
    kaggle_link_dataset: str = os.getenv("KAGGLE_LINK_DATASET", "")   # "<user>/agentlab-endpoint"
    kaggle_boot_timeout: int = int(os.getenv("KAGGLE_BOOT_TIMEOUT", "900"))  # seconds to wait for the model
    hf_token: str = os.getenv("HF_TOKEN", "")  # optional, baked into the Kaggle worker
    # Ollama — run open models locally via the Ollama app (ollama.com). It exposes
    # an OpenAI-compatible API, so we just point the planner at it.
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2-vision")


CFG = Config()

# pyautogui safety: slamming the mouse into a corner raises FailSafeException.
pyautogui.FAILSAFE = CFG.failsafe
pyautogui.PAUSE = 0.0  # we pace things ourselves


# Distinct per-agent cursor colours. The app chrome stays black & white; each
# agent's orb cursor is tinted with one of these so you can track them on screen.
AGENT_COLORS = [
    "#00E5FF",  # cyan
    "#FF5252",  # red
    "#69F0AE",  # green
    "#B388FF",  # purple
    "#FFAB40",  # orange
    "#FFEB3B",  # yellow
    "#FF4081",  # pink
    "#40C4FF",  # blue
]


class AgentState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    ACTING = "acting"
    PAUSED = "paused"
    DONE = "done"
    ERROR = "error"
    KILLED = "killed"


# ---------------------------------------------------------------------------
# Screen capture / geometry helpers
# ---------------------------------------------------------------------------
class Screen:
    """Fast primary-monitor capture with normalized-coordinate helpers."""

    def __init__(self) -> None:
        self._sct = mss.mss()
        # monitors[0] is the virtual "all monitors" union; [1:] are the physical
        # screens in no guaranteed order. We MUST capture the PRIMARY monitor —
        # the same one the Electron overlay is drawn on — otherwise we'd screenshot
        # (and click) an empty secondary screen. Allow an override via MONITOR env.
        phys = self._sct.monitors[1:]
        override = os.getenv("MONITOR", "").strip()
        if override.isdigit() and 1 <= int(override) < len(self._sct.monitors):
            self.monitor = self._sct.monitors[int(override)]
        else:
            self.monitor = (
                next((m for m in phys if m.get("is_primary")), None)
                or next((m for m in phys if m["left"] == 0 and m["top"] == 0), None)
                or phys[0]
            )
        self.width = self.monitor["width"]
        self.height = self.monitor["height"]

    def grab(self, max_side: int = 1280) -> tuple[Image.Image, float]:
        """Return a downscaled RGB screenshot and the scale factor applied."""
        raw = self._sct.grab(self.monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        scale = 1.0
        longest = max(img.width, img.height)
        if longest > max_side:
            scale = max_side / longest
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        return img, scale

    @staticmethod
    def to_base64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def norm_to_pixels(self, nx: float, ny: float) -> tuple[int, int]:
        """Normalized (0..1) -> absolute screen pixels on the primary monitor."""
        px = self.monitor["left"] + int(max(0.0, min(1.0, nx)) * self.width)
        py = self.monitor["top"] + int(max(0.0, min(1.0, ny)) * self.height)
        return self.clamp(px, py)

    def clamp(self, px: int, py: int) -> tuple[int, int]:
        """Keep a click a couple px inside the monitor so it never lands exactly
        on a corner (which would trip pyautogui's failsafe and abort the agent)."""
        left, top = self.monitor["left"], self.monitor["top"]
        px = max(left + 2, min(left + self.width - 2, int(px)))
        py = max(top + 2, min(top + self.height - 2, int(py)))
        return px, py


# ---------------------------------------------------------------------------
# Windows background input — the agent's OWN cursor
# ---------------------------------------------------------------------------
# Clicks/types by posting messages directly to the target window, so the agent
# never moves or steals the user's real mouse. Best-effort: some Chromium /
# Electron / UWP windows (and full-screen games) ignore synthetic messages —
# for those, switch INPUT_MODE=physical.
class WinInput:
    WM_MOUSEMOVE = 0x0200
    WM_LBUTTONDOWN, WM_LBUTTONUP, WM_LBUTTONDBLCLK = 0x0201, 0x0202, 0x0203
    WM_RBUTTONDOWN, WM_RBUTTONUP = 0x0204, 0x0205
    WM_MOUSEWHEEL = 0x020A
    WM_KEYDOWN, WM_KEYUP, WM_CHAR = 0x0100, 0x0101, 0x0102
    MK_LBUTTON = 0x0001

    # single, non-modifier keys we can post reliably in the background
    VK = {
        "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
        "space": 0x20, "backspace": 0x08, "delete": 0x2E, "del": 0x2E,
        "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
        "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    }

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        self.wintypes = wintypes
        self.u32 = ctypes.windll.user32
        self.k32 = ctypes.windll.kernel32
        self.u32.WindowFromPoint.restype = wintypes.HWND
        self.u32.WindowFromPoint.argtypes = [wintypes.POINT]
        self.u32.ScreenToClient.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.u32.PostMessageW.argtypes = [
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self.u32.GetForegroundWindow.restype = wintypes.HWND
        self.u32.GetDesktopWindow.restype = wintypes.HWND
        self.u32.GetAncestor.restype = wintypes.HWND
        self.u32.ChildWindowFromPointEx.restype = wintypes.HWND
        self.u32.ChildWindowFromPointEx.argtypes = [
            wintypes.HWND, wintypes.POINT, wintypes.UINT]

    # CWP flags: skip invisible | disabled | transparent (click-through).
    _CWP_SKIP = 0x1 | 0x2 | 0x4

    def _win_at(self, x: int, y: int):
        """Deepest real window at a screen point, SKIPPING transparent click-
        through windows like our own fullscreen overlay."""
        desktop = self.u32.GetDesktopWindow()
        pt = self.wintypes.POINT(x, y)
        hwnd = self.u32.ChildWindowFromPointEx(desktop, pt, self._CWP_SKIP)
        if not hwnd:
            return self.u32.WindowFromPoint(self.wintypes.POINT(x, y))
        while True:  # descend into child controls for a precise target
            cpt = self.wintypes.POINT(x, y)
            self.u32.ScreenToClient(hwnd, self.ctypes.byref(cpt))
            child = self.u32.ChildWindowFromPointEx(hwnd, cpt, self._CWP_SKIP)
            if not child or child == hwnd:
                break
            hwnd = child
        return hwnd

    def _root_class(self, hwnd) -> str:
        GA_ROOT = 2
        root = self.u32.GetAncestor(hwnd, GA_ROOT) or hwnd
        buf = self.ctypes.create_unicode_buffer(256)
        self.u32.GetClassNameW(root, buf, 256)
        return buf.value or ""

    def needs_physical(self, x: int, y: int) -> bool:
        """Chromium/Electron windows (Spotify, Chrome, Discord, VS Code, …) drop
        synthetic PostMessage clicks — those need a real (physical) click."""
        hwnd = self._win_at(x, y)
        if not hwnd:
            return False
        return self._root_class(hwnd).startswith("Chrome_WidgetWin")

    def focused_is_chromium(self) -> bool:
        hwnd = self._focused_hwnd()
        return bool(hwnd) and self._root_class(hwnd).startswith("Chrome_WidgetWin")

    def _client_lparam(self, hwnd, x: int, y: int) -> int:
        pt = self.wintypes.POINT(x, y)
        self.u32.ScreenToClient(hwnd, self.ctypes.byref(pt))
        return ((pt.y & 0xFFFF) << 16) | (pt.x & 0xFFFF)

    def _focused_hwnd(self):
        """The control with keyboard focus in the foreground window."""
        fg = self.u32.GetForegroundWindow()
        if not fg:
            return None
        tid = self.u32.GetWindowThreadProcessId(fg, None)
        cur = self.k32.GetCurrentThreadId()
        self.u32.AttachThreadInput(cur, tid, True)
        try:
            focus = self.u32.GetFocus()
        finally:
            self.u32.AttachThreadInput(cur, tid, False)
        return focus or fg

    def click(self, x: int, y: int, button: str = "left", double: bool = False) -> None:
        hwnd = self._win_at(x, y)
        if not hwnd:
            return
        lp = self._client_lparam(hwnd, x, y)
        self.u32.PostMessageW(hwnd, self.WM_MOUSEMOVE, 0, lp)
        if button == "right":
            self.u32.PostMessageW(hwnd, self.WM_RBUTTONDOWN, 0, lp)
            self.u32.PostMessageW(hwnd, self.WM_RBUTTONUP, 0, lp)
            return
        self.u32.PostMessageW(hwnd, self.WM_LBUTTONDOWN, self.MK_LBUTTON, lp)
        self.u32.PostMessageW(hwnd, self.WM_LBUTTONUP, 0, lp)
        if double:
            self.u32.PostMessageW(hwnd, self.WM_LBUTTONDBLCLK, self.MK_LBUTTON, lp)
            self.u32.PostMessageW(hwnd, self.WM_LBUTTONUP, 0, lp)

    def type_text(self, text: str) -> None:
        hwnd = self._focused_hwnd()
        if not hwnd:
            return
        for ch in text:
            self.u32.PostMessageW(hwnd, self.WM_CHAR, ord(ch), 0)

    def press(self, key_combo: str) -> None:
        """Post a single non-modifier key (enter/tab/arrows/…). Modifier combos
        (ctrl+s etc.) can't be posted reliably in the background and are skipped."""
        parts = [k.strip().lower() for k in key_combo.split("+") if k.strip()]
        if len(parts) != 1:
            return  # combos unsupported in background mode
        vk = self.VK.get(parts[0])
        if vk is None:
            return
        hwnd = self._focused_hwnd()
        if not hwnd:
            return
        self.u32.PostMessageW(hwnd, self.WM_KEYDOWN, vk, 0)
        self.u32.PostMessageW(hwnd, self.WM_KEYUP, vk, 0)

    def scroll(self, x: int, y: int, amount: int) -> None:
        hwnd = self._win_at(x, y)
        if not hwnd:
            return
        wparam = (int(amount) * 120) << 16  # WHEEL_DELTA per notch
        lp = ((y & 0xFFFF) << 16) | (x & 0xFFFF)  # wheel uses SCREEN coords
        self.u32.PostMessageW(hwnd, self.WM_MOUSEWHEEL, wparam, lp)


# System-wide media keys — control the active player (Spotify, etc.) directly,
# with no window focus and no mouse. Works regardless of INPUT_MODE.
_MEDIA_VK = {
    "playpause": 0xB3, "play": 0xB3, "pause": 0xB3, "toggle": 0xB3,
    "next": 0xB0, "skip": 0xB0, "prev": 0xB1, "previous": 0xB1, "back": 0xB1,
    "stop": 0xB2,
}


def type_text_like_human(text: str) -> None:
    """Type arbitrary text into the focused field. Uses clipboard paste so ANY
    characters work (unicode, emoji, other languages), then restores the
    clipboard. Falls back to per-key typing if the clipboard is unavailable."""
    if not text:
        return
    try:
        import pyperclip
        try:
            prev = pyperclip.paste()
        except Exception:
            prev = None
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.08)
        if prev is not None:
            try:
                pyperclip.copy(prev)
            except Exception:
                pass
        return
    except Exception:
        pass
    pyautogui.typewrite(text, interval=0.02)


def find_green_play_button(img: "Image.Image") -> Optional[tuple[float, float]]:
    """Deterministically locate Spotify's green PLAY button (brand green #1ED760)
    by colour — no model guessing. Returns normalized (x,y) or None."""
    try:
        import numpy as np
        import cv2
    except Exception:
        return None
    rgb = np.array(img.convert("RGB"))
    r = rgb[:, :, 0].astype(int); g = rgb[:, :, 1].astype(int); b = rgb[:, :, 2].astype(int)
    mask = ((np.abs(r - 30) < 55) & (g > 165) & (np.abs(b - 96) < 70)).astype(np.uint8) * 255
    H, W = mask.shape
    n, _labels, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 300 or area > 12000:
            continue
        if not (0.6 <= (w / h if h else 0) <= 1.5):   # roughly circular
            continue
        if cent[i][1] > 0.6 * H:                       # button sits in the upper area
            continue
        if area / (w * h) < 0.55:                       # solid, filled circle
            continue
        if best is None or area > best[2]:
            best = (cent[i][0], cent[i][1], area)
    if best is None:
        return None
    return best[0] / W, best[1] / H


def open_url(url: str) -> tuple[bool, str]:
    """Open a website in the default browser. Adds https:// if missing."""
    u = (url or "").strip()
    if not u:
        return False, "empty url"
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "https://" + u
    try:
        if sys.platform == "win32":
            os.startfile(u)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", u])
        return True, f"opened {u}"
    except Exception as exc:
        return False, f"couldn't open url: {exc}"


def spotify_search(query: str) -> tuple[bool, str]:
    """Open Spotify straight to the search results for a song via the spotify:
    URI — reliable, no mouse. The agent then plays the top result."""
    from urllib.parse import quote
    q = query.strip()
    if not q:
        return False, "empty search query"
    uri = "spotify:search:" + quote(q)
    try:
        if sys.platform == "win32":
            os.startfile(uri)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", uri])
        return True, f"opened Spotify search for '{q}'"
    except Exception as exc:
        return False, f"couldn't open Spotify search: {exc}"


def send_media_key(command: str) -> bool:
    """Fire a global media key (play/pause/next/prev/stop). Windows only."""
    if sys.platform != "win32":
        return False
    vk = _MEDIA_VK.get(str(command).strip().lower())
    if vk is None:
        return False
    import ctypes
    u32 = ctypes.windll.user32
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP = 0x0001, 0x0002
    u32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY, 0)
    u32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
    return True


# Common apps whose executable name differs from how people say it. Used as a
# fallback so "chrome" / "word" / "vscode" resolve via Windows App Paths + PATH.
_APP_ALIASES = {
    "chrome": "chrome", "google chrome": "chrome",
    "edge": "msedge", "microsoft edge": "msedge",
    "firefox": "firefox", "brave": "brave",
    "word": "winword", "microsoft word": "winword",
    "excel": "excel", "powerpoint": "powerpnt", "outlook": "outlook",
    "vscode": "code", "vs code": "code", "visual studio code": "code",
    "notepad": "notepad", "calculator": "calc", "calc": "calc",
    "explorer": "explorer", "file explorer": "explorer",
    "cmd": "cmd", "terminal": "wt", "powershell": "powershell",
    "paint": "mspaint", "spotify": "spotify",
}


_UIA_ROLES = {
    "ButtonControl", "EditControl", "HyperlinkControl", "ListItemControl",
    "TabItemControl", "MenuItemControl", "ComboBoxControl", "CheckBoxControl",
    "RadioButtonControl", "SplitButtonControl", "TreeItemControl", "TextControl",
    "DataItemControl",
}


# Window titles to ignore when picking which window to read (our own UI).
_OWN_WINDOW_MARKERS = ("agentlab", "claude")


def _target_hwnd() -> int:
    """The window the agent should act on: the focused window, unless that's our
    own AgentLab panel — then the topmost real app window behind it. This stops
    the agent from reading its OWN buttons instead of Chrome/Spotify/etc."""
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    u.GetForegroundWindow.restype = wintypes.HWND

    def title(h) -> str:
        n = u.GetWindowTextLengthW(h)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(h, buf, n + 1)
        return buf.value or ""

    def usable(h) -> bool:
        if not h or not u.IsWindowVisible(h) or u.IsIconic(h):
            return False
        t = title(h).lower()
        if not t:
            return False
        return not any(m in t for m in _OWN_WINDOW_MARKERS)

    fg = u.GetForegroundWindow()
    if usable(fg):
        return fg
    h = u.GetTopWindow(0)          # walk top-of-z-order downward
    GW_HWNDNEXT = 2
    while h:
        if usable(h):
            return h
        h = u.GetWindow(h, GW_HWNDNEXT)
    return fg


def capture_ui_elements(monitor: dict, max_elements: int = 70) -> tuple[str, list[dict[str, Any]]]:
    """Read the REAL clickable controls of the TARGET app window via Windows UI
    Automation, with exact on-screen centers — so the model never has to guess
    pixel coordinates. Returns (window_title, elements); ([], "") if unavailable."""
    if sys.platform != "win32":
        return "", []
    try:
        import uiautomation as auto
    except Exception:
        return "", []
    try:
        hwnd = _target_hwnd()
        root = auto.ControlFromHandle(hwnd) if hwnd else auto.GetForegroundControl()
    except Exception:
        return "", []
    if not root:
        return "", []
    win_title = ""
    try:
        win_title = (root.Name or "").strip()
    except Exception:
        pass
    fg = root
    L, T, W, H = monitor["left"], monitor["top"], monitor["width"], monitor["height"]
    # Traverse deep enough for nested web/Electron UIs (Spotify's search bar sits
    # ~20 levels down) but cap node visits so it stays fast on huge trees.
    max_depth = int(os.getenv("UIA_DEPTH", "26"))
    max_nodes = int(os.getenv("UIA_MAX_NODES", "2600"))
    cands: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    nodes = 0
    try:
        for c, _depth in auto.WalkControl(fg, includeTop=True, maxDepth=max_depth):
            nodes += 1
            if nodes > max_nodes:
                break
            if c.ControlTypeName not in _UIA_ROLES:
                continue
            try:
                if not c.IsEnabled or c.IsOffscreen:
                    continue
                r = c.BoundingRectangle
                if r.width() <= 1 or r.height() <= 1 or r.width() > W or r.height() > H:
                    continue
                cx, cy = r.xcenter(), r.ycenter()
                nx, ny = (cx - L) / W, (cy - T) / H
                if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
                    continue
                name = (c.Name or "").strip().replace("\n", " ").replace("\r", " ")
                role = c.ControlTypeName[:-7]  # strip "Control"
                if role == "Text" and not name:
                    continue  # unlabeled static text is noise
                key = (round(cx / 8), round(cy / 8))
                if key in seen:
                    continue
                seen.add(key)
                cands.append({"role": role, "name": name[:44], "cx": cx, "cy": cy,
                              "nx": nx, "ny": ny})
            except Exception:
                continue
    except Exception:
        pass
    # Keep the most useful ones when there are more than we can mark: prefer
    # named, interactive controls. Then order top-to-bottom for tidy numbering.
    prio = {"Edit": 0, "ComboBox": 0, "Button": 1, "TabItem": 1, "MenuItem": 1,
            "CheckBox": 1, "RadioButton": 1, "SplitButton": 1, "Hyperlink": 2,
            "ListItem": 2, "DataItem": 2, "TreeItem": 2, "Text": 3}
    cands.sort(key=lambda e: (0 if e["name"] else 1, prio.get(e["role"], 2)))
    out = cands[:max_elements]
    out.sort(key=lambda e: (round(e["cy"] / 22), e["cx"]))
    return win_title, out


def draw_grid(img: "Image.Image", step: float = 0.1) -> "Image.Image":
    """Overlay a light coordinate ruler on the screenshot (AI-only). Numbers along
    the top edge are the x fraction (0..1); down the left edge are y. Lets the
    model read an accurate x/y for anything it must click without an element."""
    from PIL import ImageDraw, ImageFont
    base = img.convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    line = (0, 210, 255, 60)      # faint cyan gridlines
    label = (0, 225, 255, 235)
    n = max(2, int(round(1 / step)))
    for i in range(1, n):
        f = i * step
        x, y = int(f * w), int(f * h)
        d.line([(x, 0), (x, h)], fill=line, width=1)
        d.line([(0, y), (w, y)], fill=line, width=1)
        tag = f"{f:.1f}"
        d.text((x + 2, 1), tag, fill=label, font=font)          # top ruler
        d.text((1, y + 1), tag, fill=label, font=font)          # left ruler
    return Image.alpha_composite(base, overlay).convert("RGB")


def annotate_elements(img: "Image.Image", elements: list[dict[str, Any]]) -> "Image.Image":
    """Draw a numbered badge on the screenshot at each element (Set-of-Marks)."""
    from PIL import ImageDraw, ImageFont
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    try:
        font = ImageFont.truetype("arialbd.ttf", 13)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
    for i, e in enumerate(elements, 1):
        x, y = int(e["nx"] * w), int(e["ny"] * h)
        label = str(i)
        bw = 7 * len(label) + 8
        d.rectangle([x - bw // 2, y - 10, x + bw // 2, y + 10],
                    fill=(255, 0, 90), outline=(255, 255, 255))
        d.text((x - bw // 2 + 4, y - 8), label, fill=(255, 255, 255), font=font)
    return im


def elements_prompt(elements: list[dict[str, Any]], win_title: str = "") -> str:
    """Compact numbered list of the marked elements for the model."""
    if not elements:
        return ""
    lines = []
    for i, e in enumerate(elements, 1):
        nm = e["name"]
        lines.append(f'[{i}] {e["role"]}' + (f' "{nm}"' if nm else ""))
    head = (f"CLICKABLE ELEMENTS in the active window \"{win_title}\" " if win_title
            else "CLICKABLE ELEMENTS on screen ")
    note = ""
    if len(elements) < 12:
        note = ("\n(NOTE: this app exposes only a few elements — the list is likely "
                "INCOMPLETE. If your target isn't here, use a specialized action or "
                "click what you see; do NOT pick an unrelated number.)")
    return (head + "(each is marked with its number on the screenshot). Click one "
            'with {"type":"click_element","index":N}:\n' + "\n".join(lines) + note)


def launch_app(name: str) -> tuple[bool, str]:
    """Launch an installed app by (fuzzy) name. Tries the Start-menu AUMID first
    (Store + Start apps), then falls back to Start-Process by executable name,
    which resolves classic desktop apps (Chrome, etc.) via the App Paths registry
    and PATH — many of which do NOT appear in Get-StartApps."""
    safe = re.sub(r"[^\w .+-]", "", name).strip()
    if not safe:
        return False, "empty app name"
    low = safe.lower()

    # Build ordered, de-duplicated executable candidates for the fallback.
    cands: list[str] = []
    if low in _APP_ALIASES:
        cands.append(_APP_ALIASES[low])
    cands += [safe, safe.replace(" ", ""), low.split()[0] if low.split() else safe]
    seen: set[str] = set()
    cands = [c for c in cands if c and not (c.lower() in seen or seen.add(c.lower()))]
    ps_cands = ",".join("'" + c.replace("'", "''") + "'" for c in cands)

    ps = (
        f"$q='{safe}'; "
        # 1) Start-menu app (Store or Start-listed desktop app) via its AUMID
        f"$a = Get-StartApps | Where-Object {{ $_.Name -like \"*$q*\" }} | Select-Object -First 1; "
        f"if ($a) {{ Start-Process \"shell:AppsFolder\\$($a.AppID)\"; Write-Output $a.Name; exit 0 }} "
        # 2) fall back to launching an executable by name (App Paths / PATH)
        f"foreach ($c in @({ps_cands})) {{ try {{ Start-Process $c -ErrorAction Stop; "
        f"Write-Output $c; exit 0 }} catch {{}} }} "
        f"exit 2"
    )
    no_window = 0x08000000 if sys.platform == "win32" else 0
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
             "-Command", ps],
            capture_output=True, text=True, timeout=25, creationflags=no_window,
        )
    except Exception as exc:
        return False, f"launch failed: {exc}"
    if res.returncode == 0:
        return True, (res.stdout.strip() or safe)
    if res.returncode == 2:
        return False, f"no installed app matching '{safe}'"
    return False, (res.stderr.strip() or f"launch failed (code {res.returncode})")


# ---------------------------------------------------------------------------
# ElevenLabs voice — speak to the agent, and it speaks back
# ---------------------------------------------------------------------------
class Voice:
    """Text-to-speech + speech-to-text via ElevenLabs. Blocking HTTP — the
    caller runs these in a thread. Uses httpx (already an openai dependency)."""

    TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        key = (cfg.el_api_key or "").strip()
        self.enabled = bool(key) and not key.startswith("PASTE_")
        self.last_error = ""

    def _friendly(self, exc: Exception, kind: str) -> str:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            try:
                detail = exc.response.json().get("detail", {})
                msg = detail.get("message") if isinstance(detail, dict) else str(detail)
            except Exception:
                msg = ""
            if code == 401:
                return (f"{kind}: your ElevenLabs API key is missing a required "
                        f"permission ({'speech_to_text' if kind == 'STT' else 'text_to_speech'}). "
                        f"Enable it on the key at elevenlabs.io -> Settings -> API Keys.")
            if code == 402:
                # Could be no credits OR a paid-only voice — show the real reason.
                if msg and "voice" in msg.lower():
                    return (f"{kind}: this voice needs a paid ElevenLabs plan. Use a free "
                            f"premade voice like George (JBFqnCBsd6RMkjVDRZzb) in ELEVENLABS_VOICE_ID.")
                return f"{kind} (402): {msg or 'out of credits — top up or wait for the monthly reset'}"
            return f"{kind} error {code}: {msg or exc}"
        return f"{kind} error: {exc}"

    def tts(self, text: str) -> Optional[bytes]:
        """Synthesize speech -> mp3 bytes, or None on failure/disabled."""
        if not self.enabled or not text.strip():
            return None
        import httpx
        try:
            r = httpx.post(
                self.TTS_URL.format(vid=self.cfg.el_voice_id),
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": self.cfg.el_api_key,
                         "content-type": "application/json"},
                json={
                    "text": text,
                    "model_id": self.cfg.el_tts_model,
                    # tuned for a warm, natural, conversational delivery
                    "voice_settings": {"stability": 0.45, "similarity_boost": 0.75,
                                       "style": 0.30, "use_speaker_boost": True},
                },
                timeout=30,
            )
            r.raise_for_status()
            self.last_error = ""
            return r.content
        except Exception as exc:  # network / auth / quota
            self.last_error = self._friendly(exc, "TTS")
            print(f"  ElevenLabs {self.last_error}")
            return None

    def stt(self, audio: bytes, mime: str = "audio/webm") -> Optional[str]:
        """Transcribe recorded audio -> text, or None on failure/disabled."""
        if not self.enabled or not audio:
            return None
        import httpx
        try:
            r = httpx.post(
                self.STT_URL,
                headers={"xi-api-key": self.cfg.el_api_key},
                data={"model_id": self.cfg.el_stt_model},
                files={"file": ("speech.webm", audio, mime)},
                timeout=60,
            )
            r.raise_for_status()
            self.last_error = ""
            return ((r.json() or {}).get("text") or "").strip()
        except Exception as exc:
            self.last_error = self._friendly(exc, "STT")
            print(f"  ElevenLabs {self.last_error}")
            return None


# ---------------------------------------------------------------------------
# LLM planner (OpenAI-compatible)
# ---------------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = """You are an autonomous computer-control agent. You see the user's screen
and issue ONE action at a time to accomplish a goal.

Coordinates are ALWAYS normalized floats in [0,1]:
  x = fraction from the left edge, y = fraction from the top edge.
The screen origin (0,0) is the top-left corner; (1,1) is the bottom-right.

The screenshot has a faint COORDINATE GRID overlaid for you: the numbers along
the TOP edge are x (0.1, 0.2, … left→right) and along the LEFT edge are y
(0.1, 0.2, … top→bottom). When you must click a spot with raw x/y (something with
no numbered element), read its position against these gridlines to get an accurate
value — e.g. a target halfway between the 0.4 and 0.5 vertical lines is x≈0.45.

Respond with STRICT JSON only (no markdown, no prose) in this schema:
{
  "thought": "short reasoning about the current screen and next step",
  "say": "OPTIONAL short spoken line — see the SPEAKING rule below",
  "action": {
    "type": "open_app | open_url | play_track | media | click_element | double_click_element | type_into | move | click | double_click | right_click | type | key | scroll | drag | wait | done",
    "app": "<the app named in the GOAL>",   // for open_app: e.g. Chrome, Notepad, Spotify
    "url": "example.com",  // for open_url: opens a website in the default browser
    "query": "<the song named in the GOAL>", // for play_track ONLY
    "command": "playpause", // for media: playpause | next | prev | stop
    "index": 0,          // for click_element/double_click_element/type_into: the [N] of a listed element
    "x": 0.0,            // for move/click/double_click/right_click/drag start
    "y": 0.0,
    "to_x": 0.0,         // for drag end only
    "to_y": 0.0,
    "text": "...",       // for type
    "keys": "ctrl+c",    // for key: any key or shortcut, e.g. enter, tab, esc, f5, ctrl+c, alt+tab, ctrl+shift+t
    "amount": -3,        // for scroll (negative = down, positive = up)
    "reason": "human-readable label of what this action does"
  }
}

You control the computer with a REAL mouse and keyboard, exactly like a person
sitting at it: you can click ANYTHING, type ANY text, use ANY keyboard shortcut,
drag, and scroll — in any app (browsers, Spotify, games, editors). Whatever a
human can do at this computer, you can do. The user watches your cursor move.

CRITICAL — DO EXACTLY WHAT THE GOAL SAYS:
- Read the user's GOAL and do THAT, nothing else. If the goal is "open Chrome",
  open Chrome. If it's "open Notepad", open Notepad. Use the exact app/website/
  song the user named.
- The examples in this prompt show JSON FORMAT ONLY. NEVER copy their literal
  values. Do not open Spotify, and do not play any song, unless the GOAL itself
  asks to play music. There is no default app and no default song.

Rules:
- Emit exactly one action per response.
- TO OPEN / LAUNCH AN APP, use "open_app" with the app name TAKEN FROM THE GOAL
  (e.g. if the goal says Chrome -> {"type":"open_app","app":"Chrome"}). This is
  more reliable than hunting the Start menu — do NOT click the Start button or
  taskbar to launch apps. After open_app, use "wait" once for the window, then
  interact visually.
- DO NOT open an app that the action history already says is open or "ALREADY
  OPEN". Opening it once is enough — move on. Trust the history: it reports what
  REALLY happened.
- If the GOAL was simply to open/launch an app, then AFTER the open_app step (the
  history will say "opened ..."), immediately emit {"type":"done"} — do not keep
  clicking. Opening the app IS the whole task.
- TO OPEN A WEBSITE / URL (e.g. "go to youtube.com", "open gmail"), use
  {"type":"open_url","url":"youtube.com"} — it opens in the default browser. Then
  interact with the page using the numbered elements.
- SPOTIFY: to PLAY a specific song, use ONE "play_track" action with the song as
  "query" (it opens Spotify, searches, and plays — most reliable). For pause/
  resume/skip use "media". For any OTHER Spotify interaction (open a playlist,
  click a button), use the numbered CLICKABLE ELEMENTS like any app. If the goal
  is NOT about music, never use play_track.
- TO CONTROL PLAYBACK of the CURRENT track (pause, resume, next, previous, stop),
  use "media" ({"type":"media","command":"playpause"}). One media action, then
  "done". Only when the goal is about controlling playback.
CLICKING — USE THE NUMBERED ELEMENTS, DON'T GUESS COORDINATES:
- When the message lists "CLICKABLE ELEMENTS", each real button/field/link/tab on
  screen has a NUMBER shown on the screenshot. To click something, find it in the
  list and use {"type":"click_element","index":N}. This clicks the EXACT element —
  it is far more accurate than guessing x/y, so ALWAYS prefer it.
- To type into a field, use {"type":"type_into","index":N,"text":"..."} — it
  clicks that field first, then types. (e.g. to search, type_into the search box.)
- Use double_click_element for items that need a double-click (e.g. a song row).
- Match the GOAL to an element by its NAME (e.g. Edit "Search", Button "Play").
  Only pick an index whose name actually matches what you want.
- If your target is NOT clearly in the numbered list (some apps, like Spotify,
  expose very few elements), do NOT pick a random number and do NOT guess — the
  cursor would fly to the wrong place. Instead use the right specialized action
  (open_app / play_track / media), or click the spot you can SEE with x/y.
- Use raw "click"/"type" with x/y coordinates only when the target has no number.
- You may use the "key" action for ANY key or shortcut a human would use —
  "enter", "tab", "esc", arrow keys, "f5", or combos like "ctrl+c", "ctrl+v",
  "ctrl+a", "alt+tab", "ctrl+shift+t". Use shortcuts when they're the fastest
  way (e.g. Ctrl+T for a new browser tab, Ctrl+L to focus the address bar).
- Move to a target before clicking it.
- Use "done" when the goal is achieved; put the outcome in "reason".
- If the screen is not what you expect, use "wait" and re-observe.
- Never invent coordinates outside [0,1].

SPEAKING (the "say" field) — this text is READ ALOUD to the user:
- "say" is the ONLY thing the user HEARS. Write it as a natural, friendly, spoken
  sentence — the way a helpful assistant would talk out loud. The voice system
  reads exactly this text, so make it clean speech (no coordinates, no JSON, no
  file paths, no technical jargon).
- ALWAYS include "say" on your FIRST action of a task (a brief confirmation of
  what you're about to do, referring to the ACTUAL goal, e.g. "Sure, opening
  Chrome now.") and ALWAYS include it on the "done" action (the result, e.g.
  "Chrome's open." or "Done, it's playing.").
- For routine middle steps, OMIT "say" — don't narrate every click. Speak about
  once at the start and once at the end.
- Keep it to ONE short sentence, and only about what the user actually asked."""


_CHAT_RE = re.compile(
    r"^\s*(hi|hey+|hello|yo|sup|thx|thanks?|thank you|ok(ay)?|cool|nice|great|lol|"
    r"good\s?(morning|afternoon|evening|night)|how\s?are\s?you|who\s?are\s?you|"
    r"what('?s| is)\s?up|what can you do|help|hmm+)\b", re.I)
_TASK_VERBS = ("open", "play", "click", "type", "go to", "search", "launch", "close",
               "write", "create", "download", "install", "send", "email", "scroll",
               "find", "start", "run", "pause", "next", "mute", "volume", "screenshot")


_CHAT_REPLY = ("Hi! I'm AgentLab — tell me something to do on your computer, like "
               "“open Chrome” or “play a song”, and I'll do it.")


def heuristic_classify(text: str) -> tuple[str, str]:
    """Fast task/chat/unknown guess. 'task' if an action verb is present, 'chat'
    for greetings/very short small talk, else 'unknown' (let the LLM decide)."""
    t = (text or "").strip()
    low = t.lower()
    if any(v in low for v in _TASK_VERBS):
        return "task", ""
    if _CHAT_RE.match(t) or len(t.split()) <= 3:
        return "chat", _CHAT_REPLY
    return "unknown", ""


def build_context(goal: str, history: list[str]) -> str:
    """The user-turn text shared by the cloud and local planners."""
    return (
        f"GOAL: {goal}\n\n"
        f"RECENT ACTIONS:\n" + ("\n".join(history[-8:]) if history else "(none yet)") +
        "\n\nDecide the next single action as JSON."
    )


class Planner:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.provider = "cloud"                 # cloud (Z.AI) | kaggle
        self.model = cfg.model
        self.client = AsyncOpenAI(api_key=cfg.api_key or "none", base_url=cfg.base_url)

    def use_kaggle(self, base_url: str) -> None:
        """Point the planner at the Kaggle worker's OpenAI-compatible endpoint."""
        self.provider = "kaggle"
        self.model = self.cfg.kaggle_model
        self.client = AsyncOpenAI(api_key="kaggle", base_url=base_url)

    def _timeout(self) -> float:
        # local GPU providers are slower than the cloud; give them room but a
        # finite cap so a hang becomes a visible error, not an endless spinner.
        return {"cloud": 60.0, "kaggle": 240.0, "ollama": 180.0}.get(self.provider, 120.0)

    def use_ollama(self, base_url: str, model: str) -> None:
        """Point the planner at a local Ollama server (OpenAI-compatible API)."""
        self.provider = "ollama"
        self.model = model
        self.client = AsyncOpenAI(api_key="ollama", base_url=base_url)

    def use_cloud(self) -> None:
        self.provider = "cloud"
        self.model = self.cfg.model
        self.client = AsyncOpenAI(api_key=self.cfg.api_key or "none", base_url=self.cfg.base_url)

    async def next_action(self, goal: str, history: list[str], screenshot_b64: Optional[str],
                          elements_text: str = "") -> dict[str, Any]:
        user_content: Any
        context = build_context(goal, history)
        if elements_text:
            context += "\n\n" + elements_text

        if self.cfg.vision_enabled and screenshot_b64:
            user_content = [
                {"type": "text", "text": context},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                },
            ]
        else:
            # Text-only fallback: the model plans blind from the goal + history.
            user_content = context + "\n\n(No screenshot available — reason from the goal and history.)"

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=500,
            timeout=self._timeout(),
        )
        # The "disable thinking" flag is a Z.AI extension — only the cloud
        # provider understands it; a local Kaggle server would reject it.
        if self.provider == "cloud":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        resp = await self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        return self._parse(text)

    async def locate(self, instruction: str, screenshot_b64: str) -> Optional[tuple[float, float]]:
        """One focused vision call: return normalized (x,y) of a described target,
        or None. Used for deterministic sub-steps like 'play the top result'."""
        content = [
            {"type": "text", "text": (
                instruction + "\n\nRespond with STRICT JSON only: "
                '{"x":0.0,"y":0.0} as fractions of the screen (0..1). '
                'If the target is not visible, respond {"x":-1,"y":-1}.')},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
        ]
        try:
            kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0, max_tokens=100, timeout=self._timeout(),
            )
            if self.provider == "cloud":
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            resp = await self.client.chat.completions.create(**kwargs)
            data = self._parse_obj(resp.choices[0].message.content or "")
            x, y = float(data.get("x", -1)), float(data.get("y", -1))
        except Exception:
            return None
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x, y
        return None

    async def classify(self, text: str) -> tuple[str, str]:
        """Decide whether a message is a computer TASK or just CHAT, and (for
        chat) produce a friendly reply. Returns (mode, reply)."""
        sys_p = (
            "You are AgentLab, a friendly assistant that can control the user's "
            "computer. Classify the user's message:\n"
            "- \"task\" = they want you to DO something on the PC (open/close apps, "
            "click, type, browse a site, play/pause music, write a file, search).\n"
            "- \"chat\" = greetings, thanks, small talk, or a question you can just "
            "answer in words (who are you, what can you do, tell me a joke, math).\n"
            "Examples: 'hi'->chat, 'how are you'->chat, 'what can you do'->chat, "
            "'tell me a joke'->chat, 'open chrome'->task, 'play softly on spotify'->"
            "task, 'search youtube for lofi'->task, 'type my email'->task.\n"
            'Respond with STRICT JSON only: {"mode":"task"|"chat","reply":"..."}. '
            "For chat, put a warm, brief, natural spoken-style reply in \"reply\". "
            'For a task, set reply to "".')
        try:
            kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user", "content": text}],
                temperature=0.4, max_tokens=160, timeout=self._timeout())
            if self.provider == "cloud":
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            resp = await self.client.chat.completions.create(**kwargs)
            data = self._parse_obj(resp.choices[0].message.content or "")
            mode = "chat" if str(data.get("mode", "")).lower() == "chat" else "task"
            return mode, str(data.get("reply", ""))
        except Exception:
            return "task", ""   # default to acting if unsure

    @staticmethod
    def _parse_obj(text: str) -> dict[str, Any]:
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        """Extract the first JSON object, tolerating code fences and stray prose."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"thought": "unparseable", "action": {"type": "wait", "reason": "no JSON returned"}}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"thought": "bad json", "action": {"type": "wait", "reason": "invalid JSON"}}
        if "action" not in data or not isinstance(data["action"], dict):
            return {"thought": data.get("thought", ""), "action": {"type": "wait", "reason": "no action"}}
        return data


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, coordinator: "Coordinator", goal: str, index: int) -> None:
        self.id = f"agent-{uuid.uuid4().hex[:6]}"
        self.name = f"Agent {index + 1}"
        self.color = AGENT_COLORS[index % len(AGENT_COLORS)]
        self.goal = goal
        self.coord = coordinator
        self.state = AgentState.IDLE
        self.history: list[str] = []
        self.step = 0
        self._elements: list[dict[str, Any]] = []  # UI elements from the last screenshot
        # Virtual cursor position in normalized coords; starts centre-screen.
        self.vx = 0.5
        self.vy = 0.5
        self._task: Optional[asyncio.Task] = None
        self._pause_evt = asyncio.Event()
        self._pause_evt.set()  # not paused initially
        self._killed = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def pause(self) -> None:
        if self.state not in (AgentState.DONE, AgentState.KILLED, AgentState.ERROR):
            self._pause_evt.clear()
            self._set_state(AgentState.PAUSED)

    def resume(self) -> None:
        if self.state == AgentState.PAUSED:
            self._pause_evt.set()
            self._set_state(AgentState.PLANNING)

    def kill(self) -> None:
        self._killed = True
        self._pause_evt.set()  # unblock any pause wait so the loop can exit
        if self._task:
            self._task.cancel()
        self._set_state(AgentState.KILLED)

    # -- main loop ---------------------------------------------------------
    async def _run(self) -> None:
        self._log(f"started with goal: {self.goal}")
        try:
            # If the user picked the Kaggle GPU, wait for it to finish booting the
            # model before planning (falls through if it never becomes ready).
            if self.coord.provider == "kaggle" and not self.coord.kaggle_ready:
                self._log("waiting for the Kaggle GPU to be ready…")
                waited = 0
                while (self.coord.provider == "kaggle" and not self.coord.kaggle_ready
                       and not self._killed and waited < self.coord.cfg.kaggle_boot_timeout):
                    await asyncio.sleep(2)
                    waited += 2

            while self.step < self.coord.cfg.max_steps and not self._killed:
                await self._pause_evt.wait()
                if self._killed:
                    break

                self.step += 1
                self._set_state(AgentState.PLANNING)

                # Local mode always needs a screenshot; cloud mode only if vision is on.
                shot_b64 = None
                elements_text = ""
                self._elements = []
                if self.coord.local_mode or self.coord.cfg.vision_enabled:
                    img, _ = await asyncio.to_thread(self.coord.screen.grab)
                    # Coordinate ruler for accurate raw x/y (AI-only; your screen
                    # is untouched). Drawn first so element badges sit on top.
                    if self.coord.cfg.use_grid:
                        img = await asyncio.to_thread(draw_grid, img)
                    # Read the real clickable elements and mark them on the shot so
                    # the model clicks by number instead of guessing coordinates.
                    if not self.coord.local_mode and self.coord.cfg.use_ui_elements:
                        win_title, els = await asyncio.to_thread(capture_ui_elements, self.coord.screen.monitor)
                        if els:
                            self._elements = els
                            img = await asyncio.to_thread(annotate_elements, img, els)
                            elements_text = elements_prompt(els, win_title)
                            self._log(f"reading {len(els)} elements from '{win_title}'")
                    shot_b64 = await asyncio.to_thread(Screen.to_base64, img)

                try:
                    if self.coord.local_mode:
                        plan = await self.coord.request_local_plan(self.goal, self.history, shot_b64)
                    else:
                        plan = await self.coord.planner.next_action(self.goal, self.history, shot_b64, elements_text)
                except Exception as exc:  # network / API / local failures
                    self._log(self._explain_api_error(exc))
                    self._set_state(AgentState.ERROR)
                    return

                action = plan.get("action", {})
                thought = plan.get("thought", "")
                say = (plan.get("say") or "").strip()
                if thought:
                    self._log(f"think: {thought}")
                # Speak the AI's chosen line WHILE it works (fire-and-forget, so
                # the action loop never blocks on audio). Only "say" is voiced.
                if say:
                    self._log(f"say: {say}")
                    asyncio.create_task(self.coord.speak(say, self))

                self._set_state(AgentState.ACTING)
                try:
                    done, result = await self._execute(action)
                except pyautogui.FailSafeException:
                    self._log("stopped — mouse reached a screen corner (failsafe)")
                    self._set_state(AgentState.KILLED)
                    return
                except Exception as exc:
                    self._log(f"couldn't do that action ({type(exc).__name__}); re-observing")
                    result = f"error: {exc}"
                    done = False
                # Record the REAL outcome (not the model's guess) so it doesn't
                # repeat itself — e.g. it learns Spotify is already open.
                self.history.append(f"step {self.step}: {action.get('type')} → {result}")

                if done:
                    self._set_state(AgentState.DONE)
                    self._log(f"goal complete: {action.get('reason', '')}")
                    if not say:  # make sure completion is heard in voice mode
                        asyncio.create_task(self.coord.speak(action.get("reason") or "All done.", self))
                    return

                await asyncio.sleep(self.coord.cfg.step_delay)

            if not self._killed:
                self._set_state(AgentState.DONE)
                self._log("max steps reached — stopping")
        except asyncio.CancelledError:
            self._log("cancelled")
        except pyautogui.FailSafeException:
            self._log("FAILSAFE triggered — mouse hit a corner")
            self._set_state(AgentState.KILLED)

    def _explain_api_error(self, exc: Exception) -> str:
        """Turn raw SDK exceptions into a message that says what to fix."""
        from openai import APIConnectionError, AuthenticationError, NotFoundError
        base = self.coord.cfg.base_url
        if isinstance(exc, APIConnectionError):
            return (f"can't reach the model API at {base} — no internet or endpoint down.")
        if isinstance(exc, AuthenticationError):
            return ("the model API rejected the API key. Put a valid ZAI_API_KEY "
                    "(from z.ai/manage-apikey/apikey-list) into .env.")
        if isinstance(exc, NotFoundError):
            return (f"model not found ({self.coord.cfg.model}) — check AGENT_MODEL in .env.")
        if isinstance(exc, RuntimeError):
            return str(exc)   # local-mode messages are already user-friendly
        return f"planner error: {type(exc).__name__}: {exc}"

    # -- action execution --------------------------------------------------
    async def _execute(self, action: dict[str, Any]) -> tuple[bool, str]:
        """Run one action. Returns (done, outcome) — outcome is the REAL result,
        fed back into history so the agent knows what actually happened."""
        atype = str(action.get("type", "wait")).lower()
        reason = action.get("reason", atype)
        self._broadcast_action(atype, reason)

        if atype == "done":
            return True, "goal complete"
        if atype == "wait":
            await asyncio.sleep(1.0)
            return False, "waited and re-observed"
        if atype == "open_app":
            name = str(action.get("app") or action.get("text") or "").strip()
            key = name.lower()
            if key in self.coord.launched_apps:
                msg = f"{name} is ALREADY OPEN — do not open it again; interact with it"
                self._log(msg)
                return False, msg
            ok, detail = await asyncio.to_thread(launch_app, name)
            if ok:
                self.coord.launched_apps.add(key)
                msg = f"opened {detail} (now on screen — do not open again)"
            else:
                msg = f"couldn't open '{name}' — {detail}"
            self._log(msg)
            return False, msg
        if atype == "open_url":
            url = str(action.get("url") or action.get("text") or "").strip()
            ok, detail = await asyncio.to_thread(open_url, url)
            self._log(detail)
            return False, detail
        if atype == "media":
            cmd = str(action.get("command") or action.get("keys") or "playpause").strip()
            ok = await asyncio.to_thread(send_media_key, cmd)
            msg = f"sent media '{cmd}'" if ok else f"unknown media command '{cmd}'"
            self._log(msg)
            return False, msg
        if atype in ("click_element", "double_click_element", "type_into"):
            try:
                idx = int(action.get("index", 0))
            except (TypeError, ValueError):
                idx = 0
            els = self._elements or []
            if not (1 <= idx <= len(els)):
                return False, f"no element #{idx} on screen"
            e = els[idx - 1]
            await self._glide_to(e["nx"], e["ny"])
            async with self.coord.input_lock:
                self.coord.active_agent = self.id
                self._broadcast_cursor(active=True)
                px, py = self.coord.screen.clamp(int(e["cx"]), int(e["cy"]))
                self.vx, self.vy = e["nx"], e["ny"]
                await asyncio.to_thread(pyautogui.moveTo, px, py, 0.35, pyautogui.easeInOutQuad)
                if atype == "double_click_element":
                    await asyncio.to_thread(pyautogui.doubleClick)
                else:
                    await asyncio.to_thread(pyautogui.click)
                if atype == "type_into":
                    await asyncio.sleep(0.15)
                    await asyncio.to_thread(type_text_like_human, str(action.get("text", "")))
                self.coord.active_agent = None
                self._broadcast_cursor(active=False)
            label = f'{e["role"]} "{e["name"]}"' if e["name"] else e["role"]
            return False, f"{atype.replace('_', ' ')} {label}"
        if atype == "play_track":
            query = str(action.get("query") or action.get("text") or action.get("app") or "").strip()
            ok, detail = await asyncio.to_thread(spotify_search, query)
            self.coord.launched_apps.add("spotify")
            if not ok:
                self._log(detail)
                return True, detail  # nothing more we can do; stop cleanly
            self._log(f"{detail} — playing…")
            # Poll until the results render, then click the top result's PLAY
            # button. Detection is by colour (deterministic) with a model-vision
            # fallback. Self-contained: playing a song is ONE step, no loop.
            pt = None
            for attempt in range(8):
                await asyncio.sleep(1.2)
                # full-resolution grab for precise colour detection
                img, _ = await asyncio.to_thread(self.coord.screen.grab, 100000)
                pt = await asyncio.to_thread(find_green_play_button, img)
                if pt:
                    break
                if attempt >= 3:  # colour miss after a few tries → ask the model
                    shot = await asyncio.to_thread(Screen.to_base64, img)
                    pt = await self.coord.planner.locate(
                        "Spotify search results. Give the green round PLAY button of "
                        "the TOP RESULT card, else the center of the first song row.", shot)
                    if pt:
                        break
            if pt is None:
                msg = f"searched Spotify for '{query}' but the results didn't load in time"
                self._log(msg)
                return True, msg  # stop — do not loop retrying
            self.vx, self.vy = pt
            await self._glide_to(pt[0], pt[1])
            async with self.coord.input_lock:
                self.coord.active_agent = self.id
                self._broadcast_cursor(active=True)
                px, py = self.coord.screen.norm_to_pixels(pt[0], pt[1])
                await asyncio.to_thread(pyautogui.moveTo, px, py, 0.4, pyautogui.easeInOutQuad)
                await asyncio.to_thread(pyautogui.click)
                self.coord.active_agent = None
                self._broadcast_cursor(active=False)
            msg = f"playing '{query}' on Spotify"
            self._log(msg)
            return True, msg  # goal achieved — one step, done

        # Movement-bearing actions animate the virtual cursor first.
        if atype in ("move", "click", "double_click", "right_click", "drag"):
            tx = float(action.get("x", self.vx))
            ty = float(action.get("y", self.vy))
            await self._glide_to(tx, ty)

        bkg = self.coord.win_input  # the agent's own cursor, if available

        # Physical actions share ONE input lock so agents don't collide.
        async with self.coord.input_lock:
            self.coord.active_agent = self.id
            self._broadcast_cursor(active=True)
            px, py = self.coord.screen.norm_to_pixels(self.vx, self.vy)

            if bkg is not None:
                # Background mode: post messages to the target window; the real
                # mouse is never touched. The on-screen orb IS the agent cursor.
                # EXCEPTION: Chromium/Electron windows (Spotify!) ignore synthetic
                # clicks, so for those we do a quick real click instead.
                tween = pyautogui.easeInOutQuad
                if atype == "move":
                    pass  # orb already glided; nothing physical to do
                elif atype in ("click", "double_click", "right_click"):
                    if bkg.needs_physical(px, py):
                        await asyncio.to_thread(pyautogui.moveTo, px, py, 0.35, tween)
                        if atype == "click":
                            await asyncio.to_thread(pyautogui.click)
                        elif atype == "double_click":
                            await asyncio.to_thread(pyautogui.doubleClick)
                        else:
                            await asyncio.to_thread(pyautogui.rightClick)
                    else:
                        btn = "right" if atype == "right_click" else "left"
                        await asyncio.to_thread(bkg.click, px, py, btn, atype == "double_click")
                elif atype == "type":
                    text = str(action.get("text", ""))
                    if bkg.focused_is_chromium():
                        await asyncio.to_thread(pyautogui.typewrite, text, 0.02)
                    else:
                        await asyncio.to_thread(bkg.type_text, text)
                elif atype == "key":
                    keys = str(action.get("keys", ""))
                    if bkg.focused_is_chromium():
                        parts = [k.strip() for k in keys.split("+") if k.strip()]
                        if parts:
                            await asyncio.to_thread(pyautogui.hotkey, *parts)
                    else:
                        await asyncio.to_thread(bkg.press, keys)
                elif atype == "scroll":
                    await asyncio.to_thread(bkg.scroll, px, py, int(action.get("amount", -3)))
                elif atype == "drag":
                    dx = float(action.get("to_x", self.vx))
                    dy = float(action.get("to_y", self.vy))
                    self.vx, self.vy = dx, dy  # drag is unsupported in background mode
            else:
                # Physical mode: visibly glide the REAL Windows cursor and act.
                glide = 0.6
                tween = pyautogui.easeInOutQuad
                if atype == "move":
                    await asyncio.to_thread(pyautogui.moveTo, px, py, glide, tween)
                elif atype == "click":
                    await asyncio.to_thread(pyautogui.moveTo, px, py, glide, tween)
                    await asyncio.to_thread(pyautogui.click)
                elif atype == "double_click":
                    await asyncio.to_thread(pyautogui.moveTo, px, py, glide, tween)
                    await asyncio.to_thread(pyautogui.doubleClick)
                elif atype == "right_click":
                    await asyncio.to_thread(pyautogui.moveTo, px, py, glide, tween)
                    await asyncio.to_thread(pyautogui.rightClick)
                elif atype == "type":
                    await asyncio.to_thread(type_text_like_human, str(action.get("text", "")))
                elif atype == "key":
                    keys = [k.strip().lower() for k in str(action.get("keys", "")).split("+") if k.strip()]
                    if keys:
                        await asyncio.to_thread(pyautogui.hotkey, *keys)
                elif atype == "scroll":
                    amount = int(action.get("amount", -3))
                    await asyncio.to_thread(pyautogui.scroll, amount * 100)
                elif atype == "drag":
                    dx = float(action.get("to_x", self.vx))
                    dy = float(action.get("to_y", self.vy))
                    epx, epy = self.coord.screen.norm_to_pixels(dx, dy)
                    await asyncio.to_thread(pyautogui.moveTo, px, py, 0.1)
                    await asyncio.to_thread(pyautogui.dragTo, epx, epy, 0.4, button="left")
                    self.vx, self.vy = dx, dy

            self.coord.active_agent = None
            self._broadcast_cursor(active=False)

        where = ""
        if atype in ("move", "click", "double_click", "right_click"):
            where = f" at ({self.vx:.2f},{self.vy:.2f})"
        via = "own cursor" if bkg is not None else "mouse"
        return False, f"{atype}{where} via {via}"

    async def _glide_to(self, tx: float, ty: float) -> None:
        """Stream interpolated virtual-cursor positions for smooth overlay motion."""
        steps = max(1, self.coord.cfg.move_steps)
        sx, sy = self.vx, self.vy
        for i in range(1, steps + 1):
            t = i / steps
            # ease-in-out for a natural feel
            e = t * t * (3 - 2 * t)
            self.vx = sx + (tx - sx) * e
            self.vy = sy + (ty - sy) * e
            self._broadcast_cursor(active=True)
            await asyncio.sleep(0.012)

    # -- telemetry ---------------------------------------------------------
    def _set_state(self, state: AgentState) -> None:
        self.state = state
        self.coord.broadcast({
            "type": "agent_status",
            "agent_id": self.id,
            "name": self.name,
            "color": self.color,
            "state": state.value,
            "step": self.step,
            "goal": self.goal,
        })

    def _broadcast_cursor(self, active: bool) -> None:
        self.coord.broadcast({
            "type": "cursor",
            "agent_id": self.id,
            "name": self.name,
            "color": self.color,
            "x": round(self.vx, 5),
            "y": round(self.vy, 5),
            "active": active,
        })

    def _broadcast_action(self, action: str, reason: str) -> None:
        self.coord.broadcast({
            "type": "action",
            "agent_id": self.id,
            "name": self.name,
            "color": self.color,
            "action": action,
            "reason": reason,
            "step": self.step,
        })

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {self.name}: {message}")
        self.coord.broadcast({
            "type": "log",
            "agent_id": self.id,
            "name": self.name,
            "color": self.color,
            "message": message,
            "ts": stamp,
        })

    def snapshot(self) -> dict[str, Any]:
        return {
            "agent_id": self.id,
            "name": self.name,
            "color": self.color,
            "state": self.state.value,
            "goal": self.goal,
            "step": self.step,
            "x": round(self.vx, 5),
            "y": round(self.vy, 5),
        }


# ---------------------------------------------------------------------------
# Coordinator + WebSocket server
# ---------------------------------------------------------------------------
class Coordinator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.screen = Screen()
        self.planner = Planner(cfg)
        # Background input backend = the agent's own cursor (Windows only).
        self.win_input: Optional[WinInput] = None
        if cfg.input_mode == "background" and sys.platform == "win32":
            try:
                self.win_input = WinInput()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"  WARNING: background input unavailable ({exc}); "
                      f"falling back to physical mouse.")
        self.agents: dict[str, Agent] = {}
        self.clients: set[Any] = set()
        self.launched_apps: set[str] = set()  # apps opened this session
        self.voice = Voice(cfg)               # ElevenLabs TTS/STT
        self.voice_mode = False               # set by the UI; gates spoken replies
        from kaggle_worker import KaggleWorker
        self.kaggle = KaggleWorker(cfg)       # optional GPU backup provider
        self.provider = "cloud"               # cloud | local | kaggle
        self.kaggle_ready = False
        self.input_lock = asyncio.Lock()  # only one physical mouse
        self.active_agent: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._agent_count = 0
        # local (WebGPU, in-app) planning bridge
        self.local_mode = os.getenv("USE_LOCAL", "false").lower() == "true"
        self.pending_plans: dict[str, asyncio.Future] = {}

    # -- fan-out broadcast (thread/async safe) -----------------------------
    def broadcast(self, payload: dict[str, Any]) -> None:
        if not self.clients:
            return
        msg = json.dumps(payload)
        loop = self._loop or asyncio.get_event_loop()
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(self._safe_send(ws, msg), loop)

    async def _safe_send(self, ws: Any, msg: str) -> None:
        try:
            await ws.send(msg)
        except Exception:
            self.clients.discard(ws)

    # -- agent management --------------------------------------------------
    def _busy(self) -> bool:
        """True if any agent is still working (not finished/stopped)."""
        return any(a.state not in (AgentState.DONE, AgentState.KILLED, AgentState.ERROR)
                   for a in self.agents.values())

    def create_agent(self, goal: str) -> Agent:
        agent = Agent(self, goal, self._agent_count)
        self._agent_count += 1
        self.agents[agent.id] = agent
        self.broadcast({"type": "agent_created", **agent.snapshot()})
        agent.start()
        return agent

    async def route_message(self, text: str, source: str = "text") -> None:
        """Classify a message as chat vs task. Chat -> reply in chat (and speak if
        voice mode). Task -> spawn an agent to do it on screen."""
        text = (text or "").strip()
        if not text:
            return
        # Heuristic first (reliable + free); only ask the model when unsure.
        mode, reply = heuristic_classify(text)
        if mode == "unknown":
            if self.provider in ("cloud", "kaggle"):
                mode, reply = await self.planner.classify(text)
            else:
                mode, reply = "task", ""
        if mode == "chat":
            reply = reply or "Hey! Tell me a task and I'll do it on your screen."
            self.broadcast({"type": "chat", "text": reply})
            if self.voice_mode:
                await self.speak(reply)
            return
        # task: voice input is gated to one at a time (segments/noise); typed isn't
        if source == "voice" and self._busy():
            self.broadcast({"type": "chat", "text": "I'm still working on the last one — one sec."})
            return
        self.create_agent(text)

    def control(self, msg: dict[str, Any]) -> None:
        ctype = msg.get("type")
        aid = msg.get("agent_id")
        if ctype == "create_agent":
            asyncio.create_task(self.route_message(msg.get("goal", ""), "text"))
        elif ctype == "pause_agent" and aid in self.agents:
            self.agents[aid].pause()
        elif ctype == "resume_agent" and aid in self.agents:
            self.agents[aid].resume()
        elif ctype == "kill_agent" and aid in self.agents:
            self.agents[aid].kill()
        elif ctype == "kill_all":
            for agent in list(self.agents.values()):
                agent.kill()
        elif ctype == "set_mode":
            # provider: "cloud" | "local" | "kaggle" | "ollama" (older clients send local:bool)
            provider = msg.get("provider") or ("local" if msg.get("local") else "cloud")
            if provider == "ollama" and msg.get("model"):
                self.cfg.ollama_model = str(msg.get("model")).strip()
            self._set_provider(provider)
        elif ctype == "plan_response":
            fut = self.pending_plans.get(msg.get("request_id"))
            if fut and not fut.done():
                fut.set_result({
                    "thought": msg.get("thought", ""),
                    "action": msg.get("action") or {"type": "wait", "reason": "empty local response"},
                })
        elif ctype == "set_voice":
            self.voice_mode = bool(msg.get("on"))
            print(f"  voice mode -> {'ON' if self.voice_mode else 'off'}")
        elif ctype == "voice_input":
            asyncio.create_task(self._voice_input(msg))

    # -- provider switching -----------------------------------------------
    def _set_provider(self, provider: str) -> None:
        if provider not in ("cloud", "local", "kaggle", "ollama"):
            provider = "cloud"
        prev = self.provider
        # Leaving Kaggle -> revert planner to cloud and stop the worker.
        if prev == "kaggle" and provider != "kaggle":
            self.planner.use_cloud()
            self.kaggle_ready = False
            asyncio.create_task(self.kaggle.stop())
        if prev == "ollama" and provider != "ollama":
            self.planner.use_cloud()
        self.provider = provider
        self.local_mode = provider == "local"
        print(f"  provider -> {provider.upper()}")
        self.broadcast({"type": "mode", "local": self.local_mode, "provider": provider})
        if provider == "kaggle":
            self.kaggle_ready = False
            asyncio.create_task(self._start_kaggle())
        elif provider == "ollama":
            asyncio.create_task(self._start_ollama())

    async def _start_ollama(self) -> None:
        """Check Ollama is running and the model is available, then use it."""
        import httpx
        base = self.cfg.ollama_base_url.rstrip("/")
        root = base[:-3] if base.endswith("/v1") else base   # http://localhost:11434
        model = self.cfg.ollama_model

        def prog(m: str, ready: bool = False) -> None:
            self.broadcast({"type": "ollama_status", "message": m, "ready": ready})

        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(root + "/api/tags")
            r.raise_for_status()
            tags = [m.get("name", "") for m in (r.json().get("models") or [])]
        except Exception:
            prog("Ollama isn't running. Install it from ollama.com, start it, "
                 "then pick Ollama again.")
            self._set_provider("cloud")
            return

        base_name = model.split(":")[0]
        if not any(t == model or t.split(":")[0] == base_name for t in tags):
            prog(f"pulling {model} (first time — this downloads the model)…")
            try:
                async with httpx.AsyncClient(timeout=None) as c:
                    # streaming pull; wait for it to finish
                    async with c.stream("POST", root + "/api/pull", json={"name": model}) as resp:
                        async for _ in resp.aiter_lines():
                            pass
            except Exception as exc:
                prog(f"couldn't pull {model}: {exc}. Try `ollama pull {model}` in a terminal.")
                self._set_provider("cloud")
                return

        if self.provider == "ollama":
            self.planner.use_ollama(base, model)
            prog(f"Ollama ready ({model})", ready=True)

    async def _start_kaggle(self) -> None:
        def prog(m: str) -> None:
            print(f"  [kaggle] {m}")
            self.broadcast({"type": "kaggle_status", "message": m})
        if not self.kaggle.available:
            prog("Kaggle not configured — add KAGGLE_USERNAME/KAGGLE_KEY (or "
                 "KAGGLE_ENDPOINT_URL) to .env, then reselect Kaggle GPU")
            self._set_provider("cloud")
            return
        base = await self.kaggle.ensure_running(prog)
        if base and self.provider == "kaggle":
            self.planner.use_kaggle(base)
            self.kaggle_ready = True
            self.broadcast({"type": "kaggle_status", "message": "Kaggle GPU ready", "ready": True})
        elif self.provider == "kaggle":
            self.broadcast({"type": "kaggle_status",
                            "message": "couldn't start the Kaggle worker — back on cloud", "ready": False})
            self._set_provider("cloud")

    # -- voice -------------------------------------------------------------
    async def speak(self, text: str, agent: Optional["Agent"] = None) -> None:
        """Synthesize `text` with ElevenLabs and push the audio to the UI. Only
        runs in voice mode; safe to fire-and-forget so the agent keeps working."""
        if not self.voice_mode or not self.voice.enabled or not text or not text.strip():
            return
        audio = await asyncio.to_thread(self.voice.tts, text)
        if not audio:
            if self.voice.last_error:
                self.broadcast({"type": "voice_error", "text": self.voice.last_error})
            return
        payload = {"type": "speak", "text": text,
                   "audio": base64.b64encode(audio).decode("ascii"), "mime": "audio/mpeg"}
        if agent is not None:
            payload.update({"agent_id": agent.id, "name": agent.name, "color": agent.color})
        self.broadcast(payload)

    async def _voice_input(self, msg: dict[str, Any]) -> None:
        """Transcribe mic audio from the UI, then spawn an agent for that goal.
        Fully guarded — a voice/API failure must never crash the engine task."""
        try:
            try:
                audio = base64.b64decode(msg.get("audio") or "")
            except Exception:
                audio = b""
            mime = msg.get("mime") or "audio/webm"
            if not self.voice.enabled:
                self.broadcast({"type": "transcript", "text": "",
                                "error": "voice not configured — add ELEVENLABS_API_KEY to .env"})
                return
            text = await asyncio.to_thread(self.voice.stt, audio, mime)
            if not text or len(text.strip()) < 2:
                return  # ignore empty / noise-only transcripts
            self.broadcast({"type": "transcript", "text": text})   # show what the user said
            # Route through the chat/task classifier (also gates duplicate voice
            # segments while an agent is already working).
            await self.route_message(text, "voice")
        except Exception as exc:
            self.broadcast({"type": "transcript", "text": "", "error": f"voice input failed: {exc}"})

    async def request_local_plan(self, goal: str, history: list[str], image_b64: Optional[str]) -> dict[str, Any]:
        """Ask the in-app WebGPU model (renderer) for the next action over WS."""
        req_id = uuid.uuid4().hex[:10]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self.pending_plans[req_id] = fut
        self.broadcast({
            "type": "plan_request",
            "request_id": req_id,
            "system": PLANNER_SYSTEM_PROMPT,
            "user": build_context(goal, history),
            "image": image_b64 or "",
        })
        try:
            return await asyncio.wait_for(fut, timeout=180)
        except asyncio.TimeoutError:
            raise RuntimeError("local model didn't respond — open the app, switch on Local mode and Load the model.")
        finally:
            self.pending_plans.pop(req_id, None)

    # -- WebSocket handler -------------------------------------------------
    async def handler(self, ws: Any) -> None:
        self.clients.add(ws)
        # Send current world state to the freshly-connected overlay.
        def _set(v: str) -> bool:
            v = (v or "").strip()
            return bool(v) and not v.startswith("PASTE_")
        await ws.send(json.dumps({
            "type": "hello",
            "screen": {"width": self.screen.width, "height": self.screen.height},
            "agents": [a.snapshot() for a in self.agents.values()],
            "local": self.local_mode,
            "provider": self.provider,
            "config": {
                "zai": _set(self.cfg.api_key),
                "elevenlabs": self.voice.enabled,
                "kaggle": self.kaggle.available,
            },
        }))
        try:
            async for raw in ws:
                try:
                    self.control(json.loads(raw))
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        finally:
            self.clients.discard(ws)

    async def serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        print(f"AgentLab engine :: ws://{self.cfg.ws_host}:{self.cfg.ws_port}")
        mode = ("background (agent's own cursor)" if self.win_input
                else "physical (shares your mouse)")
        print(f"  model={self.cfg.model}  vision={self.cfg.vision_enabled}  "
              f"input={mode}  screen={self.screen.width}x{self.screen.height}")
        print(f"  voice={'ready (ElevenLabs)' if self.voice.enabled else 'off — set ELEVENLABS_API_KEY in .env'}")
        print(f"  ui_elements={'on (UI Automation — accurate clicks)' if self.cfg.use_ui_elements else 'off'}")
        print(f"  kaggle={'available (⚡ GPU backup selectable)' if self.kaggle.available else 'off — set KAGGLE_* in .env'}")
        if not self.cfg.api_key or self.cfg.api_key.startswith("PASTE_"):
            print("  WARNING: ZAI_API_KEY is not set in .env — planner calls will fail.")
        try:
            async with websockets.serve(self.handler, self.cfg.ws_host, self.cfg.ws_port,
                                        ping_interval=20, ping_timeout=20):
                await asyncio.Future()  # run forever
        except OSError as exc:
            if getattr(exc, "errno", None) in (48, 98, 10048):
                print(f"\n  ERROR: port {self.cfg.ws_port} is already in use.")
                print("  Another AgentLab engine is probably still running.")
                print("  Close it (or change WS_PORT in .env), then start again.")
                sys.exit(1)
            raise


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
async def main() -> None:
    coord = Coordinator(CFG)

    stop = asyncio.Event()

    def _shutdown(*_: Any) -> None:
        for agent in list(coord.agents.values()):
            agent.kill()
        # Best-effort: tell the Kaggle worker to stop so the GPU kernel ends.
        try:
            asyncio.get_event_loop().create_task(coord.kaggle.stop())
        except Exception:
            pass
        stop.set()

    try:
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
    except (ValueError, AttributeError):
        pass  # signals not settable on some Windows contexts

    server = asyncio.create_task(coord.serve())
    try:
        await server  # serve() blocks until cancelled
    finally:
        # Ensure the Kaggle GPU worker is shut down when the app exits.
        try:
            await asyncio.wait_for(coord.kaggle.stop(), timeout=10)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgentLab engine stopped.")
