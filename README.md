# AgentLab 🤖🖥️

**Talk to your computer — and watch AI agents actually use it.**

AgentLab spawns autonomous agents that control your Windows desktop like a human:
they see the screen, click real buttons, type, launch apps, play music, browse the
web — and narrate what they're doing. You can drive it by **typing** or by **voice**
(talk to it, and it talks back).

It's an Electron app (chat panel + on‑screen cursor overlay) backed by a Python
engine that does the screen capture, planning (via a vision LLM), and input.

---

## ✨ Features

- **Natural‑language control** — describe a task ("open Chrome and search for lofi")
  and an agent carries it out step by step.
- **Accurate clicking, not pixel‑guessing** — reads real UI elements via Windows
  UI Automation *and* overlays a coordinate grid on the AI's screenshot, so clicks
  land on the actual button. DPI‑aware (works at any display scaling).
- **Voice chat** 🎙️ — continuous listening (talk naturally, it detects when you're
  done), and it speaks back only what matters via **ElevenLabs**; everything else
  stays in the chat.
- **Real human toolkit** — click / double‑click / right‑click, type any text
  (unicode via clipboard), any keyboard shortcut, drag, scroll.
- **Smart shortcuts** — `open_app` (launch any installed app), `open_url` (open a
  website), `play_track` (search & play a song on Spotify), `media` (play/pause/
  next/prev), plus element‑based clicks.
- **Multi‑agent** — run several agents at once; each has its own coloured cursor
  orb on the overlay, and you can pause/resume/stop any of them.
- **Three model providers**, switchable in‑app:
  1. **☁ Cloud API** — any OpenAI‑compatible vision model (default: **Z.AI GLM‑4.6V‑Flash**).
  2. **⚡ Kaggle GPU** — run your own open model (e.g. **GLM‑4.1V‑9B**) on a free
     Kaggle GPU; the app starts the notebook, tunnels to it, and stops it on exit.
  3. **▣ Local WebGPU** — run a vision model in‑browser on your GPU (**SmolVLM**,
     **Gemma 3**, or a custom model) — free, offline, no key.

---

## 🚀 Quick start

### 1. Prerequisites
- **Windows 10/11**, **Python 3.10+**, **Node.js 18+**

### 2. Install
```bash
git clone <your-repo-url> AgentLab
cd AgentLab
pip install -r requirements.txt
npm install
```

### 3. Configure
```bash
copy .env.example .env      # then edit .env
```
Set at least one provider key (see **Configuration** below). For the default cloud
provider, add a **Z.AI** key.

### 4. Run
```bash
npm start
```
This launches the Electron app **and** the Python engine together. Type a task, or
click 🎙 to talk.

---

## ⚙️ Configuration (`.env`)

| Key | What it's for |
|---|---|
| `ZAI_API_KEY` | Z.AI (GLM) cloud vision model — the default provider. |
| `AGENT_MODEL` | Cloud model id (default `glm-4.6v-flash`). |
| `INPUT_MODE` | `physical` (real mouse/keyboard, works everywhere) or `background`. |
| `USE_GRID` | Overlay a coordinate ruler on the AI's screenshot (default `true`). |
| `USE_UI_ELEMENTS` | Read real clickable elements for exact clicks (default `false`). |
| `ELEVENLABS_API_KEY` | Voice chat (needs **Text‑to‑Speech** + **Speech‑to‑Text** scopes). |
| `ELEVENLABS_VOICE_ID` | A free‑tier premade voice (default: George). |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Run the model on your Kaggle GPU. |
| `KAGGLE_MODEL` | Open model the Kaggle worker loads (default GLM‑4.1V‑9B). |
| `KAGGLE_ENDPOINT_URL` | Skip auto‑start and point at an already‑running worker. |
| `MONITOR` | Force a specific monitor (multi‑monitor setups). |

> **Never commit `.env`** — it's gitignored. Keys stay on your machine.

---

## 🧠 Providers in depth

### ☁ Cloud API (default)
Uses any OpenAI‑compatible endpoint. Point `AGENT_MODEL` + the base URL at Z.AI,
Gemini (OpenAI‑compat), OpenAI, etc.

### ⚡ Kaggle GPU
Select **⚡ Kaggle GPU** in Settings. The app pushes [`kaggle_kernel/server.py`](kaggle_kernel/server.py)
to your Kaggle account, which loads your open vision model, serves an
OpenAI‑compatible endpoint, opens a **cloudflared** tunnel, and publishes its URL.
The app routes requests there and shuts the worker down when you close it.
*Fastest path:* run your notebook yourself and paste its URL into
`KAGGLE_ENDPOINT_URL`.

### ▣ Local WebGPU
Runs a vision model in the Electron renderer on your GPU via transformers.js —
free and offline. Pick **SmolVLM** (light), **Gemma 3 4B** (strong, needs a big
GPU), or type a custom model id.

---

## 🎛️ Controls & safety

- **Emergency stop:** slam the mouse into a screen corner (pyautogui failsafe),
  press **Ctrl+Shift+H** to hide the overlay, or click **Stop all agents**.
- The agent shares your physical cursor in `physical` mode — watch it or step away.

---

## 🗂️ Project layout

```
agent_engine.py     Python engine: capture, planning, input, voice, providers
kaggle_worker.py    Starts/monitors/stops the Kaggle GPU worker
kaggle_kernel/      server.py that runs on Kaggle (model + tunnel)
main.js             Electron main process
control.html/.js    Chat panel UI
overlay.html/render.js  On‑screen cursor overlay
local-vlm.js        Local WebGPU model runner
theme.css / sfx.js  Styling and sound
```

---

## ⚠️ Notes & limitations
- Windows‑only (uses Win32 UI Automation + DPI APIs).
- Small local/vision models are weaker at precise clicking than cloud ones.
- Kaggle GPUs have weekly quota; the worker runs until you close the app.

---

## 📄 License
MIT — see [LICENSE](LICENSE).
