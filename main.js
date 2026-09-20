// ===========================================================================
// AgentLab — Electron main process
//
// Creates two windows and AUTO-STARTS the Python engine as a child process,
// so a single `npm start` launches the whole system (engine + overlay + chat).
//   1. overlay  — fullscreen, transparent, click-through, always-on-top cursors.
//   2. control  — the interactive chat panel.
// ===========================================================================

const { app, BrowserWindow, screen, ipcMain, globalShortcut, session } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

// Let the agent's spoken replies play without a per-clip user gesture.
app.commandLine.appendSwitch("autoplay-policy", "no-user-gesture-required");

let overlayWin = null;
let controlWin = null;
let engineProc = null;

// --------------------------------------------------------------- engine ----
function startEngine() {
  // Try common Python launchers in order until one spawns.
  const candidates = process.platform === "win32"
    ? ["py", "python", "python3"]
    : ["python3", "python"];
  let i = 0;

  const tryNext = () => {
    if (i >= candidates.length) {
      console.error("[APP] Could not find Python. Install Python 3 and `pip install -r requirements.txt`.");
      return;
    }
    const cmd = candidates[i++];
    const proc = spawn(cmd, ["agent_engine.py"], { cwd: __dirname, env: process.env });
    let ok = false;

    proc.on("spawn", () => {
      ok = true;
      engineProc = proc;
      console.log(`[APP] engine started via "${cmd}"`);
    });
    proc.on("error", () => { if (!ok) tryNext(); });          // launcher not found → try next
    proc.stdout.on("data", (d) => process.stdout.write(`[ENGINE] ${d}`));
    proc.stderr.on("data", (d) => process.stderr.write(`[ENGINE] ${d}`));
    proc.on("exit", (code) => {
      if (ok) console.log(`[APP] engine exited (code ${code}).`);
    });
  };
  tryNext();
}

function stopEngine() {
  if (engineProc && !engineProc.killed) {
    try { engineProc.kill(); } catch {}
    engineProc = null;
  }
}

// --------------------------------------------------------------- windows ---
function createOverlay() {
  const primary = screen.getPrimaryDisplay();
  const { x, y, width, height } = primary.bounds;

  overlayWin = new BrowserWindow({
    x, y, width, height,
    transparent: true, frame: false, alwaysOnTop: true, skipTaskbar: true,
    resizable: false, movable: false, focusable: false, hasShadow: false,
    fullscreenable: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true, nodeIntegration: false,
    },
  });

  overlayWin.setIgnoreMouseEvents(true, { forward: true });   // clicks pass through
  overlayWin.setAlwaysOnTop(true, "screen-saver");
  overlayWin.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  overlayWin.loadFile("overlay.html");
}

function createControl() {
  const primary = screen.getPrimaryDisplay();
  const { width } = primary.workAreaSize;

  controlWin = new BrowserWindow({
    x: width - 400, y: 40, width: 380, height: 620,
    frame: false, transparent: true, resizable: true, alwaysOnTop: true,
    skipTaskbar: false, minWidth: 320, minHeight: 320,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true, nodeIntegration: false,
    },
  });

  controlWin.loadFile("control.html");
  controlWin.setAlwaysOnTop(true, "floating");
}

// control-panel window buttons
ipcMain.on("control:minimize", () => controlWin && controlWin.minimize());
ipcMain.on("control:close", () => app.quit());
ipcMain.on("control:maximize", () => {
  if (!controlWin) return;
  controlWin.isMaximized() ? controlWin.unmaximize() : controlWin.maximize();
});

// --------------------------------------------------------------- lifecycle -
app.whenReady().then(() => {
  // Allow the control panel to use the microphone (voice chat).
  session.defaultSession.setPermissionRequestHandler((_wc, permission, cb) => {
    cb(permission === "media" || permission === "microphone" || permission === "audioCapture");
  });
  try {
    session.defaultSession.setPermissionCheckHandler(() => true);
  } catch {}

  startEngine();          // spin up the Python engine first
  createOverlay();
  createControl();

  // Panic hotkey: hide/show the overlay instantly.
  globalShortcut.register("CommandOrControl+Shift+H", () => {
    if (!overlayWin) return;
    overlayWin.isVisible() ? overlayWin.hide() : overlayWin.show();
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) { createOverlay(); createControl(); }
  });
});

app.on("will-quit", () => { globalShortcut.unregisterAll(); stopEngine(); });
app.on("before-quit", stopEngine);
app.on("window-all-closed", () => { stopEngine(); if (process.platform !== "darwin") app.quit(); });
