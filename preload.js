// Secure bridge between the renderer and Electron main.
// Exposes only the two window-control channels the control panel needs,
// plus the WebSocket endpoint pulled from the environment (with defaults).
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("agentlab", {
  wsUrl: `ws://${process.env.WS_HOST || "127.0.0.1"}:${process.env.WS_PORT || "8765"}`,
  minimize: () => ipcRenderer.send("control:minimize"),
  maximize: () => ipcRenderer.send("control:maximize"),
  close: () => ipcRenderer.send("control:close"),
});
