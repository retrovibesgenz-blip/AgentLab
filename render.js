// ===========================================================================
// AgentLab — Overlay renderer
// Draws one animated black-orb mascot cursor per agent from streamed telemetry.
// Positions arrive normalized (0..1) and map to the overlay window in pixels,
// so rendering is resolution / DPI independent.
// ===========================================================================

const stage = document.getElementById("stage");
const cursors = new Map(); // agent_id -> { el, labelName, labelAct, n }
const WS_URL = (window.agentlab && window.agentlab.wsUrl) || "ws://127.0.0.1:8765";
let seq = 0;

function ensureCursor(agent) {
  let entry = cursors.get(agent.agent_id);
  if (entry) return entry;

  const n = ++seq;
  const color = agent.color || "#00e5ff";
  const el = document.createElement("div");
  el.className = "cursor idle";
  el.style.setProperty("--tint", color);
  el.innerHTML = `
    <span class="pulse"></span>
    <span class="aim"></span>
    <span class="orb tint" style="--tint:${color}">
      <span class="eye l"></span><span class="eye r"></span>
      <span class="badge">${n}</span>
    </span>
    <span class="label"><span class="who">${escapeHtml(agent.name || agent.agent_id)}</span>
      <span class="act"></span></span>`;
  stage.appendChild(el);

  entry = { el, labelAct: el.querySelector(".act"), n, x: agent.x ?? 0.5, y: agent.y ?? 0.5 };
  cursors.set(agent.agent_id, entry);
  move(entry, entry.x, entry.y);
  return entry;
}

function move(entry, nx, ny) {
  entry.el.style.transform = `translate(${nx * window.innerWidth}px, ${ny * window.innerHeight}px)`;
}

function removeCursor(id) {
  const e = cursors.get(id);
  if (!e) return;
  e.el.style.transition = "opacity .4s, transform .4s";
  e.el.style.opacity = "0";
  e.el.style.transform += " scale(1.5)";
  setTimeout(() => { e.el.remove(); cursors.delete(id); }, 420);
}

function connect() {
  const ws = new WebSocket(WS_URL);
  ws.onopen = () => console.log("[overlay] connected", WS_URL);
  ws.onmessage = (ev) => { let m; try { m = JSON.parse(ev.data); } catch { return; } handle(m); };
  ws.onclose = () => setTimeout(connect, 1500);
  ws.onerror = () => ws.close();
}

function handle(msg) {
  switch (msg.type) {
    case "hello":
      (msg.agents || []).forEach(ensureCursor);
      break;
    case "agent_created":
      ensureCursor(msg);
      break;
    case "cursor": {
      const e = ensureCursor(msg);
      e.x = msg.x; e.y = msg.y;
      move(e, msg.x, msg.y);
      e.el.classList.toggle("active", !!msg.active);
      e.el.classList.toggle("idle", !msg.active);
      break;
    }
    case "action": {
      const e = ensureCursor(msg);
      e.labelAct.textContent = `${msg.action}: ${msg.reason || ""}`.slice(0, 46);
      break;
    }
    case "agent_status": {
      const e = ensureCursor(msg);
      if (["done", "killed", "error"].includes(msg.state)) {
        e.labelAct.textContent = msg.state.toUpperCase();
        e.el.classList.remove("active"); e.el.classList.add("idle");
        if (msg.state === "killed") removeCursor(msg.agent_id);
      }
      break;
    }
  }
}

function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

connect();
