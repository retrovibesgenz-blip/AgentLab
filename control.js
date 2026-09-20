// ===========================================================================
// AgentLab — Chat control panel
// A conversational front end: type a task -> an agent is spawned to do it,
// and every thought / action / result streams back into the chat. Active
// agents show as chips you can pause or kill. Overlay draws the orb cursors.
// ===========================================================================

const WS_URL = (window.agentlab && window.agentlab.wsUrl) || "ws://127.0.0.1:8765";

const el = {
  hdr: document.getElementById("hdr"),
  status: document.getElementById("status"),
  agents: document.getElementById("agents"),
  chat: document.getElementById("chat"),
  welcome: document.getElementById("welcome"),
  goal: document.getElementById("goal"),
  send: document.getElementById("send"),
  killall: document.getElementById("killall"),
  theme: document.getElementById("theme"),
  sound: document.getElementById("sound"),
  maxbtn: document.getElementById("maxbtn"),
  min: document.getElementById("min"),
  close: document.getElementById("close"),
  banner: document.getElementById("banner"),
  settings: document.getElementById("settings"),
  panel: document.getElementById("panel"),
  modeCloud: document.getElementById("mode-cloud"),
  modeOllama: document.getElementById("mode-ollama"),
  modeKaggle: document.getElementById("mode-kaggle"),
  modeLocal: document.getElementById("mode-local"),
  cloudInfo: document.getElementById("cloud-info"),
  ollamaInfo: document.getElementById("ollama-info"),
  ollamaModel: document.getElementById("ollama-model"),
  ollamaStatus: document.getElementById("ollama-status"),
  kaggleInfo: document.getElementById("kaggle-info"),
  kaggleStatus: document.getElementById("kaggle-status"),
  localInfo: document.getElementById("local-info"),
  localModel: document.getElementById("local-model"),
  localPreset: document.getElementById("local-preset"),
  loadModel: document.getElementById("load-model"),
  localStatus: document.getElementById("local-status"),
  voice: document.getElementById("voice"),
  voicebar: document.getElementById("voicebar"),
  blob: document.getElementById("blob"),
  talk: document.getElementById("talk"),
  vstatus: document.getElementById("vstatus"),
};

const agents = new Map(); // agent_id -> { n, name, state, goal }
let ws = null;
let seq = 0;
let pendingGoal = null;    // goal text awaiting its agent_created, to attribute it
let provider = "cloud";    // "cloud" | "local" | "kaggle"
try { provider = localStorage.getItem("agentlab.provider") || "cloud"; } catch {}
let localMode = provider === "local";  // planning runs in-browser on WebGPU when true

// --- window chrome + toggles ---------------------------------------------
el.min.onclick = () => { SFX.play("click"); window.agentlab.minimize(); };
el.close.onclick = () => { SFX.play("click"); window.agentlab.close(); };
if (el.maxbtn) el.maxbtn.onclick = () => { SFX.play("click"); window.agentlab.maximize && window.agentlab.maximize(); };

(function initTheme() {
  let t = "light"; try { t = localStorage.getItem("agentlab.theme") || "light"; } catch {}
  document.documentElement.dataset.theme = t;
})();
el.theme.onclick = () => {
  const root = document.documentElement;
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  try { localStorage.setItem("agentlab.theme", root.dataset.theme); } catch {}
  SFX.play("click");
};

function refreshSound() { el.sound.textContent = SFX.isMuted() ? "♪̸" : "♪"; el.sound.style.opacity = SFX.isMuted() ? ".4" : "1"; }
el.sound.onclick = () => { SFX.toggle(); refreshSound(); };
refreshSound();

// --- settings / provider mode --------------------------------------------
el.settings.onclick = () => { el.panel.hidden = !el.panel.hidden; SFX.play("click"); };

function applyModeUI() {
  el.modeCloud.classList.toggle("active", provider === "cloud");
  el.modeOllama.classList.toggle("active", provider === "ollama");
  el.modeKaggle.classList.toggle("active", provider === "kaggle");
  el.modeLocal.classList.toggle("active", provider === "local");
  el.cloudInfo.hidden = provider !== "cloud";
  el.ollamaInfo.hidden = provider !== "ollama";
  el.kaggleInfo.hidden = provider !== "kaggle";
  el.localInfo.hidden = provider !== "local";
}
function setProvider(p) {
  provider = p;
  localMode = p === "local";
  try { localStorage.setItem("agentlab.provider", p); } catch {}
  applyModeUI();
  const msg = { type: "set_mode", provider: p, local: localMode };
  if (p === "ollama" && el.ollamaModel) msg.model = el.ollamaModel.value.trim();
  send(msg);   // tell the engine where to plan
  SFX.play("click");
}
el.modeCloud.onclick = () => setProvider("cloud");
el.modeOllama.onclick = () => setProvider("ollama");
el.modeKaggle.onclick = () => setProvider("kaggle");
el.modeLocal.onclick = () => setProvider("local");
applyModeUI();

if (el.localPreset) el.localPreset.onchange = () => {
  const v = el.localPreset.value;
  if (v === "__custom__") { el.localModel.focus(); el.localModel.select(); }
  else { el.localModel.value = v; }
};

el.loadModel.onclick = async () => {
  const id = el.localModel.value.trim() || "onnx-community/SmolVLM-256M-Instruct";
  el.loadModel.disabled = true;
  el.localStatus.classList.remove("ok");
  try {
    await LocalVLM.load(id, (s) => {
      if (s.status === "progress" && s.file) el.localStatus.textContent = `${s.file} — ${s.pct}%`;
      else if (s.message) el.localStatus.textContent = s.message;
      else if (s.status) el.localStatus.textContent = s.status + (s.file ? ` ${s.file}` : "");
    });
    el.localStatus.textContent = `ready · ${id}`;
    el.localStatus.classList.add("ok");
    SFX.play("chime");
  } catch (e) {
    el.localStatus.textContent = "load failed: " + (e && e.message ? e.message : e);
  } finally {
    el.loadModel.disabled = false;
  }
};

// --- websocket ------------------------------------------------------------
function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify(obj)); return true; }
  return false;
}

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    el.hdr.classList.add("online");
    el.status.textContent = localMode ? "engine online · local" : "engine online";
    send({ type: "set_mode", provider, local: localMode });  // sync engine to our chosen provider
    send({ type: "set_voice", on: voiceMode });      // re-sync voice mode after (re)connect
  };
  ws.onclose = () => {
    el.hdr.classList.remove("online");
    el.status.textContent = "starting engine…";
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => { let m; try { m = JSON.parse(ev.data); } catch { return; } route(m); };
}

function route(msg) {
  switch (msg.type) {
    case "hello":
      (msg.agents || []).forEach((a) => { if (!agents.has(a.agent_id)) { a._n = ++seq; register(a); } });
      renderAgents();
      if (msg.config) showKeyBanner(msg.config);
      break;

    case "agent_created": {
      msg._n = ++seq;
      register(msg);
      renderAgents();
      SFX.play("boop");
      const goal = pendingGoal || msg.goal || "";
      pendingGoal = null;
      addAgent(agents.get(msg.agent_id), `is on it — <i>${escapeHtml(goal)}</i>`, "status");
      break;
    }

    case "agent_status": {
      const a = agents.get(msg.agent_id); if (!a) break;
      a.state = msg.state; a.step = msg.step;
      renderAgents();
      if (msg.state === "done") SFX.play("chime");
      if (msg.state === "killed") { SFX.play("thunk"); addAgent(a, `<span class="ic">■</span> stopped`, "stop"); dropAgent(msg.agent_id); }
      if (msg.state === "error") addAgent(a, `⚠ hit an error and stopped`, "stop");
      break;
    }

    case "action": {
      const a = agents.get(msg.agent_id); if (!a) break;
      SFX.play("tick");
      addAgent(a, `<span class="ic">⚡</span><b>${escapeHtml(msg.action)}</b> — ${escapeHtml(msg.reason || "")}`, "action");
      break;
    }

    case "log": {
      const a = agents.get(msg.agent_id); if (!a) break;
      const m = msg.message || "";
      if (m.startsWith("think:")) addAgent(a, escapeHtml(m.slice(6).trim()), "think");
      else if (m.startsWith("say:")) addAgent(a, `<span class="ic">🔊</span> ${escapeHtml(m.slice(4).trim())}`, "say");
      else if (m.startsWith("goal complete")) addAgent(a, `<span class="ic">✓</span> ${escapeHtml(m.replace("goal complete:", "").trim())}`, "done");
      else if (m.startsWith("started")) { /* covered by the 'is on it' bubble */ }
      else addAgent(a, escapeHtml(m), "think");
      break;
    }

    case "speak":
      playSpeech(msg);   // AI's spoken line — audio + blob animation
      break;

    case "transcript":
      if (msg.text) addUser(msg.text);
      else if (msg.error) { el.vstatus.textContent = msg.error; addSystem("🎙 " + msg.error); }
      break;

    case "voice_error":
      el.vstatus.textContent = msg.text || "voice error";
      addSystem("🔊 " + (msg.text || "voice error"));
      break;

    case "mode":
      provider = msg.provider || (msg.local ? "local" : "cloud");
      localMode = provider === "local";
      applyModeUI();
      el.status.textContent = provider === "cloud" ? "engine online" : `engine online · ${provider}`;
      break;

    case "chat":
      pendingGoal = null;              // this was conversation, not a task
      addAssistant(msg.text || "");
      break;

    case "kaggle_status":
      if (el.kaggleStatus) { el.kaggleStatus.textContent = msg.message || ""; el.kaggleStatus.classList.toggle("ok", !!msg.ready); }
      el.status.textContent = msg.ready ? "engine online · kaggle" : (msg.message || "engine online");
      addSystem("⚡ " + (msg.message || "Kaggle"));
      break;

    case "ollama_status":
      if (el.ollamaStatus) { el.ollamaStatus.textContent = msg.message || ""; el.ollamaStatus.classList.toggle("ok", !!msg.ready); }
      el.status.textContent = msg.ready ? "engine online · ollama" : (msg.message || "engine online");
      addSystem("🦙 " + (msg.message || "Ollama"));
      break;

    case "plan_request":
      handlePlanRequest(msg);   // run the local WebGPU model
      break;
  }
}

// --- local WebGPU planning ------------------------------------------------
async function handlePlanRequest(msg) {
  const reply = (action, thought) => send({ type: "plan_response", request_id: msg.request_id, action, thought });

  if (!LocalVLM.isLoaded()) {
    reply({ type: "wait", reason: "local model not loaded" },
          "Local mode is on but no model is loaded — open ⚙ Settings and click Load model.");
    return;
  }
  try {
    const imgUrl = msg.image ? `data:image/png;base64,${msg.image}` : null;
    if (!imgUrl) { reply({ type: "wait", reason: "no screenshot" }, ""); return; }
    const raw = await LocalVLM.generate(imgUrl, msg.system || "", msg.user || "");
    const parsed = parseAction(raw);
    reply(parsed.action, parsed.thought);
  } catch (e) {
    reply({ type: "wait", reason: "local inference error" }, String(e && e.message ? e.message : e));
  }
}

// Extract the first JSON object from model text -> {thought, action}.
function parseAction(text) {
  try {
    const m = String(text).match(/\{[\s\S]*\}/);
    if (!m) return { thought: "", action: { type: "wait", reason: "no JSON from model" } };
    const d = JSON.parse(m[0]);
    return { thought: d.thought || "", action: d.action || { type: "wait", reason: "no action field" } };
  } catch {
    return { thought: "", action: { type: "wait", reason: "unparseable model output" } };
  }
}

// --- agent bookkeeping ----------------------------------------------------
function register(a) { agents.set(a.agent_id, { id: a.agent_id, n: a._n, name: a.name, color: a.color || "#00e5ff", state: a.state || "planning", goal: a.goal }); }
function dropAgent(id) { agents.delete(id); setTimeout(renderAgents, 400); }

function renderAgents() {
  el.agents.innerHTML = "";
  for (const a of agents.values()) {
    const chip = document.createElement("div");
    chip.className = `achip ${a.state}`;
    const paused = a.state === "paused";
    chip.innerHTML = `
      <span class="orb tint" style="--tint:${a.color}">${orbMarkup(a.n)}</span>
      <span><div class="nm">${escapeHtml(a.name)}</div><div class="st">${a.state}</div></span>
      <button class="mini pr" title="${paused ? "Resume" : "Pause"}">${paused ? "▶" : "⏸"}</button>
      <button class="mini kl" title="Kill">✕</button>`;
    chip.querySelector(".pr").onclick = () => { SFX.play("click"); send({ type: paused ? "resume_agent" : "pause_agent", agent_id: a.id }); };
    chip.querySelector(".kl").onclick = () => { SFX.play("thunk"); send({ type: "kill_agent", agent_id: a.id }); };
    el.agents.appendChild(chip);
  }
}

// --- chat rendering -------------------------------------------------------
function orbMarkup(n) { return `<span class="eye l"></span><span class="eye r"></span><span class="badge">${n}</span>`; }

function hideWelcome() { if (el.welcome) { el.welcome.remove(); el.welcome = null; } }

function addUser(text) {
  hideWelcome();
  const m = document.createElement("div");
  m.className = "msg user";
  m.innerHTML = `<div class="body"><div class="bubble">${escapeHtml(text)}</div></div>`;
  el.chat.appendChild(m);
  scroll();
}

function addAgent(a, html, cls) {
  hideWelcome();
  const m = document.createElement("div");
  m.className = "msg agent";
  m.innerHTML = `
    <div class="avatar orb tint" style="--tint:${a.color}">${orbMarkup(a.n)}</div>
    <div class="body">
      <div class="who">${escapeHtml(a.name)}</div>
      <div class="bubble ${cls || ""}">${html}</div>
    </div>`;
  el.chat.appendChild(m);
  scroll();
}

function addSystem(text) {
  const m = document.createElement("div");
  m.className = "msg system";
  m.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  el.chat.appendChild(m);
  scroll();
}

// A plain conversational reply from AgentLab (not an agent doing a task).
function addAssistant(text) {
  hideWelcome();
  const m = document.createElement("div");
  m.className = "msg agent";
  m.innerHTML = `
    <div class="avatar orb"><span class="eye l"></span><span class="eye r"></span></div>
    <div class="body"><div class="who">AgentLab</div>
      <div class="bubble">${escapeHtml(text)}</div></div>`;
  el.chat.appendChild(m);
  scroll();
}

function showKeyBanner(cfg) {
  if (!el.banner) return;
  const missing = [];
  if (!cfg.zai) missing.push("ZAI_API_KEY (cloud model)");
  if (!cfg.elevenlabs) missing.push("ELEVENLABS_API_KEY (voice)");
  if (!missing.length) { el.banner.hidden = true; return; }
  el.banner.innerHTML =
    `⚠️ <span><b>Set your keys in .env</b> to unlock everything — missing: ${missing.map(escapeHtml).join(", ")}. ` +
    `You can still use Local WebGPU with no key.</span><span class="x" id="banner-x">✕</span>`;
  el.banner.hidden = false;
  const x = document.getElementById("banner-x");
  if (x) x.onclick = () => { el.banner.hidden = true; };
}

function scroll() { el.chat.scrollTop = el.chat.scrollHeight; }

function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// --- composer -------------------------------------------------------------
function autoGrow() { el.goal.style.height = "40px"; el.goal.style.height = Math.min(120, el.goal.scrollHeight) + "px"; }
el.goal.addEventListener("input", autoGrow);

function submit() {
  const goal = el.goal.value.trim();
  if (!goal) { el.goal.focus(); return; }
  if (!send({ type: "create_agent", goal })) {
    addSystem("Engine is starting — try again in a moment.");
    return;
  }
  pendingGoal = goal;
  addUser(goal);
  el.goal.value = ""; autoGrow(); el.goal.focus();
}

el.send.onclick = submit;
el.goal.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
});
el.killall.onclick = () => { SFX.play("thunk"); send({ type: "kill_all" }); };

// ===========================================================================
// Voice chat: talk to the agent (mic -> engine STT) and hear it back
// (engine TTS -> here). A reactive gradient blob animates while it speaks.
// ===========================================================================
let voiceMode = false;
try { voiceMode = localStorage.getItem("agentlab.voice") === "1"; } catch {}

let audioCtx = null;          // shared Web Audio context
let analyser = null;          // current amplitude source (speech or mic)
let ampData = null;           // byte buffer for the analyser
let blobMode = "idle";        // idle | listening | speaking
let mediaRec = null;          // MediaRecorder for push-to-talk
let recChunks = [];
let micStream = null;

function ensureAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

function applyVoiceUI() {
  el.voice.classList.toggle("on", voiceMode);
  el.voicebar.hidden = !voiceMode;
}

// continuous-listening state
let listening = false, aiSpeaking = false;
let micAnalyser = null, micData = null;
let seg = null, segChunks = [], segStartT = 0, lastVoiceT = 0, inSpeech = false, vadTimer = null;

function setVoiceMode(on) {
  voiceMode = on;
  try { localStorage.setItem("agentlab.voice", on ? "1" : "0"); } catch {}
  applyVoiceUI();
  send({ type: "set_voice", on });
  SFX.play("click");
  if (on) { ensureAudioCtx(); startBlobLoop(); startListening(); }
  else { stopListening(); blobMode = "idle"; }
}
el.voice.onclick = () => setVoiceMode(!voiceMode);
// the big button = mute / unmute toggle (keeps listening continuously otherwise)
el.talk.onclick = () => { if (!voiceMode) return; listening ? stopListening() : startListening(); };
applyVoiceUI();
if (voiceMode) setTimeout(() => setVoiceMode(true), 300);

function setTalkUI() {
  if (listening) { el.talk.classList.add("rec"); el.talk.textContent = "🔴 Listening — tap to mute"; }
  else { el.talk.classList.remove("rec"); el.talk.textContent = "🔇 Muted — tap to talk"; }
}

async function startListening() {
  if (listening) return;
  try {
    if (!micStream) micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  } catch (e) {
    el.vstatus.textContent = "mic blocked — allow microphone access in Windows privacy settings";
    return;
  }
  try {
    const ctx = ensureAudioCtx();
    const src = ctx.createMediaStreamSource(micStream);
    micAnalyser = ctx.createAnalyser(); micAnalyser.fftSize = 512;
    micData = new Uint8Array(micAnalyser.frequencyBinCount);
    src.connect(micAnalyser);
  } catch {}
  listening = true; inSpeech = false; blobMode = "listening";
  analyser = micAnalyser; ampData = micData;   // blob reacts to your voice
  setTalkUI();
  el.vstatus.textContent = "listening… just speak";
  vadTimer = setInterval(vadTick, 60);
}

function stopListening() {
  listening = false;
  if (vadTimer) { clearInterval(vadTimer); vadTimer = null; }
  if (seg && seg.state === "recording") { seg._discard = true; seg.stop(); }
  inSpeech = false;
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  micAnalyser = null; analyser = null;
  if (blobMode === "listening") blobMode = "idle";
  setTalkUI();
  el.vstatus.textContent = voiceMode ? "muted — tap the button to talk" : "";
}

function micAmp() {
  if (!micAnalyser) return 0;
  micAnalyser.getByteFrequencyData(micData);
  let s = 0; for (let i = 0; i < micData.length; i++) s += micData[i];
  return (s / micData.length) / 128;
}

// Voice-activity detection: auto-capture each utterance and send it on a pause.
// Higher start threshold + longer silence window = one command per sentence,
// not one per pause, and it ignores background noise.
const START_AMP = 0.085, KEEP_AMP = 0.05, SILENCE_MS = 1100, MIN_MS = 450, MAX_MS = 14000;
function vadTick() {
  if (!listening || aiSpeaking) return;   // never capture while the AI is talking
  const amp = micAmp(), now = performance.now();
  if (!inSpeech) {
    if (amp > START_AMP) { inSpeech = true; segStartT = now; lastVoiceT = now; startSeg(); }
  } else {
    if (amp > KEEP_AMP) lastVoiceT = now;
    if (now - lastVoiceT > SILENCE_MS || now - segStartT > MAX_MS) { endSeg(now - segStartT); inSpeech = false; }
  }
}

function startSeg() {
  const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus"
             : (MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "");
  segChunks = [];
  seg = new MediaRecorder(micStream, mime ? { mimeType: mime } : undefined);
  seg.ondataavailable = (e) => { if (e.data && e.data.size) segChunks.push(e.data); };
  seg.onstop = finishSeg;
  seg.start();
}
function endSeg(dur) { if (seg && seg.state === "recording") { seg._dur = dur; seg.stop(); } }

async function finishSeg() {
  const discard = seg && seg._discard, dur = seg ? (seg._dur || 0) : 0;
  const blob = new Blob(segChunks, { type: segChunks[0] ? segChunks[0].type : "audio/webm" });
  segChunks = []; seg = null;
  if (discard || dur < MIN_MS || blob.size < 1500) return;
  el.vstatus.textContent = "thinking…";
  const b64 = await blobToBase64(blob);
  if (!send({ type: "voice_input", audio: b64, mime: blob.type || "audio/webm" }))
    el.vstatus.textContent = "engine offline — try again";
  else if (listening) el.vstatus.textContent = "listening… just speak";
}

function blobToBase64(blob) {
  return new Promise((res) => {
    const r = new FileReader();
    r.onloadend = () => res(String(r.result).split(",")[1] || "");
    r.readAsDataURL(blob);
  });
}

// --- play the AI's spoken reply + drive the blob -------------------------
async function playSpeech(msg) {
  if (!msg.audio) return;
  const ctx = ensureAudioCtx();
  try { await ctx.resume(); } catch {}
  let bytes;
  try { bytes = Uint8Array.from(atob(msg.audio), (c) => c.charCodeAt(0)); }
  catch { return; }

  aiSpeaking = true; blobMode = "speaking"; el.vstatus.textContent = "speaking…";
  const restore = () => {
    aiSpeaking = false;
    if (listening) { analyser = micAnalyser; ampData = micData; blobMode = "listening"; el.vstatus.textContent = "listening… just speak"; }
    else { analyser = null; blobMode = "idle"; el.vstatus.textContent = voiceMode ? "muted — tap the button to talk" : ""; }
  };

  // Preferred path: decode to an AudioBuffer and play through the (resumed)
  // AudioContext. Not subject to <audio> autoplay rules, so it always sounds.
  try {
    const buf = await ctx.decodeAudioData(bytes.buffer.slice(0));
    const src = ctx.createBufferSource(); src.buffer = buf;
    const sa = ctx.createAnalyser(); sa.fftSize = 256;
    const sd = new Uint8Array(sa.frequencyBinCount);
    src.connect(sa); sa.connect(ctx.destination);   // -> speakers
    analyser = sa; ampData = sd;                     // blob reacts to the voice
    src.onended = restore;
    src.start();
    return;
  } catch (e) {
    // Fallback: plain <audio> element.
    try {
      const a = new Audio(`data:${msg.mime || "audio/mpeg"};base64,${msg.audio}`);
      a.onended = restore; a.onerror = restore;
      await a.play();
    } catch { restore(); }
  }
}

// --- the gradient voice blob (canvas, 60fps) -----------------------------
let blobRAF = null, blobT = 0, ampSmooth = 0;
function startBlobLoop() { if (!blobRAF) blobRAF = requestAnimationFrame(drawBlob); }

function currentAmp() {
  if (!analyser) return 0;
  analyser.getByteFrequencyData(ampData);
  let s = 0; for (let i = 0; i < ampData.length; i++) s += ampData[i];
  return Math.min(1, (s / ampData.length) / 128);
}

function drawBlob(ts) {
  blobRAF = requestAnimationFrame(drawBlob);
  const cvs = el.blob, ctx = cvs.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const CSS = 160;
  if (cvs.width !== CSS * dpr) { cvs.width = CSS * dpr; cvs.height = CSS * dpr; }
  const W = cvs.width, H = cvs.height, cx = W / 2, cy = H / 2;
  ctx.clearRect(0, 0, W, H);
  if (el.voicebar.hidden) { blobRAF = null; return; }  // stop when hidden

  blobT += 0.016;
  const target = blobMode === "idle" ? 0.06 : currentAmp();
  ampSmooth += (target - ampSmooth) * 0.15;

  const base = Math.min(W, H) * (blobMode === "idle" ? 0.24 : 0.26);
  const puff = 1 + ampSmooth * (blobMode === "speaking" ? 0.9 : 0.6);
  const hue = (blobT * 26) % 360;
  const listening = blobMode === "listening";

  function petal(hueShift, sat, off, scale, comp) {
    ctx.globalCompositeOperation = comp;
    ctx.beginPath();
    const N = 72;
    for (let i = 0; i <= N; i++) {
      const a = (i / N) * Math.PI * 2;
      const r = base * scale * puff * (
        1 + 0.06 * Math.sin(a * 3 + blobT * 1.3 + off)
          + 0.05 * Math.sin(a * 5 - blobT * 1.7 + off)
          + 0.04 * Math.sin(a * 2 + blobT * 0.9));
      const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.closePath();
    const h = listening ? 205 : (hue + hueShift);
    const g = ctx.createRadialGradient(cx - base * 0.2, cy - base * 0.2, base * 0.1, cx, cy, base * scale * puff * 1.2);
    g.addColorStop(0, `hsla(${h},95%,68%,0.95)`);
    g.addColorStop(0.6, `hsla(${h + 30},90%,58%,0.55)`);
    g.addColorStop(1, `hsla(${h + 60},90%,50%,0)`);
    ctx.fillStyle = g; ctx.fill();
  }
  ctx.save();
  ctx.shadowColor = `hsla(${listening ? 205 : hue},90%,60%,0.5)`; ctx.shadowBlur = 24 * dpr;
  petal(0, 90, 0, 1.0, "source-over");
  petal(120, 90, 2.1, 0.82, "lighter");
  petal(240, 90, 4.2, 0.7, "lighter");
  ctx.restore();
  ctx.globalCompositeOperation = "source-over";
}

connect();
