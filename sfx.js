// ===========================================================================
// AgentLab — procedural sound effects (Web Audio, no asset files).
// Every sound is synthesised on the fly, so nothing ships from any video.
// Exposes window.SFX with named cues + a persisted mute toggle.
// ===========================================================================
(function () {
  let ctx = null;
  let muted = false;
  try { muted = localStorage.getItem("agentlab.muted") === "1"; } catch {}

  function ac() {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) ctx = new AC();
    }
    if (ctx && ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  // One oscillator "note" with an ADSR-ish gain envelope.
  function tone(freq, start, dur, type, peak) {
    const c = ac();
    if (!c) return;
    const osc = c.createOscillator();
    const gain = c.createGain();
    osc.type = type || "sine";
    osc.frequency.setValueAtTime(freq, c.currentTime + start);
    const t0 = c.currentTime + start;
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.exponentialRampToValueAtTime(peak || 0.18, t0 + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.connect(gain).connect(c.destination);
    osc.start(t0);
    osc.stop(t0 + dur + 0.02);
  }

  const cues = {
    // spawn: playful two-note rise
    boop() { tone(523.25, 0, 0.12, "sine", 0.16); tone(783.99, 0.09, 0.16, "sine", 0.16); },
    // action: tiny tick (kept quiet + throttled by callers)
    tick() { tone(1200, 0, 0.04, "triangle", 0.05); },
    // done: pleasant major triad chime
    chime() {
      tone(659.25, 0, 0.5, "sine", 0.14);
      tone(830.61, 0.06, 0.5, "sine", 0.12);
      tone(987.77, 0.12, 0.6, "sine", 0.12);
    },
    // kill: short descending thunk
    thunk() { tone(300, 0, 0.16, "sawtooth", 0.12); tone(160, 0.06, 0.18, "sawtooth", 0.1); },
    // ui: soft click
    click() { tone(440, 0, 0.05, "square", 0.06); },
  };

  let lastTick = 0;
  const SFX = {
    play(name) {
      if (muted || !cues[name]) return;
      if (name === "tick") { const now = Date.now(); if (now - lastTick < 140) return; lastTick = now; }
      try { cues[name](); } catch {}
    },
    isMuted() { return muted; },
    setMuted(v) {
      muted = !!v;
      try { localStorage.setItem("agentlab.muted", muted ? "1" : "0"); } catch {}
      if (!muted) { try { cues.click(); } catch {} }
      return muted;
    },
    toggle() { return this.setMuted(!muted); },
  };

  window.SFX = SFX;
})();
