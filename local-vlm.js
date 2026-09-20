// ===========================================================================
// AgentLab — Local vision model (WebGPU, in-browser) via transformers.js
//
// Runs an open vision-language model entirely on the GPU in the Electron
// renderer — no API key, no cloud, free. The Python engine sends a screenshot
// + prompt over WebSocket; this module runs the model and returns the chosen
// action.
//
// IMPORTANT: transformers.js does NOT expose a `pipeline("image-text-to-text")`
// task, so we use the lower-level AutoProcessor + AutoModelForImageTextToText
// API (the pattern the official SmolVLM/Qwen2-VL WebGPU demos use). Models are
// loaded 4-bit ("q4") by default — optimized: smaller download, less VRAM.
// ===========================================================================

window.LocalVLM = (function () {
  let model = null;
  let processor = null;
  let RawImage = null;
  let loadingModel = null;
  let readyId = null;

  const CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

  function progressCb(report) {
    return (p) => {
      if (p && p.status === "progress" && p.file) {
        report({ status: "progress", file: p.file, pct: Math.round(p.progress || 0) });
      } else if (p && p.status) {
        report({ status: p.status, file: p.file });
      }
    };
  }

  async function load(modelId, onProgress) {
    if (readyId === modelId && model) return;
    loadingModel = modelId;
    const report = (s) => { try { onProgress && onProgress(s); } catch {} };

    report({ status: "init", message: "loading transformers.js…" });
    const TJ = await import(/* @vite-ignore */ CDN);
    const { AutoProcessor, AutoModelForImageTextToText, env } = TJ;
    RawImage = TJ.RawImage;
    env.allowLocalModels = false;   // fetch from the Hugging Face hub
    env.useBrowserCache = true;     // cache weights across runs

    if (!navigator.gpu) {
      throw new Error("WebGPU not available in this build. Update Electron/Chromium or enable WebGPU.");
    }

    // 4-bit = optimized: much smaller download + fits more GPUs.
    const dtype = "q4";
    report({ status: "download", message: `downloading ${modelId} (${dtype}, optimized). First run is large…` });

    processor = await AutoProcessor.from_pretrained(modelId, {
      progress_callback: progressCb(report),
    });
    model = await AutoModelForImageTextToText.from_pretrained(modelId, {
      dtype,
      device: "webgpu",
      progress_callback: progressCb(report),
    });

    readyId = modelId;
    loadingModel = null;
    report({ status: "ready", message: "model ready" });
  }

  // Returns the raw generated assistant text.
  async function generate(imageDataUrl, systemText, userText) {
    if (!model || !processor) throw new Error("model not loaded");

    const image = await RawImage.fromURL(imageDataUrl);
    // Fold the system prompt into the user turn — many VLM chat templates
    // (e.g. Gemma) don't accept a separate "system" role.
    const text = (systemText ? systemText + "\n\n" : "") + userText;
    const messages = [
      { role: "user", content: [{ type: "image" }, { type: "text", text }] },
    ];

    const prompt = processor.apply_chat_template(messages, { add_generation_prompt: true });
    const inputs = await processor(prompt, [image]);
    const generated = await model.generate({
      ...inputs,
      max_new_tokens: 320,
      do_sample: false,
    });

    // Drop the prompt tokens, keep only what the model generated.
    let decoded;
    try {
      const trimmed = generated.slice(null, [inputs.input_ids.dims.at(-1), null]);
      decoded = processor.batch_decode(trimmed, { skip_special_tokens: true });
    } catch {
      decoded = processor.batch_decode(generated, { skip_special_tokens: true });
    }
    return Array.isArray(decoded) ? (decoded[0] || "") : String(decoded);
  }

  return {
    load,
    generate,
    isLoaded: () => !!model,
    isLoading: () => !!loadingModel,
    current: () => readyId,
  };
})();
