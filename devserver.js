// ===========================================================================
// AgentLab — tiny zero-dependency static file server.
// Used ONLY to preview the frontend (control panel / overlay) in a browser.
// The real app runs via Electron (`npm start`); this just serves the HTML/CSS/JS
// so the UI can be inspected in the preview pane without a desktop window.
// ===========================================================================
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = Number(process.env.PORT) || 8791;
const ROOT = __dirname;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".ico": "image/x-icon",
};

const server = http.createServer((req, res) => {
  let urlPath = decodeURIComponent(req.url.split("?")[0]);
  if (urlPath === "/") urlPath = "/control.html";

  // Resolve safely inside ROOT (prevent path traversal).
  const filePath = path.normalize(path.join(ROOT, urlPath));
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403).end("Forbidden");
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain" }).end("Not found: " + urlPath);
      return;
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream" });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`AgentLab UI preview  ->  http://localhost:${PORT}/control.html`);
  console.log(`                         http://localhost:${PORT}/overlay.html`);
  console.log("(This is a preview server only. Run the real app with: npm start)");
});
