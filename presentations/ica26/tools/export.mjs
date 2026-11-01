// Renders the deck and the speaker script to PDF with headless Chrome.
//
//   cd presentations/ica26/tools && npm install && node export.mjs
//
// Writes ../export/AgriPerceiver-ICA26.pdf (one 1920×1080 page per slide,
// every build at its final state) and ../export/AgriPerceiver-ICA26-speaker-script.pdf.
// Serves the folder itself, so no other server needs to be running.
// CHROME=<path to chrome.exe> overrides browser detection.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.join(root, "export");
fs.mkdirSync(out, { recursive: true });

const { default: puppeteer } = await import(process.env.PPTR ? pathToFileURL(process.env.PPTR).href : "puppeteer-core");

const chrome = process.env.CHROME || [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].find((p) => fs.existsSync(p));
if (!chrome) throw new Error("No Chrome/Edge found; set CHROME=/path/to/chrome");

const types = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml", ".woff2": "font/woff2", ".pdf": "application/pdf", ".png": "image/png" };
const server = http.createServer((req, res) => {
  const p = path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "Content-Type": types[path.extname(p)] || "application/octet-stream" });
  fs.createReadStream(p).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

const browser = await puppeteer.launch({ executablePath: chrome });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });
  await page.goto(`${base}/index.html?print#/1`, { waitUntil: "networkidle0" });
  await page.evaluate(() => document.fonts.ready);
  await page.emulateMediaType("print");
  const deck = path.join(out, "AgriPerceiver-ICA26.pdf");
  await page.pdf({ path: deck, preferCSSPageSize: true, printBackground: true });
  console.log("wrote", path.relative(root, deck));

  const notes = await browser.newPage();
  await notes.goto(`${base}/presenter.html?script`, { waitUntil: "networkidle0" });
  await notes.waitForSelector(".script-doc");
  await notes.evaluate(() => document.fonts.ready);
  const script = path.join(out, "AgriPerceiver-ICA26-speaker-script.pdf");
  await notes.pdf({ path: script, preferCSSPageSize: true, printBackground: true });
  console.log("wrote", path.relative(root, script));
} finally {
  await browser.close();
  server.close();
}
