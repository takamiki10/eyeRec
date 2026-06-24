"""Local browser GUI for capturing a face photo and running EyeRec."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
INPUT_FACES_DIR = PROJECT_ROOT / "data" / "input_faces"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "recommendations" / "top3_recommendations.json"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eyewear_system.pipeline.full_pipeline import EyewearRecommendationPipeline


class EyeRecState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pipeline: EyewearRecommendationPipeline | None = None
        self.current_image_path: Path | None = None

    def save_capture(self, image_data_url: str) -> dict:
        prefix = "data:image/jpeg;base64,"
        if not image_data_url.startswith(prefix):
            raise ValueError("Expected a JPEG capture from the browser camera.")

        INPUT_FACES_DIR.mkdir(parents=True, exist_ok=True)
        image_path = INPUT_FACES_DIR / f"eyerec_capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        image_path.write_bytes(base64.b64decode(image_data_url.removeprefix(prefix)))

        with self.lock:
            self.current_image_path = image_path

        return {
            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            "image_url": f"/input_faces/{image_path.name}",
        }

    def run_pipeline(self) -> dict:
        with self.lock:
            image_path = self.current_image_path
            if self.pipeline is None:
                self.pipeline = EyewearRecommendationPipeline()
            pipeline = self.pipeline

        if image_path is None:
            raise ValueError("Capture a photo before running EyeRec.")

        result = pipeline.run(image_path, include_debug=False)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


STATE = EyeRecState()


class EyeRecRequestHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self._send_html(HTML)
            return

        if route.startswith("/input_faces/"):
            self._send_input_face(route.removeprefix("/input_faces/"))
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            if route == "/api/capture":
                payload = self._read_json()
                self._send_json(STATE.save_capture(str(payload.get("image", ""))))
                return

            if route == "/api/run":
                result = STATE.run_pipeline()
                self._send_json(
                    {
                        "result": result,
                        "formatted": format_result(result),
                        "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
                    }
                )
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        return json.loads(raw_body.decode("utf-8")) if raw_body else {}

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_input_face(self, filename: str) -> None:
        path = (INPUT_FACES_DIR / filename).resolve()
        if not path.is_file() or INPUT_FACES_DIR.resolve() not in path.parents:
            self._send_json({"error": "Image not found"}, status=HTTPStatus.NOT_FOUND)
            return

        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def format_result(result: dict) -> str:
    features = result["detected_features"]
    rule_based = result["rule_based"]
    dnn = result["dnn"]

    lines = ["Detected features", ""]
    for key in ("face_shape", "eye_shape", "eye_color", "pupil_distance"):
        value = format_value(features[key])
        lines.append(f"{key.replace('_', ' ').title()}: {value}")

    lines.extend(
        [
            "",
            "Recommended fit",
            f"Shapes: {join_values(rule_based.get('best_shapes', []))}",
            f"Colors: {join_values(rule_based.get('best_colors', []))}",
            f"Bridge: {rule_based.get('bridge_fit')}",
            f"Avoid: {join_values(rule_based.get('avoid', []))}",
            "",
            str(rule_based.get("summary", "")),
            "",
            "Top picks",
        ]
    )
    for item in dnn.get("top_picks", []):
        lines.append(f"{item['rank']}. {item['frame']} - {item['style']}")
    return "\n".join(lines)


def format_value(value: object) -> object:
    if isinstance(value, float):
        return round(value, 3)
    return value


def join_values(values: list) -> str:
    return ", ".join(str(value) for value in values) if values else "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EyeRec browser GUI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local GUI server.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local GUI server.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), EyeRecRequestHandler)
    url = f"http://{args.host}:{args.port}"

    print(f"eyeRec GUI running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped eyeRec GUI.")
    finally:
        server.server_close()


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>eyeRec</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #141712;
      --muted: #687064;
      --paper: #f7f8f5;
      --panel: #ffffff;
      --line: #d9ded2;
      --soft: #eef1ea;
      --accent: #9fb7a0;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    main {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(360px, 0.85fr);
      gap: 0;
    }

    .camera-side {
      padding: 32px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 18px;
    }

    h1 {
      margin: 0;
      font-size: clamp(36px, 5vw, 68px);
      line-height: 0.95;
      letter-spacing: 0;
      font-weight: 750;
    }

    .subtitle {
      margin-top: 8px;
      color: var(--muted);
      font-size: 15px;
    }

    .viewfinder {
      min-height: 420px;
      background: #0f120f;
      border: 1px solid var(--line);
      display: grid;
      place-items: center;
      overflow: hidden;
    }

    video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: scaleX(-1);
    }

    .empty {
      color: #e7ebe1;
      padding: 24px;
      text-align: center;
    }

    .actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    button {
      border: 0;
      background: #e7ebe1;
      color: var(--ink);
      padding: 14px 16px;
      font: inherit;
      font-weight: 720;
      cursor: pointer;
      min-height: 48px;
    }

    button:hover {
      background: #dfe5d8;
    }

    button.primary {
      background: var(--ink);
      color: #fff;
    }

    button.primary:hover {
      background: #2b3128;
    }

    button:disabled {
      cursor: wait;
      opacity: 0.58;
    }

    .results-side {
      background: var(--panel);
      border-left: 1px solid var(--line);
      padding: 32px;
      display: grid;
      grid-template-rows: auto auto auto minmax(0, 1fr);
      gap: 18px;
    }

    h2 {
      margin: 0;
      font-size: 28px;
      letter-spacing: 0;
    }

    .status {
      color: var(--muted);
      line-height: 1.45;
      min-height: 44px;
    }

    .capture-preview {
      min-height: 170px;
      background: var(--soft);
      border: 1px solid var(--line);
      display: grid;
      place-items: center;
      color: var(--muted);
      overflow: hidden;
    }

    .capture-preview img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow: auto;
      line-height: 1.45;
      font-size: 14px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }

    canvas {
      display: none;
    }

    @media (max-width: 900px) {
      main {
        grid-template-columns: 1fr;
      }

      .results-side {
        border-left: 0;
        border-top: 1px solid var(--line);
      }

      .actions {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="camera-side">
      <div>
        <h1>eyeRec</h1>
        <div class="subtitle">Capture a face photo and run the recommendation model.</div>
      </div>
      <div class="viewfinder">
        <video id="video" autoplay playsinline></video>
        <div class="empty" id="empty" hidden>Camera access is unavailable.</div>
      </div>
      <div class="actions">
        <button id="capture">Capture</button>
        <button class="primary" id="run">eyeRec</button>
      </div>
    </section>
    <section class="results-side">
      <h2>Results</h2>
      <div class="status" id="status">Allow camera access, then capture a photo.</div>
      <div class="capture-preview" id="preview">No photo captured yet</div>
      <pre id="results">Capture a face photo, then click eyeRec.</pre>
    </section>
  </main>
  <canvas id="canvas"></canvas>
  <script>
    const video = document.getElementById("video");
    const empty = document.getElementById("empty");
    const canvas = document.getElementById("canvas");
    const statusEl = document.getElementById("status");
    const preview = document.getElementById("preview");
    const results = document.getElementById("results");
    const captureButton = document.getElementById("capture");
    const runButton = document.getElementById("run");
    let stream = null;
    let hasCapture = false;

    async function startCamera() {
      try {
        if (stream) {
          stream.getTracks().forEach((track) => track.stop());
        }
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 900 } },
          audio: false
        });
        video.srcObject = stream;
        video.hidden = false;
        empty.hidden = true;
        statusEl.textContent = "Camera ready. Center your face and capture a photo.";
      } catch (error) {
        video.hidden = true;
        empty.hidden = false;
        statusEl.textContent = "Camera permission was blocked or unavailable.";
        results.textContent = error.name + ": " + error.message;
      }
    }

    async function postJson(url, payload = {}) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }
      return data;
    }

    captureButton.addEventListener("click", async () => {
      if (!video.videoWidth) {
        statusEl.textContent = "Camera is not ready yet.";
        return;
      }
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      context.translate(canvas.width, 0);
      context.scale(-1, 1);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const image = canvas.toDataURL("image/jpeg", 0.92);

      captureButton.disabled = true;
      statusEl.textContent = "Saving capture...";
      try {
        const data = await postJson("/api/capture", { image });
        hasCapture = true;
        preview.innerHTML = "";
        const img = document.createElement("img");
        img.src = data.image_url + "?t=" + Date.now();
        preview.appendChild(img);
        statusEl.textContent = "Saved to " + data.image_path + ".";
      } catch (error) {
        statusEl.textContent = "Could not save the capture.";
        results.textContent = error.message;
      } finally {
        captureButton.disabled = false;
      }
    });

    runButton.addEventListener("click", async () => {
      if (!hasCapture) {
        statusEl.textContent = "Capture a photo before running EyeRec.";
        return;
      }
      runButton.disabled = true;
      runButton.textContent = "Running...";
      statusEl.textContent = "Running EyeRec models...";
      results.textContent = "Analyzing image...";
      try {
        const data = await postJson("/api/run");
        statusEl.textContent = "Done. Saved results to " + data.output_path + ".";
        results.textContent = data.formatted;
      } catch (error) {
        statusEl.textContent = "EyeRec could not finish.";
        results.textContent = error.message;
      } finally {
        runButton.disabled = false;
        runButton.textContent = "eyeRec";
      }
    });

    startCamera();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
