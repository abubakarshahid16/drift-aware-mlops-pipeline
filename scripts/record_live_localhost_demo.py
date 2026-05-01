"""Record a real localhost walkthrough video using Playwright.

This is different from the screenshot slideshow. It opens the running local
services and records the browser session while navigating through FastAPI,
Prometheus, MLflow, and Grafana.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo_artifacts"
VIDEO_WEBM = OUT / "live_localhost_walkthrough.webm"
VIDEO_MP4 = OUT / "live_localhost_walkthrough.mp4"
FFMPEG_CANDIDATES = [
    shutil.which("ffmpeg"),
    Path.home() / "AppData/Local/ms-playwright/ffmpeg-1011/ffmpeg-win64.exe",
]

WIDTH = 1366
HEIGHT = 768


def pause(page: Page, ms: int = 2500) -> None:
    page.wait_for_timeout(ms)


def banner(page: Page, text: str) -> None:
    page.evaluate(
        """
        (text) => {
          const existing = document.getElementById("codex-demo-banner");
          if (existing) existing.remove();
          const el = document.createElement("div");
          el.id = "codex-demo-banner";
          el.textContent = text;
          Object.assign(el.style, {
            position: "fixed",
            left: "18px",
            bottom: "18px",
            zIndex: "2147483647",
            maxWidth: "calc(100vw - 36px)",
            padding: "12px 16px",
            borderRadius: "10px",
            background: "rgba(15, 23, 42, 0.94)",
            color: "#fff",
            font: "600 16px Arial, sans-serif",
            boxShadow: "0 12px 30px rgba(0,0,0,.35)",
            border: "1px solid rgba(255,255,255,.18)"
          });
          document.body.appendChild(el);
        }
        """,
        text,
    )


def render_json_page(page: Page, title: str, url: str, payload: object) -> None:
    body = json.dumps(payload, indent=2)
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>{title}</title>
      <style>
        body {{
          margin: 0;
          background: #f8fafc;
          color: #0f172a;
          font-family: Arial, sans-serif;
        }}
        header {{
          background: #0f172a;
          color: white;
          padding: 22px 30px;
        }}
        h1 {{
          margin: 0;
          font-size: 30px;
        }}
        .url {{
          margin-top: 8px;
          color: #93c5fd;
          font-family: Consolas, monospace;
        }}
        main {{
          padding: 28px 34px 96px;
        }}
        pre {{
          white-space: pre-wrap;
          word-break: break-word;
          font-size: 17px;
          line-height: 1.42;
          background: white;
          border: 1px solid #cbd5e1;
          border-radius: 10px;
          padding: 22px;
          box-shadow: 0 10px 28px rgba(15,23,42,.08);
        }}
      </style>
    </head>
    <body>
      <header>
        <h1>{title}</h1>
        <div class="url">{url}</div>
      </header>
      <main><pre>{body}</pre></main>
    </body>
    </html>
    """
    page.set_content(html, wait_until="domcontentloaded")
    banner(page, "Live localhost evidence rendered from the running service")


def fetch_from_page(page: Page, script: str) -> object:
    return page.evaluate(script)


def login_grafana(page: Page) -> None:
    page.goto("http://localhost:3000/login", wait_until="domcontentloaded", timeout=30_000)
    pause(page, 1500)
    if page.locator('input[name="user"]').count():
        page.locator('input[name="user"]').fill("admin")
        page.locator('input[name="password"]').fill("admin")
        page.locator('button[type="submit"]').click()
        pause(page, 2500)

    skip = page.get_by_role("button", name="Skip")
    try:
        skip.wait_for(state="visible", timeout=8_000)
        skip.click(timeout=3_000)
        pause(page, 1500)
    except Exception:
        pass


def record() -> Path:
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            record_video_dir=str(OUT),
            record_video_size={"width": WIDTH, "height": HEIGHT},
        )
        page = context.new_page()

        page.goto("http://localhost:8000/docs", wait_until="networkidle", timeout=30_000)
        banner(page, "Step 1: FastAPI docs running at http://localhost:8000/docs")
        pause(page, 4500)

        page.goto("http://localhost:8000/health", wait_until="domcontentloaded", timeout=30_000)
        banner(page, "Step 2: Live API health endpoint at http://localhost:8000/health")
        pause(page, 3500)

        page.goto("http://localhost:8000/docs", wait_until="domcontentloaded", timeout=30_000)
        payload = fetch_from_page(
            page,
            """
            async () => {
              const response = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                  features: [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
                  label: 1
                })
              });
              return await response.json();
            }
            """,
        )
        render_json_page(
            page, "Step 3: Live POST /predict response", "http://localhost:8000/predict", payload
        )
        pause(page, 4500)

        page.goto("http://localhost:9090/targets", wait_until="networkidle", timeout=30_000)
        banner(page, "Step 4: Prometheus scrape targets at http://localhost:9090/targets")
        pause(page, 4500)

        page.goto("http://localhost:9090", wait_until="domcontentloaded", timeout=30_000)
        prometheus_payload = fetch_from_page(
            page,
            """
            async () => {
              const [predictions, targets] = await Promise.all([
                fetch("/api/v1/query?query=ml_predictions_total").then(r => r.json()),
                fetch("/api/v1/query?query=up").then(r => r.json())
              ]);
              return {ml_predictions_total: predictions.data.result, up_targets: targets.data.result};
            }
            """,
        )
        render_json_page(
            page,
            "Step 5: Live Prometheus metric query",
            "http://localhost:9090/api/v1/query?query=ml_predictions_total",
            prometheus_payload,
        )
        pause(page, 5000)

        page.goto("http://localhost:5000", wait_until="domcontentloaded", timeout=30_000)
        mlflow_payload = fetch_from_page(
            page,
            """
            async () => {
              const experiments = await fetch("/api/2.0/mlflow/experiments/search", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({max_results: 10})
              }).then(r => r.json());
              const runs = await fetch("/api/2.0/mlflow/runs/search", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({experiment_ids: ["1"], max_results: 3})
              }).then(r => r.json());
              return {experiments, runs};
            }
            """,
        )
        render_json_page(
            page,
            "Step 6: Live MLflow tracking evidence",
            "http://localhost:5000",
            mlflow_payload,
        )
        pause(page, 5000)

        login_grafana(page)
        page.goto(
            "http://localhost:3000/d/model-performance/model-performance?orgId=1&from=now-15m&to=now",
            wait_until="networkidle",
            timeout=45_000,
        )
        banner(page, "Step 7: Grafana Model Performance dashboard at http://localhost:3000")
        pause(page, 5500)

        page.goto(
            "http://localhost:3000/d/drift-monitoring/drift-and-adaptive-retraining?orgId=1&from=now-15m&to=now",
            wait_until="networkidle",
            timeout=45_000,
        )
        banner(page, "Step 8: Grafana Drift and Adaptive Retraining dashboard")
        pause(page, 5500)

        page.goto(
            "http://localhost:3000/d/infra-health/infrastructure-and-api-health?orgId=1&from=now-15m&to=now",
            wait_until="networkidle",
            timeout=45_000,
        )
        banner(page, "Step 9: Grafana Infrastructure and API Health dashboard")
        pause(page, 5500)

        video = page.video
        context.close()
        browser.close()
        webm_path = Path(video.path())

    if VIDEO_WEBM.exists():
        VIDEO_WEBM.unlink()
    webm_path.replace(VIDEO_WEBM)
    return VIDEO_WEBM


def convert_to_mp4(webm: Path) -> None:
    ffmpeg = next(
        (
            Path(candidate)
            for candidate in FFMPEG_CANDIDATES
            if candidate and Path(candidate).exists()
        ),
        None,
    )
    if ffmpeg is None:
        print("ffmpeg not found; kept the WebM recording only")
        return
    if VIDEO_MP4.exists():
        VIDEO_MP4.unlink()
    command = [
        str(ffmpeg),
        "-y",
        "-i",
        str(webm),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(VIDEO_MP4),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        # The ffmpeg bundled with Playwright records WebM but cannot encode MP4.
        print(f"MP4 conversion failed with {ffmpeg}; trying OpenCV fallback")
        convert_to_mp4_with_opencv(webm)


def convert_to_mp4_with_opencv(webm: Path) -> None:
    try:
        import cv2
    except ImportError:
        print(f"OpenCV not installed; kept {webm.name}")
        return

    capture = cv2.VideoCapture(str(webm))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(VIDEO_MP4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    frames = 0
    ok, frame = capture.read()
    while ok:
        writer.write(frame)
        frames += 1
        ok, frame = capture.read()
    capture.release()
    writer.release()
    if frames == 0 or not VIDEO_MP4.exists():
        print(f"OpenCV MP4 conversion failed; kept {webm.name}")


def main() -> None:
    webm = record()
    convert_to_mp4(webm)
    if VIDEO_MP4.exists():
        print(f"created {VIDEO_MP4} ({VIDEO_MP4.stat().st_size} bytes)")
    print(f"created {VIDEO_WEBM} ({VIDEO_WEBM.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
