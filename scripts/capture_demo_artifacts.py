"""Capture demo screenshots and build an MP4 walkthrough bundle."""
from __future__ import annotations

import json
import subprocess
import textwrap
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo_artifacts"
SHOTS = OUT / "screenshots"
VIDEO = OUT / "mlops_live_demo.mp4"
ZIP_PATH = OUT / "MLOps_Demo_Pack.zip"

VIEWPORT = {"width": 1366, "height": 768}
SLIDE_SIZE = (1280, 720)
FPS = 30
SECONDS_PER_STEP = 3


def ensure_dirs() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def run_command(command: list[str], timeout: int = 60) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = completed.stdout.strip()
        if completed.stderr.strip():
            output += "\n\nSTDERR:\n" + completed.stderr.strip()
        return f"$ {' '.join(command)}\nexit={completed.returncode}\n\n{output}"
    except Exception as exc:  # noqa: BLE001
        return f"$ {' '.join(command)}\nERROR: {exc}"


def pretty_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def smoke_report() -> str:
    sections: list[str] = []

    sections.append(run_command(["docker", "compose", "ps"], timeout=30))

    try:
        sections.append("GET /health\n" + pretty_json(requests.get("http://localhost:8000/health", timeout=10).json()))
    except Exception as exc:  # noqa: BLE001
        sections.append(f"GET /health failed: {exc}")

    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "label": 1},
            timeout=10,
        )
        sections.append("POST /predict\n" + pretty_json(response.json()))
    except Exception as exc:  # noqa: BLE001
        sections.append(f"POST /predict failed: {exc}")

    try:
        prom = requests.get("http://localhost:9090/api/v1/targets", timeout=10).json()
        targets = [
            {
                "job": item["labels"].get("job"),
                "health": item.get("health"),
                "scrapeUrl": item.get("scrapeUrl"),
            }
            for item in prom["data"]["activeTargets"]
        ]
        sections.append("Prometheus targets\n" + pretty_json(targets))
    except Exception as exc:  # noqa: BLE001
        sections.append(f"Prometheus targets failed: {exc}")

    try:
        dashboards = requests.get(
            "http://localhost:3000/api/search",
            auth=("admin", "admin"),
            timeout=10,
        ).json()
        sections.append(
            "Grafana dashboards\n"
            + pretty_json([{"title": d.get("title"), "url": d.get("url")} for d in dashboards])
        )
    except Exception as exc:  # noqa: BLE001
        sections.append(f"Grafana dashboards failed: {exc}")

    try:
        runs = requests.post(
            "http://localhost:5000/api/2.0/mlflow/runs/search",
            json={"experiment_ids": ["1"], "max_results": 3},
            timeout=10,
        ).json()
        compact = [
            {
                "run_name": run["info"].get("run_name"),
                "status": run["info"].get("status"),
                "metrics": {m["key"]: m["value"] for m in run["data"].get("metrics", [])},
            }
            for run in runs.get("runs", [])
        ]
        sections.append("MLflow runs\n" + pretty_json(compact))
    except Exception as exc:  # noqa: BLE001
        sections.append(f"MLflow runs failed: {exc}")

    return "\n\n" + ("=" * 88 + "\n\n").join(sections)


def render_text_screenshot(title: str, text: str, path: Path) -> None:
    img = Image.new("RGB", (VIEWPORT["width"], VIEWPORT["height"]), "#f7f7f7")
    draw = ImageDraw.Draw(img)
    title_font = font(30, bold=True)
    body_font = font(17)
    draw.rectangle((0, 0, VIEWPORT["width"], 62), fill="#111827")
    draw.text((28, 16), title, fill="white", font=title_font)

    y = 88
    line_height = 23
    max_chars = 126
    for raw in text.splitlines():
        wrapped = textwrap.wrap(raw, width=max_chars, replace_whitespace=False) or [""]
        for line in wrapped:
            if y > VIEWPORT["height"] - 35:
                draw.text((28, y), "... output trimmed for screenshot", fill="#374151", font=body_font)
                img.save(path)
                return
            draw.text((28, y), line, fill="#111827", font=body_font)
            y += line_height
    img.save(path)


def wait_soft(page, ms: int) -> None:
    page.wait_for_timeout(ms)


def capture_page(page, title: str, url: str, path: Path, wait_ms: int = 2000) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    wait_soft(page, wait_ms)
    page.screenshot(path=str(path), full_page=True)


def login_grafana(page) -> None:
    page.goto("http://localhost:3000/login", wait_until="domcontentloaded", timeout=30_000)
    wait_soft(page, 1500)
    if page.locator('input[name="user"]').count() > 0:
        page.locator('input[name="user"]').fill("admin")
        page.locator('input[name="password"]').fill("admin")
        page.locator('button[type="submit"]').click()
        wait_soft(page, 2500)
    for selector in [
        'button:has-text("Skip")',
        'a:has-text("Skip")',
        'button:has-text("Skip password change")',
    ]:
        try:
            loc = page.locator(selector)
            if loc.count() > 0 and loc.first.is_visible():
                loc.first.click()
                wait_soft(page, 1500)
                break
        except Exception:  # noqa: BLE001
            continue


def capture_browser_screenshots() -> list[tuple[Path, str]]:
    screenshots: list[tuple[Path, str]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()

        steps = [
            ("02_api_docs.png", "FastAPI Swagger docs", "http://localhost:8000/docs", 2500),
            ("03_api_health.png", "FastAPI health response", "http://localhost:8000/health", 1000),
            ("05_prometheus_targets.png", "Prometheus targets", "http://localhost:9090/targets", 2500),
        ]
        for filename, title, url, wait_ms in steps:
            path = SHOTS / filename
            capture_page(page, title, url, path, wait_ms)
            screenshots.append((path, title))

        login_grafana(page)
        grafana_steps = [
            (
                "08_grafana_model_performance.png",
                "Grafana model performance dashboard",
                "http://localhost:3000/d/model-performance/model-performance?orgId=1&from=now-15m&to=now",
            ),
            (
                "09_grafana_drift_dashboard.png",
                "Grafana drift and retraining dashboard",
                "http://localhost:3000/d/drift-monitoring/drift-and-adaptive-retraining?orgId=1&from=now-15m&to=now",
            ),
            (
                "10_grafana_infra_dashboard.png",
                "Grafana infrastructure dashboard",
                "http://localhost:3000/d/infra-health/infrastructure-and-api-health?orgId=1&from=now-15m&to=now",
            ),
        ]
        for filename, title, url in grafana_steps:
            path = SHOTS / filename
            capture_page(page, title, url, path, 5000)
            screenshots.append((path, title))

        context.close()
        browser.close()

    return screenshots


def capture_predict_page(title: str, path: Path) -> None:
    response = requests.post(
        "http://localhost:8000/predict",
        json={"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "label": 1},
        timeout=10,
    )
    render_text_screenshot(title, "POST http://localhost:8000/predict\n\n" + pretty_json(response.json()), path)


def capture_prometheus_metrics_page(title: str, path: Path) -> None:
    metrics = {
        "ml_predictions_total": requests.get(
            "http://localhost:9090/api/v1/query",
            params={"query": "ml_predictions_total"},
            timeout=10,
        ).json(),
        "up_targets": requests.get(
            "http://localhost:9090/api/v1/query",
            params={"query": "up"},
            timeout=10,
        ).json(),
    }
    render_text_screenshot(title, "Prometheus API query evidence\n\n" + pretty_json(metrics), path)


def capture_mlflow_page(title: str, path: Path) -> None:
    experiments = requests.post(
        "http://localhost:5000/api/2.0/mlflow/experiments/search",
        json={"max_results": 10},
        timeout=10,
    ).json()
    runs = requests.post(
        "http://localhost:5000/api/2.0/mlflow/runs/search",
        json={"experiment_ids": ["1"], "max_results": 5},
        timeout=10,
    ).json()
    compact_runs = [
        {
            "run_name": run["info"].get("run_name"),
            "status": run["info"].get("status"),
            "artifact_uri": run["info"].get("artifact_uri"),
            "params": {p["key"]: p["value"] for p in run["data"].get("params", [])},
            "metrics": {m["key"]: m["value"] for m in run["data"].get("metrics", [])},
        }
        for run in runs.get("runs", [])
    ]
    payload = {"experiments": experiments.get("experiments", []), "runs": compact_runs}
    render_text_screenshot(title, "MLflow tracking API evidence\n\n" + pretty_json(payload), path)


def fit_image_to_slide(src: Path, title: str) -> Image.Image:
    base = Image.new("RGB", SLIDE_SIZE, "#0f172a")
    draw = ImageDraw.Draw(base)
    title_font = font(25, bold=True)
    draw.rectangle((0, 0, SLIDE_SIZE[0], 54), fill="#111827")
    draw.text((26, 15), title, fill="white", font=title_font)

    img = Image.open(src).convert("RGB")
    max_w = SLIDE_SIZE[0] - 42
    max_h = SLIDE_SIZE[1] - 76
    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    x = (SLIDE_SIZE[0] - img.width) // 2
    y = 62 + (max_h - img.height) // 2
    base.paste(img, (x, y))
    return base


def build_video(slides: list[tuple[Path, str]]) -> None:
    writer = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SLIDE_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {VIDEO}")

    frames_per_slide = FPS * SECONDS_PER_STEP
    for path, title in slides:
        slide = fit_image_to_slide(path, title)
        arr = cv2.cvtColor(np.array(slide), cv2.COLOR_RGB2BGR)
        for _ in range(frames_per_slide):
            writer.write(arr)
    writer.release()


def verify_video() -> dict[str, float]:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not read generated video: {VIDEO}")
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {"frames": frames, "fps": fps, "duration_s": frames / fps if fps else 0}


def write_readme(slides: list[tuple[Path, str]], video_info: dict[str, float]) -> Path:
    readme = OUT / "README_demo_artifacts.txt"
    lines = [
        "MLOps live demo artifacts",
        "",
        f"Generated from: {ROOT}",
        f"Video: {VIDEO.name}",
        f"Video duration: {video_info['duration_s']:.1f}s",
        "",
        "Screenshots:",
    ]
    for path, title in slides:
        lines.append(f"- {path.name}: {title}")
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme


def make_zip(files: list[Path]) -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            zf.write(file, file.relative_to(OUT))


def main() -> None:
    ensure_dirs()

    slides: list[tuple[Path, str]] = []
    evidence_path = SHOTS / "01_live_stack_and_smoke_checks.png"
    render_text_screenshot("Live stack and smoke checks", smoke_report(), evidence_path)
    slides.append((evidence_path, "Live stack and smoke checks"))

    slides.extend(capture_browser_screenshots())

    predict_path = SHOTS / "04_predict_response.png"
    capture_predict_page("Prediction API response", predict_path)
    slides.insert(3, (predict_path, "Prediction API response"))

    prometheus_path = SHOTS / "06_prometheus_prediction_metrics.png"
    capture_prometheus_metrics_page("Prometheus prediction metrics", prometheus_path)
    slides.insert(5, (prometheus_path, "Prometheus prediction metrics"))

    mlflow_path = SHOTS / "07_mlflow_experiment.png"
    capture_mlflow_page("MLflow experiment tracking", mlflow_path)
    slides.insert(6, (mlflow_path, "MLflow experiment tracking"))

    build_video(slides)
    video_info = verify_video()
    readme = write_readme(slides, video_info)
    make_zip([path for path, _ in slides] + [VIDEO, readme])

    print("created screenshots:")
    for path, title in slides:
        print(f"- {path} :: {title}")
    print(f"created video: {VIDEO} ({video_info['duration_s']:.1f}s)")
    print(f"created zip: {ZIP_PATH}")


if __name__ == "__main__":
    main()
