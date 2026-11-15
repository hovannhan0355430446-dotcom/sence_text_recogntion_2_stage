import io
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from pipeline import SceneTextPipeline, encode_image_base64

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SAMPLES_DIR = PROJECT_ROOT / "img_real_life"

app = FastAPI(title="Scene Text Recognition Pipeline")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
BENCHMARK_SUMMARY_PATH = BASE_DIR / "benchmark" / "benchmark_summary.json"

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = SceneTextPipeline(base_dir=BASE_DIR)
    return pipeline

@app.on_event("startup")
def startup_event():
    get_pipeline()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/benchmark")
async def get_benchmark():
    if not BENCHMARK_SUMMARY_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Chưa có báo cáo benchmark. Hãy chạy full_pipeline\\benchmark\\benchmark.py.",
        )
    try:
        return json.loads(BENCHMARK_SUMMARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Báo cáo benchmark không hợp lệ.") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Không thể đọc báo cáo benchmark.") from exc


@app.get("/api/samples")
async def get_samples():
    samples = []
    if SAMPLES_DIR.is_dir():
        for path in sorted(SAMPLES_DIR.glob("*")):
            if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                bgr = cv2.imread(str(path))
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    scale = 120.0 / max(h, w)
                    thumb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
                    samples.append({
                        "name": path.name,
                        "thumbnail": encode_image_base64(thumb, quality=70),
                    })
    return {"samples": samples}

@app.post("/api/predict")
async def predict_uploaded_file(
    file: UploadFile = File(...),
    thresh: float = Form(0.3),
    box_thresh: float = Form(0.5),
    unclip_ratio: float = Form(1.7),
):
    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        rgb = np.array(pil_img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file ảnh: {exc}")

    pipe = get_pipeline()
    result = pipe.run(rgb, thresh=thresh, box_thresh=box_thresh, unclip_ratio=unclip_ratio)
    result["original_base64"] = encode_image_base64(rgb, quality=90)
    return result

@app.post("/api/predict_sample")
async def predict_sample_image(
    sample_name: str = Form(...),
    thresh: float = Form(0.3),
    box_thresh: float = Form(0.5),
    unclip_ratio: float = Form(1.7),
):
    target_path = SAMPLES_DIR / sample_name
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh mẫu: {sample_name}")

    bgr = cv2.imread(str(target_path))
    if bgr is None:
        raise HTTPException(status_code=400, detail="Không thể tải ảnh mẫu.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pipe = get_pipeline()
    result = pipe.run(rgb, thresh=thresh, box_thresh=box_thresh, unclip_ratio=unclip_ratio)
    result["original_base64"] = encode_image_base64(rgb, quality=90)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
