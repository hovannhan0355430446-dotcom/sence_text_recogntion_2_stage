import base64
import time
from pathlib import Path
import cv2
import numpy as np

from detection import TextDetector
from recognition import TextRecognizer

def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    if cv2.contourArea(ordered.reshape(-1, 1, 2), oriented=True) < 0:
        ordered = ordered[::-1]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)

def crop_with_padding(img_rgb: np.ndarray, pts: np.ndarray, pad_ratio: float = 0.08) -> np.ndarray:
    if len(pts) != 4:
        rect = cv2.minAreaRect(pts.astype(np.float32))
        pts = cv2.boxPoints(rect)

    rect = order_points_clockwise(pts)
    w = max(4, int(round(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))))
    h = max(4, int(round(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[3] - rect[0])))))

    center = rect.mean(axis=0)
    expanded = rect + (rect - center) * pad_ratio

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(expanded, dst)
    warped = cv2.warpPerspective(img_rgb, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if warped.shape[0] > warped.shape[1] * 1.5 and warped.shape[0] > 32:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return warped

def sort_reading_order(boxes_with_scores: list) -> list:
    if not boxes_with_scores:
        return []

    entries = []
    for poly, score in boxes_with_scores:
        ymin = float(poly[:, 1].min())
        ymax = float(poly[:, 1].max())
        xmin = float(poly[:, 0].min())
        xmax = float(poly[:, 0].max())
        ymid = (ymin + ymax) / 2.0
        h = max(ymax - ymin, 1.0)
        entries.append({
            "poly": poly,
            "score": score,
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "xmax": xmax,
            "ymid": ymid,
            "h": h,
        })

    entries.sort(key=lambda item: item["ymin"])

    lines = []
    for item in entries:
        placed = False
        for line in lines:
            line_ymid = sum(e["ymid"] for e in line) / len(line)
            line_h = sum(e["h"] for e in line) / len(line)
            if abs(item["ymid"] - line_ymid) < line_h * 0.5:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    lines.sort(key=lambda line: sum(e["ymid"] for e in line) / len(line))

    sorted_entries = []
    for line in lines:
        line.sort(key=lambda item: item["xmin"])
        sorted_entries.extend(line)

    return [(e["poly"], e["score"]) for e in sorted_entries]

def encode_image_base64(img_rgb: np.ndarray, quality: int = 90) -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", bgr, params)
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

class SceneTextPipeline:
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent

        weights_dir = base_dir / "weights"
        det_weights = weights_dir / "dbnet_best.pth"
        rec_weights = weights_dir / "parseq_vietnamese_best.pth"
        vocab_path = weights_dir / "vocab.json"

        self.detector = TextDetector(str(det_weights))
        self.recognizer = TextRecognizer(str(rec_weights), str(vocab_path))

    def run(
        self,
        image_rgb: np.ndarray,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.7,
        target_size: int = 800,
    ):
        t0 = time.time()
        raw_polys, raw_scores = self.detector.detect(
            image_rgb,
            target_size=target_size,
            thresh=thresh,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
        )
        t_det = time.time() - t0

        boxes_with_scores = list(zip(raw_polys, raw_scores))
        sorted_items = sort_reading_order(boxes_with_scores)

        crops = []
        valid_items = []
        for poly, score in sorted_items:
            crop = crop_with_padding(image_rgb, poly, pad_ratio=0.08)
            if crop is not None and crop.shape[0] >= 4 and crop.shape[1] >= 4:
                crops.append(crop)
                valid_items.append((poly, score))

        t1 = time.time()
        rec_results = self.recognizer.predict_batch(crops) if crops else []
        t_rec = time.time() - t1

        items = []
        texts = []
        annotated = image_rgb.copy()

        for idx, ((poly, det_score), (text, rec_score), crop) in enumerate(zip(valid_items, rec_results, crops), start=1):
            pts_int = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts_int], True, (0, 220, 100), 2)

            x_min = int(poly[:, 0].min())
            y_min = int(poly[:, 1].min())
            cv2.putText(
                annotated,
                f"{idx}",
                (x_min, max(y_min - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

            crop_b64 = encode_image_base64(crop, quality=85)
            items.append({
                "id": idx,
                "text": text,
                "det_score": round(float(det_score), 4),
                "rec_score": round(float(rec_score), 4),
                "polygon": poly.round(1).tolist(),
                "crop_base64": crop_b64,
            })
            if text:
                texts.append(text)

        total_time = time.time() - t0

        return {
            "items": items,
            "full_text": " ".join(texts),
            "timings": {
                "detection": round(t_det, 3),
                "recognition": round(t_rec, 3),
                "total": round(total_time, 3),
            },
            "image_size": {
                "width": image_rgb.shape[1],
                "height": image_rgb.shape[0],
            },
            "annotated_base64": encode_image_base64(annotated, quality=90),
        }

