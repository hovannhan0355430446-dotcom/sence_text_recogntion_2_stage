import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import SceneTextPipeline

IOU_THRESHOLD = 0.5
LEXICON_CONFIDENCE_THRESHOLD = 0.85
INFERENCE_SIZE = 1024
VIETNAMESE_TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}
CROP_PAD_RATIO = 0.06
EXPECTED_TEST_IMAGES = 300
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def normalize_nfc(text):
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\ufeff", "").replace("\u200b", "")
    return " ".join(text.replace("\t", " ").replace("\n", " ").replace("\r", " ").split()).strip()


def strip_vietnamese_marks(text):
    text = unicodedata.normalize("NFD", normalize_nfc(text))
    replacements = {"đ": "d", "Đ": "D"}
    text = "".join(replacements.get(char, char) for char in text)
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def levenshtein_distance(source, target):
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for row, source_char in enumerate(source, start=1):
        current = [row]
        for col, target_char in enumerate(target, start=1):
            current.append(min(
                current[col - 1] + 1,
                previous[col] + 1,
                previous[col - 1] + (source_char != target_char),
            ))
        previous = current
    return previous[-1]


def polygon_iou(first, second):
    try:
        first_points = np.asarray(first, dtype=np.float64).reshape(-1, 2)
        second_points = np.asarray(second, dtype=np.float64).reshape(-1, 2)
        if (
            len(first_points) < 3
            or len(second_points) < 3
            or not np.isfinite(first_points).all()
            or not np.isfinite(second_points).all()
        ):
            return 0.0
        first_polygon = Polygon(first_points).buffer(0)
        second_polygon = Polygon(second_points).buffer(0)
        if first_polygon.is_empty or second_polygon.is_empty:
            return 0.0
        if first_polygon.area <= 0 or second_polygon.area <= 0:
            return 0.0
        union_area = first_polygon.union(second_polygon).area
        return float(first_polygon.intersection(second_polygon).area / union_area) if union_area else 0.0
    except (TypeError, ValueError):
        return 0.0


def crop_and_rectify(image, polygon, pad_ratio=CROP_PAD_RATIO):
    points = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    points = center + (points - center) * (1.0 + 2.0 * pad_ratio)

    top_left, top_right, bottom_right, bottom_left = points
    width = int(round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))))
    height = int(round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))))
    if width < 2 or height < 2:
        return None

    destination = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(points, destination)
    crop = cv2.warpPerspective(
        image, transform, (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop if crop.size else None


def load_vintext_test_items(dataset_root):
    test_dir = dataset_root / "test_image"
    labels_dir = dataset_root / "labels"
    if not test_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("Cần có hai thư mục 'test_image' và 'labels' trong thư mục VinText.")

    label_map = {}
    for label_path in labels_dir.glob("gt_*.txt"):
        match = re.search(r"(\d+)$", label_path.stem)
        if match:
            label_map[match.group(1)] = label_path

    items = []
    for image_path in sorted(test_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = re.search(r"(\d+)$", image_path.stem)
        if not match or match.group(1) not in label_map:
            continue

        annotations = []
        label_path = label_map[match.group(1)]
        for line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            parts = line.strip().split(",", 8)
            if len(parts) < 9:
                continue
            try:
                polygon = np.asarray([float(value.strip()) for value in parts[:8]], dtype=np.float32).reshape(4, 2)
            except ValueError:
                continue
            annotations.append({"polygon": polygon, "text": normalize_nfc(parts[8])})

        items.append({"image_path": image_path, "annotations": annotations})

    return items


def get_prediction_confidence(prediction, keys):
    for key in keys:
        value = prediction.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def prepare_predictions(raw_predictions):
    predictions = []
    for index, raw in enumerate(raw_predictions):
        polygon = raw.get("polygon", raw.get("points"))
        if polygon is None:
            continue
        text = normalize_nfc(raw.get("text", raw.get("transcription", "")))
        detection_confidence = get_prediction_confidence(
            raw, ("detection_confidence", "det_confidence", "det_score", "box_score", "confidence", "score")
        )
        recognition_confidence = get_prediction_confidence(
            raw, ("recognition_confidence", "recognizer_confidence", "parseq_confidence", "rec_confidence",
                  "recognition_score", "parseq_score", "rec_score", "text_confidence")
        )
        if recognition_confidence is None:
            raise ValueError(
                "Prediction thiếu confidence của PARSeq. Hãy trả về một trong các trường "
                "recognition_confidence, rec_confidence, text_confidence, confidence hoặc score."
            )
        if detection_confidence is None:
            detection_confidence = recognition_confidence
        polygon_points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(polygon_points) < 3 or not np.isfinite(polygon_points).all():
            raise ValueError("Đa giác dự đoán phải chứa ít nhất ba điểm tọa độ x/y hữu hạn.")
        predictions.append({
            "polygon": polygon_points,
            "text": text,
            "detection_confidence": detection_confidence,
            "recognition_confidence": recognition_confidence,
            "source_index": index,
        })
    return predictions


def match_predictions(predictions, ground_truth, dont_care):
    ordered = sorted(predictions, key=lambda item: item["detection_confidence"], reverse=True)
    matched_ground_truth = set()
    valid_predictions = []
    matches = []
    ignored_predictions = []
    false_positives = []

    for prediction in ordered:
        candidates = [
            (polygon_iou(prediction["polygon"], gt["polygon"]), gt_index)
            for gt_index, gt in enumerate(ground_truth)
            if gt_index not in matched_ground_truth
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            valid_predictions.append(prediction)
            matched_ground_truth.add(best_index)
            matches.append({"prediction": prediction, "ground_truth": ground_truth[best_index], "gt_index": best_index, "iou": best_iou})
            continue

        ignore_overlap = max(
            (polygon_iou(prediction["polygon"], item["polygon"]) for item in dont_care),
            default=0.0,
        )
        if ignore_overlap >= IOU_THRESHOLD:
            ignored_predictions.append(prediction)
            continue

        valid_predictions.append(prediction)
        false_positives.append(prediction)

    false_negatives = [gt for index, gt in enumerate(ground_truth) if index not in matched_ground_truth]
    return {
        "ordered_predictions": ordered,
        "valid_predictions": valid_predictions,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ignored_predictions": ignored_predictions,
    }


def load_lexicon(path):
    if path is None:
        raise ValueError("Cần cung cấp tệp từ điển tiếng Việt bằng tham số --lexicon.")
    lexicon_path = Path(path)
    if not lexicon_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy từ điển tiếng Việt: {lexicon_path}")
    words = set()
    for line in lexicon_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        word = normalize_nfc(line)
        if word and not any(char.isspace() for char in word):
            words.add(word)
    return sorted(words)


def preserve_case(source, replacement):
    if source.isupper():
        return replacement.upper()
    if source.islower():
        return replacement.lower()
    if source.istitle():
        return replacement[:1].upper() + replacement[1:].lower()

    output = []
    for index, char in enumerate(replacement):
        if index < len(source) and source[index].isupper():
            output.append(char.upper())
        else:
            output.append(char.lower())
    return "".join(output)


def count_matching_base_letters(first, second):
    def base_letters(text):
        letters = []
        for character in unicodedata.normalize("NFD", text.casefold()):
            if unicodedata.combining(character):
                if character not in VIETNAMESE_TONE_MARKS and letters:
                    letters[-1] += character
            else:
                letters.append(character)
        return letters

    first_letters = base_letters(first)
    second_letters = base_letters(second)
    previous = [0] * (len(second_letters) + 1)
    for first_letter in first_letters:
        current = [0]
        for index, second_letter in enumerate(second_letters, start=1):
            if first_letter == second_letter:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def apply_confidence_gated_lexicon(text, confidence, lexicon_index, lexicon_set):
    original = normalize_nfc(text)
    if (
        not confidence < LEXICON_CONFIDENCE_THRESHOLD
        or not original
        or original.casefold() in lexicon_set
        or not original.isalpha()
        or (original.isupper() and len(original) <= 3)
    ):
        return original, False

    lookup_key = strip_vietnamese_marks(original).casefold()
    candidates = set(lexicon_index.get(lookup_key, []))
    for index in range(len(lookup_key)):
        candidates.update(lexicon_index.get(lookup_key[:index] + lookup_key[index + 1:], []))

    original_folded = original.casefold()
    original_nfd = unicodedata.normalize("NFD", original_folded)
    scored_candidates = []
    for candidate in sorted(candidates):
        candidate_folded = candidate.casefold()
        distance = levenshtein_distance(original_folded, candidate_folded)
        if distance == 1:
            matching_base_letters = count_matching_base_letters(original_folded, candidate_folded)
            normalized_distance = levenshtein_distance(
                original_nfd,
                unicodedata.normalize("NFD", candidate_folded),
            )
            scored_candidates.append((-matching_base_letters, normalized_distance, candidate))

    if not scored_candidates:
        return original, False
    _, _, best_word = min(scored_candidates)
    return normalize_nfc(preserve_case(original, best_word)), True


def calculate_detection_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    hmean = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision * 100, "recall": recall * 100, "hmean": hmean * 100, "tp": tp, "fp": fp, "fn": fn}


def calculate_recognition_metrics(records):
    if not records:
        return {"word_acc_cs": 0.0, "word_acc_ci": 0.0, "one_ned": 0.0, "cer": 0.0, "word_count": 0}

    exact_cs = exact_ci = distance_sum = character_count = 0
    normalized_similarity = 0.0
    for predicted, expected in records:
        predicted = normalize_nfc(predicted)
        expected = normalize_nfc(expected)
        exact_cs += predicted == expected
        exact_ci += predicted.casefold() == expected.casefold()
        distance = levenshtein_distance(predicted, expected)
        distance_sum += distance
        character_count += len(expected)
        normalized_similarity += 1.0 - distance / max(len(predicted), len(expected), 1)

    count = len(records)
    return {
        "word_acc_cs": exact_cs / count * 100,
        "word_acc_ci": exact_ci / count * 100,
        "one_ned": normalized_similarity / count * 100,
        "cer": distance_sum / max(character_count, 1) * 100,
        "word_count": count,
    }


def evaluate_end_to_end(image_records, mode):
    totals = {"cs": {"tp": 0, "fp": 0, "fn": 0}, "ci": {"tp": 0, "fp": 0, "fn": 0}}
    confidence = {"tp": [], "fp": []}
    errors = []

    for record in image_records:
        matching = record["matching"]
        for match in matching["matches"]:
            prediction = match["prediction"]
            if mode == "full_lexicon":
                text, changed = apply_confidence_gated_lexicon(
                    prediction["text"], prediction["recognition_confidence"],
                    record["lexicon_index"], record["lexicon_set"],
                )
            else:
                text, changed = prediction["text"], False

            expected = match["ground_truth"]["text"]
            correct_cs = text == expected
            correct_ci = text.casefold() == expected.casefold()
            for key, correct in (("cs", correct_cs), ("ci", correct_ci)):
                if correct:
                    totals[key]["tp"] += 1
                else:
                    totals[key]["fp"] += 1
                    totals[key]["fn"] += 1
            confidence["tp" if correct_cs else "fp"].append(prediction["recognition_confidence"])
            if not correct_cs:
                errors.append({
                    "image": str(record["image_path"].name),
                    "expected": expected,
                    "predicted": text,
                    "confidence": prediction["recognition_confidence"],
                    "iou": match["iou"],
                    "lexicon_changed": changed,
                    "error_type": "Sai nội dung tại vùng khớp",
                })

        for prediction in matching["false_positives"]:
            false_positive_text = prediction["text"]
            false_positive_changed = False
            if mode == "full_lexicon":
                false_positive_text, false_positive_changed = apply_confidence_gated_lexicon(
                    false_positive_text, prediction["recognition_confidence"],
                    record["lexicon_index"], record["lexicon_set"],
                )
            confidence["fp"].append(prediction["recognition_confidence"])
            for key in ("cs", "ci"):
                totals[key]["fp"] += 1
            errors.append({
                "image": str(record["image_path"].name),
                "expected": None,
                "predicted": false_positive_text,
                "confidence": prediction["recognition_confidence"],
                "lexicon_changed": false_positive_changed,
                "iou": 0.0,
                "error_type": "Phát hiện nhầm hoặc box trùng Ground Truth đã khớp",
            })

        for missed_ground_truth in matching["false_negatives"]:
            for key in ("cs", "ci"):
                totals[key]["fn"] += 1
            errors.append({
                "image": str(record["image_path"].name),
                "expected": missed_ground_truth["text"],
                "predicted": None,
                "confidence": None,
                "iou": None,
                "lexicon_changed": False,
                "error_type": "Bỏ sót vùng chữ",
            })

    metrics = {
        key: calculate_detection_metrics(value["tp"], value["fp"], value["fn"])
        for key, value in totals.items()
    }
    return {"metrics": metrics, "confidence": confidence, "errors": errors}


def get_vietnamese_font():
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/times.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    ]
    for font_path in candidates:
        if font_path.is_file():
            return FontProperties(fname=str(font_path))
    return FontProperties(family="DejaVu Sans")


def plot_metrics(detection_metrics, e2e_none, e2e_lexicon, output_path):
    metric_names = ["Độ chính xác", "Độ bao phủ", "H-Mean"]
    series = {
        "CS": [detection_metrics, e2e_none["metrics"]["cs"], e2e_lexicon["metrics"]["cs"]],
        "CI": [detection_metrics, e2e_none["metrics"]["ci"], e2e_lexicon["metrics"]["ci"]],
    }
    values = {
        name: [[entry["precision"], entry["recall"], entry["hmean"]] for entry in stage_metrics]
        for name, stage_metrics in series.items()
    }

    font = get_vietnamese_font()
    figure, axes = plt.subplots(1, 2, figsize=(17, 6))
    x = np.arange(len(metric_names))
    series_values = [
        ("Phát hiện", [values["CS"][0]]),
        ("Toàn pipeline - Không từ điển (CS)", [values["CS"][1]]),
        ("Toàn pipeline - Không từ điển (CI)", [values["CI"][1]]),
        ("Toàn pipeline - Từ điển (CS)", [values["CS"][2]]),
        ("Toàn pipeline - Từ điển (CI)", [values["CI"][2]]),
    ]
    width = 0.14
    offsets = np.linspace(-(len(series_values) - 1) / 2, (len(series_values) - 1) / 2, len(series_values)) * width
    for series_index, (label, entries) in enumerate(series_values):
        metric_values = entries[0]
        bar = axes[0].bar(x + offsets[series_index], metric_values, width=width, label=label)
        axes[0].bar_label(bar, fmt="%.1f%%", padding=2, fontsize=7, rotation=90)
    axes[0].set_xticks(x, metric_names, fontproperties=font)
    axes[0].set_ylabel("Tỷ lệ (%)", fontproperties=font)
    axes[0].set_title("So sánh độ chính xác, độ bao phủ và H-Mean", fontproperties=font)
    axes[0].set_ylim(0, 128)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(title="Chế độ đánh giá", prop=font)

    bins = np.linspace(0, 1, 21)
    for result, label, linestyle in (
        (e2e_none, "Không từ điển", "-"),
        (e2e_lexicon, "Có từ điển", "--"),
    ):
        axes[1].hist(result["confidence"]["tp"], bins=bins, density=True, histtype="step", linewidth=2, label=f"Đúng - {label}", linestyle=linestyle)
        axes[1].hist(result["confidence"]["fp"], bins=bins, density=True, histtype="step", linewidth=2, label=f"Sai - {label}", linestyle=linestyle)
    axes[1].axvline(LEXICON_CONFIDENCE_THRESHOLD, linestyle=":", linewidth=1.5, label="Ngưỡng 0,85")
    axes[1].set_xlabel("Độ tự tin nhận diện", fontproperties=font)
    axes[1].set_ylabel("Mật độ phân phối", fontproperties=font)
    axes[1].set_title("Phân phối độ tự tin của dự đoán đúng và dự đoán sai", fontproperties=font)
    axes[1].grid(alpha=0.25)
    axes[1].legend(prop=font)

    figure.suptitle("Tổng hợp hiệu năng định lượng trên tập kiểm thử VinText chính thức", fontproperties=font, fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def draw_polygon(image, polygon, color, text=None, dashed=False, font=None):
    points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    if dashed:
        for start, end in zip(points.reshape(-1, 2), np.roll(points.reshape(-1, 2), -1, axis=0)):
            length = np.linalg.norm(end - start)
            steps = max(int(length / 12), 1)
            for index in range(0, steps, 2):
                first = start + (end - start) * (index / steps)
                second = start + (end - start) * (min(index + 1, steps) / steps)
                cv2.line(image, tuple(first.astype(int)), tuple(second.astype(int)), color, 2, cv2.LINE_AA)
    else:
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
    if text:
        font_path = next((path for path in (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/times.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        ) if path.is_file()), None)
        if font_path:
            pil_image = Image.fromarray(image)
            drawer = ImageDraw.Draw(pil_image)
            label_font = ImageFont.truetype(str(font_path), 16)
            x, y = points.reshape(-1, 2)[0]
            drawer.text((int(x), max(0, int(y) - 20)), normalize_nfc(text)[:45], font=label_font, fill=tuple(color), stroke_width=1, stroke_fill=(0, 0, 0))
            image[:] = np.asarray(pil_image)


def plot_qualitative_grid(image_records, output_path):
    if not image_records:
        return

    def displayed_text(record, prediction):
        if record.get("lexicon_index") is None or record.get("lexicon_set") is None:
            return normalize_nfc(prediction["text"])
        text, _ = apply_confidence_gated_lexicon(
            prediction["text"],
            prediction["recognition_confidence"],
            record["lexicon_index"],
            record["lexicon_set"],
        )
        return text

    ranked = sorted(
        image_records,
        key=lambda record: (
            len(record["matching"]["false_positives"]) +
            len(record["matching"]["false_negatives"]) +
            sum(
                displayed_text(record, match["prediction"]).casefold()
                != normalize_nfc(match["ground_truth"]["text"]).casefold()
                for match in record["matching"]["matches"]
            )
        ),
        reverse=True,
    )
    error_samples = ranked[:3]
    easy_samples = sorted(ranked[3:], key=lambda record: (
        len(record["matching"]["false_positives"]) + len(record["matching"]["false_negatives"])
    ))[:3]
    selected = (error_samples + easy_samples)[:6]
    font = get_vietnamese_font()
    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    for axis in axes.flat:
        axis.axis("off")

    for axis, record in zip(axes.flat, selected):
        image = cv2.imread(str(record["image_path"]))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        matching = record["matching"]
        for match in matching["matches"]:
            prediction = match["prediction"]
            expected = match["ground_truth"]["text"]
            predicted_text = displayed_text(record, prediction)
            correct = predicted_text.casefold() == normalize_nfc(expected).casefold()
            color = (30, 180, 60) if correct else (220, 40, 40)
            draw_polygon(image, prediction["polygon"], color, predicted_text)
            if not correct:
                draw_polygon(image, match["ground_truth"]["polygon"], (245, 190, 20), expected, dashed=True)
        for prediction in matching["false_positives"]:
            draw_polygon(image, prediction["polygon"], (220, 40, 40), displayed_text(record, prediction))
        for ground_truth in matching["false_negatives"]:
            draw_polygon(image, ground_truth["polygon"], (245, 190, 20), ground_truth["text"], dashed=True)
        axis.imshow(image)
        axis.set_title(record["image_path"].name, fontproperties=font)

    figure.suptitle("Trực quan hóa Full Lexicon (Case-Insensitive) trên VinText", fontproperties=font, fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def print_metrics_table(detection, recognition, e2e_none, e2e_lexicon, latency_ms, fps):
    print("\nKẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ CHÍNH THỨC VINTEXT")
    print("\n1. DETECTION-ONLY")
    print("Đánh giá polygon phát hiện so với Ground Truth, không xét nội dung nhận dạng.")
    print(f"{'Precision':>13}{'Recall':>13}{'H-Mean':>13}")
    print(f"{detection['precision']:>12.2f}%{detection['recall']:>12.2f}%{detection['hmean']:>12.2f}%")

    print("\n2. RECOGNITION-ONLY TRÊN GROUND TRUTH")
    print("Đưa vùng chữ Ground Truth vào bộ nhận dạng, không phụ thuộc kết quả Detection.")
    print(f"Số từ đánh giá: {recognition['word_count']}")
    print(f"{'Word Acc CS':>13}{'Word Acc CI':>13}{'1-NED':>13}{'CER':>13}")
    print(
        f"{recognition['word_acc_cs']:>12.2f}%"
        f"{recognition['word_acc_ci']:>13.2f}%"
        f"{recognition['one_ned']:>13.2f}%"
        f"{recognition['cer']:>13.2f}%"
    )

    print("\n3. END-TO-END DETECTION + RECOGNITION")
    print(f"{'Chế độ':<34}{'Precision':>13}{'Recall':>13}{'H-Mean':>13}")
    for label, result in (
        ("Không từ điển (CS)", e2e_none["metrics"]["cs"]),
        ("Không từ điển (CI)", e2e_none["metrics"]["ci"]),
        ("Full Lexicon (CS)", e2e_lexicon["metrics"]["cs"]),
        ("Full Lexicon (CI)", e2e_lexicon["metrics"]["ci"]),
    ):
        print(
            f"{label:<34}{result['precision']:>12.2f}%"
            f"{result['recall']:>12.2f}%{result['hmean']:>12.2f}%"
        )

    print(f"Tốc độ trung bình:   {latency_ms:.2f} ms/ảnh ({fps:.2f} FPS)")


def build_dashboard_summary(report):
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": report["protocol"]["dataset"],
        "image_count": report["protocol"]["image_count"],
        "expected_image_count": report["protocol"]["expected_image_count"],
        "detection": report["detection"],
        "recognition_on_ground_truth": report["recognition_on_ground_truth"],
        "end_to_end": report["end_to_end"],
        "speed": report["speed"],
    }


def write_dashboard_summary(report):
    summary_path = Path(__file__).resolve().parent / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(build_dashboard_summary(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def run_benchmark(dataset_path, lexicon_path, output_dir):
    dataset_root = Path(dataset_path)
    if (dataset_root / "vietnamese").is_dir():
        dataset_root = dataset_root / "vietnamese"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_items = load_vintext_test_items(dataset_root)
    if len(test_items) != EXPECTED_TEST_IMAGES:
        raise ValueError(
            f"Tập VinText Official Test phải có đúng {EXPECTED_TEST_IMAGES} ảnh ghép cặp nhãn; "
            f"hiện tìm thấy {len(test_items)} ảnh. Không tự động cắt hoặc bổ sung mẫu."
        )
    if any(not item["annotations"] for item in test_items):
        raise ValueError("Có ảnh test không có annotation hợp lệ; cần kiểm tra lại bộ nhãn.")

    lexicon = load_lexicon(lexicon_path)
    lexicon_set = {word.casefold() for word in lexicon}
    lexicon_index = defaultdict(list)
    for word in lexicon:
        key = strip_vietnamese_marks(word).casefold()
        lexicon_index[key].append(word)
        for index in range(len(key)):
            lexicon_index[key[:index] + key[index + 1:]].append(word)

    pipeline = SceneTextPipeline()
    image_records = []
    recognition_records = []
    inference_times = []
    total_ground_truth = total_dont_care = 0
    total_detection_predictions = total_ignored_predictions = 0

    print("Bắt đầu đánh giá độc lập trên 300 ảnh VinText Official Test.")
    for image_index, item in enumerate(test_items, start=1):
        bgr = cv2.imread(str(item["image_path"]))
        if bgr is None:
            raise RuntimeError(f"Không đọc được ảnh: {item['image_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ground_truth = [ann for ann in item["annotations"] if ann["text"] != "###"]
        dont_care = [ann for ann in item["annotations"] if ann["text"] == "###"]
        total_ground_truth += len(ground_truth)
        total_dont_care += len(dont_care)

        start_time = time.perf_counter()
        result = pipeline.run(
            rgb,
            thresh=0.3,
            box_thresh=0.5,
            unclip_ratio=1.7,
            target_size=INFERENCE_SIZE,
        )
        inference_times.append(time.perf_counter() - start_time)
        predictions = prepare_predictions(result.get("items", []))
        matching = match_predictions(predictions, ground_truth, dont_care)
        total_detection_predictions += len(matching["valid_predictions"])
        total_ignored_predictions += len(matching["ignored_predictions"])

        for gt in ground_truth:
            crop = crop_and_rectify(rgb, gt["polygon"], pad_ratio=CROP_PAD_RATIO)
            if crop is None or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            recognition_result = pipeline.recognizer.predict_batch([crop])
            if recognition_result:
                predicted_text = normalize_nfc(recognition_result[0][0])
            else:
                predicted_text = ""
            recognition_records.append((predicted_text, gt["text"]))

        image_records.append({
            "image_path": item["image_path"],
            "matching": matching,
            "lexicon_index": lexicon_index,
            "lexicon_set": lexicon_set,
        })
        if image_index % 25 == 0 or image_index == EXPECTED_TEST_IMAGES:
            print(f"Đã xử lý {image_index}/{EXPECTED_TEST_IMAGES} ảnh.")

    detection_tp = sum(len(record["matching"]["matches"]) for record in image_records)
    detection_fp = sum(len(record["matching"]["false_positives"]) for record in image_records)
    detection_fn = sum(len(record["matching"]["false_negatives"]) for record in image_records)
    detection_metrics = calculate_detection_metrics(detection_tp, detection_fp, detection_fn)
    recognition_metrics = calculate_recognition_metrics(recognition_records)
    e2e_none = evaluate_end_to_end(image_records, "none")
    e2e_lexicon = evaluate_end_to_end(image_records, "full_lexicon")

    average_seconds = float(np.mean(inference_times)) if inference_times else 0.0
    latency_ms = average_seconds * 1000
    fps = 1.0 / average_seconds if average_seconds > 0 else 0.0

    print_metrics_table(detection_metrics, recognition_metrics, e2e_none, e2e_lexicon, latency_ms, fps)

    metrics_path = output_dir / "benchmark_metrics.png"
    qualitative_path = output_dir / "e2e_qualitative_grid.png"
    plot_metrics(detection_metrics, e2e_none, e2e_lexicon, metrics_path)
    plot_qualitative_grid(image_records, qualitative_path)

    report = {
        "protocol": {
            "dataset": "VinText Official Test",
            "image_count": len(image_records),
            "expected_image_count": EXPECTED_TEST_IMAGES,
            "detector_input_size": [INFERENCE_SIZE, INFERENCE_SIZE],
            "iou_threshold": IOU_THRESHOLD,
            "dont_care_label": "###",
            "matching": "One-to-One, prediction confidence descending",
            "unicode_normalization": "NFC",
            "crop_pad_ratio": CROP_PAD_RATIO,
            "lexicon_confidence_threshold": LEXICON_CONFIDENCE_THRESHOLD,
            "lexicon_path": str(lexicon_path) if lexicon_path else None,
            "lexicon_size": len(lexicon),
        },
        "counts": {
            "ground_truth_non_ignore": total_ground_truth,
            "ground_truth_dont_care": total_dont_care,
            "detection_predictions_after_ignore": total_detection_predictions,
            "predictions_ignored_by_dont_care": total_ignored_predictions,
            "detection_tp": detection_tp,
            "detection_fp": detection_fp,
            "detection_fn": detection_fn,
        },
        "detection": detection_metrics,
        "recognition_on_ground_truth": recognition_metrics,
        "end_to_end": {
            "none": e2e_none["metrics"],
            "full_lexicon": e2e_lexicon["metrics"],
        },
        "speed": {
            "mean_latency_ms_per_image": latency_ms,
            "fps": fps,
            "timed_images": len(inference_times),
            "scope": "pipeline.run; không bao gồm crop recognition độc lập và vẽ biểu đồ",
        },
        "error_analysis": {
            "none": e2e_none["errors"],
            "full_lexicon": e2e_lexicon["errors"],
            "confidence_distribution": {
                "none": e2e_none["confidence"],
                "full_lexicon": e2e_lexicon["confidence"],
            },
        },
        "artifacts": {
            "metrics_chart": str(metrics_path),
            "qualitative_grid": str(qualitative_path),
        },
    }
    report_path = output_dir / "benchmark_results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = write_dashboard_summary(report)
    print(f"Đã lưu báo cáo: {report_path}")
    print(f"Đã cập nhật dữ liệu benchmark cho web: {summary_path}")
    print(f"Đã lưu biểu đồ hiệu năng: {metrics_path}")
    print(f"Đã lưu lưới trực quan: {qualitative_path}")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    script_dir = Path(__file__).resolve().parent
    pipeline_dir = script_dir.parent
    project_dir = pipeline_dir.parent
    parser = argparse.ArgumentParser(description="Đánh giá DBNet++ và PARSeq trên tập kiểm thử VinText chính thức")
    parser.add_argument(
        "--dataset",
        default=str(project_dir / "dataset" / "vintext_official" / "vietnamese"),
        help="Thư mục VinText chứa ảnh kiểm thử và nhãn",
    )
    parser.add_argument(
        "--lexicon",
        default=str(pipeline_dir / "vn_dictionary.txt"),
        help="Tệp từ điển tiếng Việt UTF-8, mỗi từ một dòng",
    )
    parser.add_argument("--output", default=".", help="Thư mục lưu JSON và biểu đồ")
    args = parser.parse_args()
    run_benchmark(args.dataset, args.lexicon, args.output)


if __name__ == "__main__":
    main()
