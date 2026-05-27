from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image

from rfdetr import RFDETRSmall

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "final_dataset"
TEST_DIR = DATASET_DIR / "test"
TEST_COCO_PATH = TEST_DIR / "_annotations.coco.json"

MODEL_PATH = (
    PROJECT_DIR
    / "outputs_train_rfdetr"
    / "final_rfdetr_f1_local"
    / "checkpoint_best_total.pth"
)

OUTPUT_DIR = PROJECT_DIR / "metrics_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_JSON = OUTPUT_DIR / "rfdetr_test_predictions_coco.json"
SUMMARY_CSV = OUTPUT_DIR / "rfdetr_summary_metrics.csv"
PER_CLASS_CSV = OUTPUT_DIR / "rfdetr_per_class_metrics.csv"

PREDICT_THRESHOLD_FOR_MAP = 0.01

CONF_THRESHOLD_FOR_F1 = 0.5
IOU_THRESHOLD_FOR_F1 = 0.5


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [
        float(x1),
        float(y1),
        float(x2 - x1),
        float(y2 - y1),
    ]


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


def load_categories():
    with open(TEST_COCO_PATH, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    categories = sorted(coco_data["categories"], key=lambda x: x["id"])
    category_ids = [cat["id"] for cat in categories]
    category_names = [cat["name"] for cat in categories]

    return categories, category_ids, category_names, coco_data


def get_image_path(file_name):
    candidate = TEST_DIR / file_name

    if candidate.exists():
        return candidate

    candidate = TEST_DIR / Path(file_name).name

    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Image not found for file_name={file_name}")


def run_rfdetr_predictions():
    categories, category_ids, category_names, coco_data = load_categories()
    num_classes = len(category_names)

    print("Classes:")
    for cat in categories:
        print(cat["id"], cat["name"])

    print("\nLoading model:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    model = RFDETRSmall(
        pretrain_weights=str(MODEL_PATH),
        num_classes=num_classes
    )

    coco_predictions = []

    images = coco_data["images"]

    print("\nRunning predictions on test images...")

    for image_info in tqdm(images):
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        image_path = get_image_path(file_name)
        image = Image.open(image_path).convert("RGB")

        detections = model.predict(
            image,
            threshold=PREDICT_THRESHOLD_FOR_MAP
        )

        if len(detections) == 0:
            continue

        for box, class_id, confidence in zip(
            detections.xyxy,
            detections.class_id,
            detections.confidence
        ):
            class_id = int(class_id)

            if 0 <= class_id < len(category_ids):
                category_id = category_ids[class_id]
            else:
                category_id = class_id

            coco_predictions.append({
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": xyxy_to_xywh(box),
                "score": float(confidence),
            })

    with open(PREDICTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(coco_predictions, f, ensure_ascii=False, indent=2)

    print("\nPredictions saved to:")
    print(PREDICTIONS_JSON)

    print("Number of predictions:", len(coco_predictions))

    return coco_predictions

def evaluate_with_coco():
    print("\nRunning COCO evaluation...")

    coco_gt = COCO(str(TEST_COCO_PATH))

    with open(PREDICTIONS_JSON, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if len(predictions) == 0:
        raise RuntimeError("No predictions found. Cannot run COCOeval.")

    coco_dt = coco_gt.loadRes(str(PREDICTIONS_JSON))

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_5095 = float(coco_eval.stats[0])
    map_50 = float(coco_eval.stats[1])
    map_75 = float(coco_eval.stats[2])

    return map_50, map_5095, map_75


def evaluate_precision_recall_f1():
    with open(TEST_COCO_PATH, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    with open(PREDICTIONS_JSON, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    categories, category_ids, category_names, _ = load_categories()
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    gt_by_image_cat = {}
    total_gt_by_class = {cat_id: 0 for cat_id in category_ids}

    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        cat_id = ann["category_id"]
        x, y, w, h = ann["bbox"]
        box = [x, y, x + w, y + h]

        gt_by_image_cat.setdefault((image_id, cat_id), []).append({
            "box": box,
            "matched": False,
        })

        total_gt_by_class[cat_id] += 1

    preds_by_class = {cat_id: [] for cat_id in category_ids}

    for pred in predictions:
        if pred["score"] < CONF_THRESHOLD_FOR_F1:
            continue

        cat_id = pred["category_id"]

        if cat_id not in preds_by_class:
            continue

        x, y, w, h = pred["bbox"]
        box = [x, y, x + w, y + h]

        preds_by_class[cat_id].append({
            "image_id": pred["image_id"],
            "box": box,
            "score": pred["score"],
        })

    rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cat_id in category_ids:
        preds = sorted(
            preds_by_class[cat_id],
            key=lambda x: x["score"],
            reverse=True
        )

        tp = 0
        fp = 0

        for pred in preds:
            image_id = pred["image_id"]
            gt_list = gt_by_image_cat.get((image_id, cat_id), [])

            best_iou = 0.0
            best_gt = None

            for gt in gt_list:
                if gt["matched"]:
                    continue

                iou = compute_iou(pred["box"], gt["box"])

                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_iou >= IOU_THRESHOLD_FOR_F1 and best_gt is not None:
                tp += 1
                best_gt["matched"] = True
            else:
                fp += 1

        fn = total_gt_by_class[cat_id] - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        rows.append({
            "model": "RF-DETR Small",
            "class_id": cat_id,
            "class_name": cat_id_to_name[cat_id],
            "gt_objects": total_gt_by_class[cat_id],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision@0.5conf_0.5iou": precision,
            "recall@0.5conf_0.5iou": recall,
            "f1@0.5conf_0.5iou": f1,
        })

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )

    per_class_df = pd.DataFrame(rows)
    per_class_df.to_csv(PER_CLASS_CSV, index=False)

    return overall_precision, overall_recall, overall_f1, per_class_df

def main():
    run_rfdetr_predictions()

    map50, map5095, map75 = evaluate_with_coco()

    precision, recall, f1, per_class_df = evaluate_precision_recall_f1()

    summary = pd.DataFrame([{
        "model": "RF-DETR Small",
        "mAP@50": map50,
        "mAP@50:95": map5095,
        "mAP@75": map75,
        "precision@0.5conf_0.5iou": precision,
        "recall@0.5conf_0.5iou": recall,
        "f1@0.5conf_0.5iou": f1,
    }])

    summary.to_csv(SUMMARY_CSV, index=False)

    print("\nRF-DETR summary metrics:")
    print(summary)

    print("\nRF-DETR per-class metrics:")
    print(per_class_df)

    print("\nSaved:")
    print(SUMMARY_CSV)
    print(PER_CLASS_CSV)


if __name__ == "__main__":
    main()