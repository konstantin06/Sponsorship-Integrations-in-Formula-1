from pathlib import Path
import pandas as pd
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "outputs_train_yolo" / "yolo11s_f1_baseline_img640" / "weights" / "best.pt"
DATA_YAML = PROJECT_DIR / "final_dataset_yolov11" / "data.yaml"

OUTPUT_DIR = PROJECT_DIR / "metrics_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        save_json=True,
        project=str(OUTPUT_DIR),
        name="yolo_test_metrics",
        exist_ok=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    summary = pd.DataFrame([{
        "model": "YOLO11s",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP@50": map50,
        "mAP@50:95": map5095,
    }])

    summary_path = OUTPUT_DIR / "yolo_summary_metrics.csv"
    summary.to_csv(summary_path, index=False)

    print("\nYOLO summary metrics:")
    print(summary)
    print("\nSaved to:", summary_path)

    class_names = metrics.names
    ap50_per_class = metrics.box.ap50
    ap5095_per_class = metrics.box.ap

    rows = []
    for class_id, class_name in class_names.items():
        rows.append({
            "model": "YOLO11s",
            "class_id": class_id,
            "class_name": class_name,
            "AP@50": float(ap50_per_class[class_id]),
            "AP@50:95": float(ap5095_per_class[class_id]),
        })

    per_class = pd.DataFrame(rows)
    per_class_path = OUTPUT_DIR / "yolo_per_class_metrics.csv"
    per_class.to_csv(per_class_path, index=False)

    print("\nYOLO per-class metrics:")
    print(per_class)
    print("\nSaved to:", per_class_path)


if __name__ == "__main__":
    main()