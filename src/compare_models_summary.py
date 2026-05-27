from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_DIR / "metrics_comparison"

RFDETR_PATH = METRICS_DIR / "rfdetr_summary_metrics.csv"
YOLO_PATH = METRICS_DIR / "yolo_summary_metrics.csv"

OUTPUT_PATH = METRICS_DIR / "models_comparison_summary.csv"


def main():
    rfdetr = pd.read_csv(RFDETR_PATH)
    yolo = pd.read_csv(YOLO_PATH)

    rfdetr = rfdetr.rename(columns={
        "precision@0.5conf_0.5iou": "precision",
        "recall@0.5conf_0.5iou": "recall",
        "f1@0.5conf_0.5iou": "f1",
    })

    comparison = pd.concat([rfdetr, yolo], ignore_index=True)

    columns = [
        "model",
        "mAP@50",
        "mAP@50:95",
        "precision",
        "recall",
        "f1",
    ]

    comparison = comparison[columns]

    comparison.to_csv(OUTPUT_PATH, index=False)

    print(comparison)
    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()