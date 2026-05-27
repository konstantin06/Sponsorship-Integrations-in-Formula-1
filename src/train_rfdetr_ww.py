from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path(
    r"D:\HW_3course\CourseWork_3course_f1sponsors\f1_project\metrics_comparison\models_comparison_summary.csv"
)

OUTPUT_DIR = CSV_PATH.parent / "comparison_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "rfdetr_vs_yolo_metrics_comparison.png"


df = pd.read_csv(CSV_PATH)
print("Loaded data:")
print(df)


metrics = ["mAP@50", "mAP@50:95", "precision", "recall", "f1"]

missing_columns = [col for col in ["model"] + metrics if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in CSV file: {missing_columns}")

models = df["model"].tolist()

x = np.arange(len(metrics))
bar_width = 0.35


plt.figure(figsize=(11, 6))

for i, model in enumerate(models):
    values = df.loc[df["model"] == model, metrics].iloc[0].values.astype(float)

    plt.bar(
        x + (i - (len(models) - 1) / 2) * bar_width,
        values,
        width=bar_width,
        label=model
    )

    for j, value in enumerate(values):
        plt.text(
            x[j] + (i - (len(models) - 1) / 2) * bar_width,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(x, ["mAP@50", "mAP@50:95", "Precision", "Recall", "F1-score"])
plt.ylabel("Metric value")
plt.ylim(0, 1.0)
plt.title("Comparison of RF-DETR Small and YOLO11s")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()

print(f"Plot saved to: {OUTPUT_PATH}")