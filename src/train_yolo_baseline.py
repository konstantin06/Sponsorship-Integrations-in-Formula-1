from pathlib import Path
import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent

DATA_YAML = PROJECT_DIR / "final_dataset_yolov11" / "data.yaml"
OUTPUT_ROOT = PROJECT_DIR / "outputs_train_yolo"

MODEL_NAME = "yolo11s.pt"

EPOCHS = 50
IMGSZ = 640
BATCH = 8

RUN_NAME = "yolo11s_f1_baseline_img640"


def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        raise RuntimeError("CUDA is not available. Проверь PyTorch CUDA и драйвер NVIDIA.")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,

        project=str(OUTPUT_ROOT),
        name=RUN_NAME,

        pretrained=True,

        plots=True,
        val=True,

        patience=15,

        seed=42,

        workers=0,

        exist_ok=False,
    )

    print("\nTraining finished.")
    print("Results saved to:", OUTPUT_ROOT / RUN_NAME)
    print(results)


if __name__ == "__main__":
    main()