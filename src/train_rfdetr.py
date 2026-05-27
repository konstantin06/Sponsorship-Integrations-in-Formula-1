from pathlib import Path
import json
import torch
import multiprocessing as mp
from rfdetr import RFDETRSmall



DATASET_DIR = Path("final_dataset")
OUTPUT_DIR = Path("outputs_train_rfdetr/final_rfdetr_f1_local")

EPOCHS = 50
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16

LR = 1e-4


def main():

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
    else:
        raise RuntimeError(
            "CUDA is not available. Проверь установку PyTorch CUDA и драйвер NVIDIA."
        )

    coco_path = DATASET_DIR / "train" / "_annotations.coco.json"

    if not coco_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {coco_path}")

    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco["categories"], key=lambda x: x["id"])
    class_names = [cat["name"] for cat in categories]
    num_classes = len(class_names)

    print("\nClasses:")
    for cat in categories:
        print(cat["id"], cat["name"])

    print("Number of classes:", num_classes)

    if num_classes != 10:
        print("\nWARNING: expected 10 classes, but found:", num_classes)
        print("Проверь датасет перед обучением.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = RFDETRSmall(num_classes=num_classes)

    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LR,
        output_dir=str(OUTPUT_DIR),
        tensorboard=True,
        wandb=False,
        early_stopping=True,
        early_stopping_patience=10,
        checkpoint_interval=10,
    )

    print("\nTraining finished.")
    print("Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    mp.freeze_support()
    main()