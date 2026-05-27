from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from rfdetr import RFDETRSmall



PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "outputs_train_rfdetr"
    / "final_rfdetr_f1_local"
    / "checkpoint_best_total.pth"
)

VIDEO_PATH = (
    PROJECT_DIR
    / "hl_monaco_test.mp4"
)

OUTPUT_DIR = PROJECT_DIR / "outputs_video_rfdetr" / "race_fragment_rfdetr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO_PATH = OUTPUT_DIR / "rfdetr_annotated_video12.mp4"
OUTPUT_DETECTIONS_CSV = OUTPUT_DIR / "rfdetr_detections_monaco12.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "rfdetr_brand_summary_monaco12.csv"

CLASS_NAMES = [
    "aramco",
    "atlasian",
    "bwt",
    "capp",
    "hp",
    "kick",
    "moneygram",
    "okx",
    "oracle",
    "petronas",
]

CONFIDENCE_THRESHOLD = 0.4


PROCESS_EVERY_N_FRAMES = 5

SAVE_ANNOTATED_VIDEO = False


def draw_detections(frame, detections):
    annotated = frame.copy()

    for box, class_id, confidence in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence
    ):
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)
        class_id = int(class_id)

        if 0 <= class_id < len(CLASS_NAMES):
            brand = CLASS_NAMES[class_id]
        else:
            brand = f"class_{class_id}"

        label = f"{brand} {confidence:.2f}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return annotated


def calculate_brand_summary(df, fps):
    if df.empty:
        return pd.DataFrame(columns=[
            "brand",
            "detections_count",
            "unique_frames",
            "approx_duration_sec",
            "avg_confidence",
            "avg_share_of_screen",
            "max_share_of_screen",
            "sum_share_of_screen",
            "visibility_index",
            "weighted_visibility_index",
        ])

    processed_fps = fps / PROCESS_EVERY_N_FRAMES

    summary = (
        df.groupby("brand")
        .agg(
            detections_count=("brand", "count"),
            unique_frames=("frame_id", "nunique"),
            avg_confidence=("confidence", "mean"),
            avg_share_of_screen=("share_of_screen", "mean"),
            max_share_of_screen=("share_of_screen", "max"),
            sum_share_of_screen=("share_of_screen", "sum"),
        )
        .reset_index()
    )

    summary["approx_duration_sec"] = summary["unique_frames"] / processed_fps

    summary["visibility_index"] = (
        summary["approx_duration_sec"] * summary["avg_share_of_screen"]
    )

    summary["weighted_visibility_index"] = (
        summary["approx_duration_sec"]
        * summary["avg_share_of_screen"]
        * summary["avg_confidence"]
    )

    summary = summary.sort_values(
        by="weighted_visibility_index",
        ascending=False
    )

    return summary


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    print("Loading RF-DETR model...")
    print("Model:", MODEL_PATH)

    model = RFDETRSmall(
        pretrain_weights=str(MODEL_PATH),
        num_classes=len(CLASS_NAMES)
    )

    model.optimize_for_inference()
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print("\nVideo info:")
    print("Path:", VIDEO_PATH)
    print("FPS:", fps)
    print("Resolution:", width, "x", height)
    print("Total frames:", total_frames)
    print("Duration sec:", duration_sec)
    print("Process every N frames:", PROCESS_EVERY_N_FRAMES)
    print("Confidence threshold:", CONFIDENCE_THRESHOLD)

    if SAVE_ANNOTATED_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(OUTPUT_VIDEO_PATH),
            fourcc,
            fps,
            (width, height)
        )
    else:
        out = None

    records = []
    frame_id = 0

    pbar = tqdm(total=total_frames, desc="Processing video")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % PROCESS_EVERY_N_FRAMES == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            detections = model.predict(
                pil_image,
                threshold=CONFIDENCE_THRESHOLD
            )

            if len(detections) > 0:
                for box, class_id, confidence in zip(
                    detections.xyxy,
                    detections.class_id,
                    detections.confidence
                ):
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = box
                    class_id = int(class_id)

                    if 0 <= class_id < len(CLASS_NAMES):
                        brand = CLASS_NAMES[class_id]
                    else:
                        brand = f"class_{class_id}"

                    bbox_width = max(0, x2 - x1)
                    bbox_height = max(0, y2 - y1)
                    bbox_area = bbox_width * bbox_height

                    frame_area = width * height
                    share_of_screen = bbox_area / frame_area if frame_area > 0 else 0

                    records.append({
                        "frame_id": frame_id,
                        "time_sec": frame_id / fps if fps > 0 else None,
                        "brand": brand,
                        "class_id": class_id,
                        "confidence": float(confidence),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "bbox_width": float(bbox_width),
                        "bbox_height": float(bbox_height),
                        "bbox_area": float(bbox_area),
                        "share_of_screen": float(share_of_screen),
                    })

            if SAVE_ANNOTATED_VIDEO:
                annotated_frame = draw_detections(frame, detections)
                out.write(annotated_frame)

        else:
            if SAVE_ANNOTATED_VIDEO:
                out.write(frame)

        frame_id += 1
        pbar.update(1)

    pbar.close()

    cap.release()

    if out is not None:
        out.release()

    detections_df = pd.DataFrame(records)
    detections_df.to_csv(OUTPUT_DETECTIONS_CSV, index=False)

    brand_summary = calculate_brand_summary(detections_df, fps)
    brand_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print("\nDone.")
    print("Detections saved to:", OUTPUT_DETECTIONS_CSV)
    print("Summary saved to:", OUTPUT_SUMMARY_CSV)

    if SAVE_ANNOTATED_VIDEO:
        print("Annotated video saved to:", OUTPUT_VIDEO_PATH)

    print("\nBrand summary:")
    print(brand_summary)


if __name__ == "__main__":
    main()