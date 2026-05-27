from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import cv2
import gc


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return fps, total_frames, width, height


def main():

    MODEL_PATH = r"D:\HW_3course\CourseWork_3course_f1sponsors\f1_project\outputs_train_yolo\yolo11s_f1_baseline_img640\weights\best.pt"

    VIDEO_PATH = r"D:\HW_3course\CourseWork_3course_f1sponsors\f1_project\hl_monaco_test.mp4"

    OUTPUT_PROJECT = r"D:\HW_3course\CourseWork_3course_f1sponsors\f1_project\video_outputs"
    OUTPUT_NAME = "yolo_monaco_stream_result1"


    IMG_SIZE = 640
    CONF_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.5
    VID_STRIDE = 5

    model_path = Path(MODEL_PATH)
    video_path = Path(VIDEO_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    output_dir = Path(OUTPUT_PROJECT) / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_csv_path = output_dir / "yolo_detections1.csv"
    summary_csv_path = output_dir / "yolo_detection_summary_by_class1.csv"
    frame_summary_csv_path = output_dir / "yolo_detection_summary_by_frame1.csv"

    fps, total_frames, video_width, video_height = get_video_info(video_path)
    frame_area = video_width * video_height

    print("Video information:")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Resolution: {video_width}x{video_height}")
    print(f"Frame area: {frame_area}")


    print("\nLoading YOLO model...")
    print(f"Model: {model_path}")

    model = YOLO(str(model_path))

    print("\nStarting video inference...")
    print(f"Video: {video_path}")
    print(f"conf={CONF_THRESHOLD}, imgsz={IMG_SIZE}, vid_stride={VID_STRIDE}")

    results = model.predict(
        source=str(video_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=True,
        stream=True,
        vid_stride=VID_STRIDE,
        project=OUTPUT_PROJECT,
        name=OUTPUT_NAME,
        exist_ok=True,
        verbose=True
    )


    detections = []

    processed_frames = 0
    total_detections = 0

    for result in results:
        processed_frames += 1

        original_frame_idx = (processed_frames - 1) * VID_STRIDE

        timestamp_sec = original_frame_idx / fps if fps else None

        orig_h, orig_w = result.orig_shape
        orig_area = orig_w * orig_h

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for box_id, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
                x1, y1, x2, y2 = box

                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area = bbox_width * bbox_height

                share_of_screen = bbox_area / orig_area if orig_area > 0 else None

                class_name = result.names[int(cls_id)]

                detections.append({
                    "model": "YOLO11s",
                    "video": video_path.name,
                    "processed_frame_idx": processed_frames - 1,
                    "original_frame_idx": original_frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": float(conf),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "bbox_width": float(bbox_width),
                    "bbox_height": float(bbox_height),
                    "bbox_area_px": float(bbox_area),
                    "frame_width": int(orig_w),
                    "frame_height": int(orig_h),
                    "frame_area_px": int(orig_area),
                    "share_of_screen": float(share_of_screen),
                    "conf_threshold": CONF_THRESHOLD,
                    "imgsz": IMG_SIZE,
                    "vid_stride": VID_STRIDE
                })

                total_detections += 1

        if processed_frames % 500 == 0:
            print(f"Processed frames: {processed_frames}, detections so far: {total_detections}")
            gc.collect()

    detections_df = pd.DataFrame(detections)

    if len(detections_df) > 0:
        detections_df.to_csv(detections_csv_path, index=False, encoding="utf-8-sig")

        summary_by_class = (
            detections_df
            .groupby("class_name")
            .agg(
                detections_count=("class_name", "count"),
                avg_confidence=("confidence", "mean"),
                max_confidence=("confidence", "max"),
                avg_bbox_area_px=("bbox_area_px", "mean"),
                total_bbox_area_px=("bbox_area_px", "sum"),
                avg_share_of_screen=("share_of_screen", "mean"),
                total_share_of_screen=("share_of_screen", "sum"),
                first_timestamp_sec=("timestamp_sec", "min"),
                last_timestamp_sec=("timestamp_sec", "max"),
                unique_processed_frames=("processed_frame_idx", "nunique"),
                unique_original_frames=("original_frame_idx", "nunique")
            )
            .reset_index()
            .sort_values("detections_count", ascending=False)
        )

        seconds_per_processed_frame = VID_STRIDE / fps if fps else None

        if seconds_per_processed_frame is not None:
            summary_by_class["estimated_duration_sec"] = (
                summary_by_class["unique_processed_frames"] * seconds_per_processed_frame
            )

        summary_by_class.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")


        summary_by_frame = (
            detections_df
            .groupby(["processed_frame_idx", "original_frame_idx", "timestamp_sec"])
            .agg(
                detections_count=("class_name", "count"),
                detected_classes=("class_name", lambda x: ", ".join(sorted(set(x)))),
                avg_confidence=("confidence", "mean"),
                total_share_of_screen=("share_of_screen", "sum")
            )
            .reset_index()
            .sort_values("processed_frame_idx")
        )

        summary_by_frame.to_csv(frame_summary_csv_path, index=False, encoding="utf-8-sig")

    else:
        empty_columns = [
            "model", "video", "processed_frame_idx", "original_frame_idx",
            "timestamp_sec", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2", "bbox_width", "bbox_height",
            "bbox_area_px", "frame_width", "frame_height", "frame_area_px",
            "share_of_screen", "conf_threshold", "imgsz", "vid_stride"
        ]

        detections_df = pd.DataFrame(columns=empty_columns)
        detections_df.to_csv(detections_csv_path, index=False, encoding="utf-8-sig")

        print("\nNo detections found. Empty detection table was saved.")

    print("\nInference finished.")
    print(f"Processed frames: {processed_frames}")
    print(f"Total detections: {total_detections}")

    print("\nSaved files:")
    print(f"Video and YOLO outputs: {output_dir}")
    print(f"All detections CSV: {detections_csv_path}")
    print(f"Summary by class CSV: {summary_csv_path}")
    print(f"Summary by frame CSV: {frame_summary_csv_path}")


if __name__ == "__main__":
    main()