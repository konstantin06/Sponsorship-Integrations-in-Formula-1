import cv2
import os

video_path = "hl6.mp4"
output_dir = "frames_hl6"
fps_extract = 1

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(video_fps / fps_extract)

frame_id = 0
saved_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{saved_id:06d}.jpg")
        cv2.imwrite(filename, frame)
        saved_id += 1

    frame_id += 1

cap.release()

print(f"Saved {saved_id} frames")