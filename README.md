# Formula 1 Sponsor Visibility Analysis

This project is devoted to the automated analysis of sponsor logo visibility in Formula 1 race broadcasts using computer vision methods. The main goal is to detect sponsor logos in video frames, measure their screen presence, and use these indicators for further interpretation of sponsorship effectiveness.

The project combines object detection, video processing and basic media exposure analytics. Two models were trained and compared: RF-DETR and YOLO. The final analysis is based mainly on the locally trained RF-DETR model, which showed better practical performance on race video fragments.

## Project goal

The goal of the project is to develop and evaluate a reproducible pipeline for sponsor logo detection in Formula 1 broadcasts.

The pipeline includes:

- extracting frames from race videos;
- detecting sponsor logos with trained object detection models;
- comparing RF-DETR and YOLO models;
- calculating visibility indicators by sponsor;
- applying the final model to race video fragments;
- interpreting the obtained results from an economic perspective.

## Dataset

The dataset was created from Formula 1 race broadcast videos. The videos were sampled at 1 FPS, after which frames with visible sponsor logos were selected manually.

The first part of the dataset was annotated manually in Roboflow using horizontal bounding boxes. After approximately 350 images had been manually labeled, the remaining images were annotated using a previously trained model and then corrected manually.

Final dataset characteristics:

| Indicator | Value |
|---|---:|
| Number of images | 1084 |
| Number of annotations | 2684 |
| Number of classes | 10 |
| Average annotations per image | 2.5 |
| Median image resolution | 1920 × 1080 |
| Train / validation / test split | 80 / 10 / 10 |

The dataset contains the following sponsor classes:

- OKX
- HP
- Aramco
- Petronas
- BWT
- MoneyGram
- Oracle
- Atlassian
- Kick
- Cash App

The class distribution is imbalanced, which reflects the real structure of race broadcasts: some teams and sponsors appear on screen more often than others.

## Models

Two object detection models were trained and compared:

1. **RF-DETR Small**
2. **YOLO11s**

Before local training, an RF-DETR Small model was also trained through Roboflow. However, after manual inspection of video fragments and analysis of detection tables, the locally trained model showed better results. Therefore, the final experiments were conducted using locally trained model weights.

## Repository structure

```text
.
├── metrics_comparison/
│   └── model comparison tables, summary metrics and plots
│
├── outputs_train_rfdetr/
│   └── final_rfdetr_f1_local/
│       └── RF-DETR training outputs and model weights
│
├── outputs_train_yolo/
│   └── yolo11s_f1_baseline_img640/
│       └── YOLO training outputs, metrics and validation plots
│
├── outputs_video_rfdetr/
│   └── race_fragment_rfdetr/
│       └── RF-DETR predictions on video fragments
│
├── video_outputs/
│   └── yolo_monaco_stream_result1/
│       └── YOLO predictions on the Monaco video fragment
│
├── src/
│   └── source code for training, prediction and analysis
│
└── final_report_f1_sponsorship_3coursework_...
    └── final project report
