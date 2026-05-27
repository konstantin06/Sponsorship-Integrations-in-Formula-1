from roboflow import Roboflow

rf = Roboflow(api_key="*")
project = rf.workspace("konstantins-workspace-zooqz").project("f1_sponsors_dataset")
version = project.version(5)
dataset = version.download("yolov11", location="../final_dataset_yolov11")
print("Dataset downloaded to:", dataset.location)