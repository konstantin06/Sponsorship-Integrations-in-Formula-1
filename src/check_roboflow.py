from roboflow import Roboflow

ROBOFLOW_API_KEY = "*"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

print(rf.workspace())