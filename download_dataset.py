from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ACTUAL_KEY_HERE")

project = rf.workspace("acne-detection-oface").project("acne-detection-tkbq0")
dataset = project.version(2).download("yolov8")

print("Dataset downloaded successfully!")
print(f"Location: {dataset.location}")