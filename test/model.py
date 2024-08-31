from ultralytics import YOLO

# 加载YOLO模型
model = YOLO('yolov8s.pt')

def YOLO_model(image):
    return model(image)
