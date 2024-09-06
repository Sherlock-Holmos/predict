from ultralytics import YOLO

# 加载模型
def load_model():
    model = YOLO('yolov8s.pt')

# 提取检测框
def detect_fish(image):
    model = YOLO('yolov8s.pt')
    results = model(image)
    detections = results[0].boxes.xyxy
    return detections