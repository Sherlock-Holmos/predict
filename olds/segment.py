from ultralytics import YOLO
import cv2
import numpy as np

# 加载 YOLO 模型
model = YOLO('yolov8n.pt')  # 确保模型路径正确

# 读取图像
image_left = cv2.imread('1_left.jpg')
image_right = cv2.imread('1_right.jpg')

# 进行目标检测
results_left = model(image_left)
results_right = model(image_right)

# 提取检测结果
def extract_boxes(results):
    boxes = []
    for result in results:
        # 获取所有边界框
        for box in result.boxes:
            boxes.append({
                'xmin': box.xmin.item(),
                'ymin': box.ymin.item(),
                'xmax': box.xmax.item(),
                'ymax': box.ymax.item(),
                'confidence': box.conf.item()
            })
    return boxes

boxes_left = extract_boxes(results_left)
boxes_right = extract_boxes(results_right)

# 打印检测结果
print("Left Image Boxes:", boxes_left)
print("Right Image Boxes:", boxes_right)
