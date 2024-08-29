import cv2
import numpy as np
import base64
from ultralytics import YOLO

# 加载YOLO模型
model = YOLO('yolov8s.pt')

def weight(length, width):
    x = length
    z = width
    # weight-length
    y = 1 / ( 0 + 163.1664135544264 * 0.9807226646189438**x )
    # weight-width
    w = 1 / ( 0 + 336.1868514764623 * 0.9396616862616823**z )
    weight = (y + w) / 2
    return weight

def process_image(image):
    """
    处理上传的图像，检测鱼体并返回图像和几何特征。
    """
    # 使用YOLO模型检测鱼体
    results = model(image)
    detections = results[0].boxes.xyxy

    if len(detections) > 0:
        # 获取检测到的鱼体的边界框
        x1, y1, x2, y2 = map(int, detections[0][:4])
        
        # 绘制红色边框在原始图像上
        marked_image = image.copy()
        cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 提取鱼体并进行Canny边缘检测
        fish_image = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(fish_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 175)
        
        # 找到轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到面积最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            
            # 计算面积
            area = cv2.contourArea(max_contour)
            
            # 计算周长
            perimeter = cv2.arcLength(max_contour, True)

            # 计算鱼体的长度和宽度
            length = (x2 - x1) / image.shape[1]  # 归一化长度
            width = (y2 - y1) / image.shape[0]  # 归一化宽度
            
            # 计算边界框的长宽比
            x, y, w, h = cv2.boundingRect(max_contour)
            aspect_ratio = float(w) / h

            # 估算鱼体质量
            fish_mass = weight(length, width)

            # 计算预估投食量
            estimated_feed = fish_mass * 20  # 假设每公斤鱼需要20克食物
            
            # 将目标检测图像编码为Base64
            _, marked_buffer = cv2.imencode('.png', marked_image)
            marked_image_b64 = base64.b64encode(marked_buffer).decode('utf-8')

            # 将边缘检测图像编码为Base64
            _, edges_buffer = cv2.imencode('.png', edges)
            edges_image_b64 = base64.b64encode(edges_buffer).decode('utf-8')
            
            return {
                "marked_image": marked_image_b64,
                "edges_image": edges_image_b64,
                "area": area,
                "perimeter": perimeter,
                "aspect_ratio": aspect_ratio,
                "mass": fish_mass,
                "estimated_feed": estimated_feed,
                "length": length,
                "width": width
            }
        else:
            return {"error": "No contours found"}, 400
    else:
        return {"error": "No fish detected"}, 400
