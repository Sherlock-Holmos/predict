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

def process(image):
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用CLAHE进行对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 直方图均衡化
    equalized = cv2.equalizeHist(enhanced)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # 使用Canny进行边缘检测
    edges = cv2.Canny(blurred, 18, 65)
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return morph

def process_image(image):
    """
    处理上传的图像，检测鱼体并返回图像和几何特征。
    """
    # 使用YOLO模型检测鱼体
    results = model(image)
    detections = results[0].boxes.xyxy

    if len(detections) > 0:
            for i, detection in enumerate(detections):
                # 获取检测到的鱼体的边界框
                x1, y1, x2, y2 = map(int, detections[0][:4])
                
                # 绘制红色边框在原始图像上
                marked_image = image.copy()
                cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                morph = process(image)
                
                # 使用Canny边缘检测
                edges = cv2.Canny(morph, 40, 160)

                # 闭运算
                kernel = np.ones((15, 15), np.uint8)
                closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                # 查找轮廓
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 在原图像上绘制轮廓
                contour_image = morph.copy()
                cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

                # 提取和绘制主要轮廓
                if contours:
                    # 找到最大面积的轮廓
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # 计算轮廓的边界框
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # 计算轮廓的面积
                    area = cv2.contourArea(largest_contour)
                    
                    # 计算轮廓的周长
                    perimeter = cv2.arcLength(largest_contour, True)

                    # 绘制边界框
                    cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 使用最小外接矩形
                    rect = cv2.minAreaRect(largest_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # 计算最小外接矩形的宽度和长度
                    rect_width = min(rect[1])
                    rect_length = max(rect[1])
                    
                    # 计算最小外接矩形的长宽比
                    aspect_ratio = rect_length / rect_width

                    # 绘制最小外接矩形
                    cv2.drawContours(contour_image, [box], 0, (0, 0, 255), 2)

                    # 估算鱼体质量
                    fish_mass = weight(rect_length, rect_width)

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
                        "length": rect_length,
                        "width": rect_width
                    }
                else:
                    return {"error": "No contours found"}, 400
    else:
        return {"error": "No fish detected"}, 400