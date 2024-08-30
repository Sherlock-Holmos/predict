import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os

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

def weight_analyse(input_image_path, output_dir='out'):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    morph = cv2.imread(input_image_path)

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
        
        # 计算轮廓的长宽比
        aspect_ratio = w / h
        
        # 绘制边界框
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 使用边界框
        width = w
        length = h

        # 使用最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 计算最小外接矩形的宽度和长度
        rect_width = min(rect[1])
        rect_length = max(rect[1])
        
        # 绘制最小外接矩形
        cv2.drawContours(contour_image, [box], 0, (0, 0, 255), 2)

        # 归纳信息
        # 初始化结果
        result = {
            'area': None,
            'perimeter': None,
            'aspect_ratio': None,
            'length': None,
            'width': None,
            'rect_length': None,
            'rect_width': None,
            'contour_image_path': None
        }

        result['area'] = area
        result['perimeter'] = perimeter
        result['aspect_ratio'] = aspect_ratio
        result['length'] = length
        result['width'] = width
        result['rect_length'] = rect_length
        result['rect_width'] = rect_width

    return result
    
def process_fish_images(image):
    # 运行模型获取结果
    results = model(image)

    # 提取边界框检测结果
    detections = results[0].boxes.xyxy

    # 确保输出目录存在
    output_dir = 'out'
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个检测到的对象
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection[:4])

        # 裁剪鱼的图像
        fish_image = image[y1:y2, x1:x2]

        # 转为灰度图
        gray = cv2.cvtColor(fish_image, cv2.COLOR_BGR2GRAY)
        
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
        morph_open = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤掉小的轮廓
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # 创建掩码去除小轮廓
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, large_contours, -1, 255, thickness=cv2.FILLED)

        # 应用掩码来过滤图像
        filtered_image = cv2.bitwise_and(fish_image, fish_image, mask=mask)

        # 查找最终轮廓
        final_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建最终掩码
        final_mask = np.zeros_like(gray)
        cv2.drawContours(final_mask, final_contours, -1, 255, thickness=cv2.FILLED)

        # 应用最终掩码
        final_filtered_image = cv2.bitwise_and(filtered_image, filtered_image, mask=final_mask)

        # 在图像上绘制最终轮廓
        cv2.drawContours(final_filtered_image, final_contours, -1, (0, 255, 0), 2)

    return morph

def process(image):
    morph = process_fish_images(image)
    result = weight_analyse(morph)
    
    # 从 result 中提取特征
    area = result['area']
    perimeter = result['perimeter']
    aspect_ratio = result['aspect_ratio']
    length = result['length']
    width = result['width']

    fish_mass = weight(length,width)
    estimated_feed = 20 * fish_mass

    return {
    "area": area,
    "perimeter": perimeter,
    "aspect_ratio": aspect_ratio,
    "mass": fish_mass,
    "estimated_feed": estimated_feed,
    "length": length,
    "width": width
}