import cv2
import numpy as np
import os

def process_image(input_image_path, output_dir='out'):
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
        
        # 输出特征信息
        print(f'最大轮廓的面积: {area}')
        print(f'最大轮廓的周长: {perimeter}')
        print(f'最大轮廓的长宽比: {aspect_ratio}')

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
        
        # 输出特征信息
        print(f'边界框宽度: {width}')
        print(f'边界框长度: {length}')
        print(f'最小外接矩形的宽度: {rect_width}')
        print(f'最小外接矩形的长度: {rect_length}')

    # 保存结果
    cv2.imwrite(os.path.join(output_dir, 'contour_image.png'), contour_image)
    cv2.imwrite(os.path.join(output_dir, 'edge_image.png'), edges)

# 示例用法
process_image('morph.png')
