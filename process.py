from ultralytics import YOLO
import cv2
import os
import numpy as np

def process_fish_images(image_path, model_path='yolov8s.pt', min_area=4000):
    # 加载YOLO模型
    model = YOLO(model_path)

    # 读取图像
    image = cv2.imread(image_path)

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

        # 保存每一步的图像到out文件夹
        cv2.imwrite(os.path.join(output_dir, f'enhanced_{i}.png'), enhanced)
        cv2.imwrite(os.path.join(output_dir, f'scrop_{i}.png'), fish_image)
        cv2.imwrite(os.path.join(output_dir, f'gray_{i}.png'), gray)
        cv2.imwrite(os.path.join(output_dir, f'blurred_{i}.png'), blurred)
        cv2.imwrite(os.path.join(output_dir, f'equalized_{i}.png'), equalized)
        cv2.imwrite(os.path.join(output_dir, f'edges_{i}.png'), edges)
        cv2.imwrite(os.path.join(output_dir, f'morph_{i}.png'), morph)
        cv2.imwrite(os.path.join(output_dir, f'morph_open_{i}.png'), morph_open)
        cv2.imwrite(os.path.join(output_dir, f'mask_{i}.png'), mask)
        cv2.imwrite(os.path.join(output_dir, f'filtered_image_{i}.png'), filtered_image)
        cv2.imwrite(os.path.join(output_dir, f'final_mask_{i}.png'), final_mask)
        cv2.imwrite(os.path.join(output_dir, f'final_filtered_image_{i}.png'), final_filtered_image)

# 示例用法:
process_fish_images('fish_image.jpg')
