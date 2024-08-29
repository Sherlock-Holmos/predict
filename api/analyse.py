from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# 加载YOLO模型
model = YOLO('yolov8s.pt')

@app.route('/process_fish', methods=['POST'])
def process_fish():
    # 读取上传的图像
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
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
            
            # 计算边界框的长宽比
            x, y, w, h = cv2.boundingRect(max_contour)
            aspect_ratio = float(w) / h
            
            # 将目标检测图像编码为Base64
            _, marked_buffer = cv2.imencode('.png', marked_image)
            marked_image_b64 = base64.b64encode(marked_buffer).decode('utf-8')

            # 将边缘检测图像编码为Base64
            _, edges_buffer = cv2.imencode('.png', edges)
            edges_image_b64 = base64.b64encode(edges_buffer).decode('utf-8')
            
            # 计算鱼体质量和预估投食量 (占位符)
            fish_mass = 1.25  # 假设为1.25公斤
            estimated_feed = fish_mass * 20  # 假设每公斤鱼需要20克食物
            
            # 返回结果，包括图像和几何特征
            return jsonify({
                "marked_image": marked_image_b64,
                "edges_image": edges_image_b64,
                "area": area,
                "perimeter": perimeter,
                "aspect_ratio": aspect_ratio,
                "mass": fish_mass,
                "estimated_feed": estimated_feed
            })
        else:
            return jsonify({"error": "No contours found"}), 400
    else:
        return jsonify({"error": "No fish detected"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
