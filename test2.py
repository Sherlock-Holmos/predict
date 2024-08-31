# 双目

import cv2
import numpy as np
import base64
from ultralytics import YOLO

# 加载YOLO模型
model = YOLO('yolov8s.pt')

# 提取检测框
def detect_fish(image):
    results = model(image)
    detections = results[0].boxes.xyxy
    return detections

# 图像校正
def correct(left_img, right_img):
    # 畸变校正  
    return left_img, right_img

# 立体匹配
def StereoSGBM(left_img, right_img):
    # 创建立体匹配器（基于半全局匹配）
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*8,  # 必须是16的倍数
        blockSize=11,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 计算视差图
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    # 深度计算参数
    focal_length = 0.8  # 焦距（单位根据相机校准结果而定）
    baseline = 0.1  # 相机基线距离（单位：米）

    # 计算深度图
    depth_map = (focal_length * baseline) / (disparity + 1e-6)  # 避免除以零

    return depth_map

# 三维重建与体积计算
def calculate_volume(depth_map, detections):
    volume = 0
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        fish_depth = depth_map[y1:y2, x1:x2]
        # 计算每个像素对应的三维坐标
        points = np.dstack(np.meshgrid(np.arange(fish_depth.shape[1]), np.arange(fish_depth.shape[0])))[..., ::-1]
        # 将深度信息加入坐标中
        points = np.concatenate((points, fish_depth[..., None]), axis=-1)
        # 使用体积计算（例如体素法或点云体积估计）
        # 简化为体素体积计算
        volume += np.sum(fish_depth > 0) * (体素大小)  # 计算体积
    return volume

# 主函数
def estimate_fish_mass(left_image_path, right_image_path):
    # 读取双目图像
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    # 计算深度图
    depth_map = StereoSGBM(left_image, right_image)

    # 检测鱼体
    detections = detect_fish(left_image)

    if len(detections) == 0:
        print("未检测到鱼体")
        return

    # 计算鱼体积
    volume = calculate_volume(depth_map, detections)

    # 估算鱼质量
    mass = volume_to_mass(volume)

    print(f"估算的鱼质量为: {mass:.2f} 克")

# 示例调用
estimate_fish_mass('left_fish_image.jpg', 'right_fish_image.jpg')
