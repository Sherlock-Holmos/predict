# 双目

import cv2
import numpy as np
import base64
from ultralytics import YOLO
import open3d as o3d

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

# 体积-质量回归模型
def volume_to_mass(volume):
    density = 1.1  # 假设鱼体密度为1.1 g/cm³
    mass = volume * density
    return mass

# 深度图和相机参数
def generate_point_cloud(depth_map, focal_length, baseline):
    """
    生成点云数据
    :param depth_map: 深度图像
    :param focal_length: 摄像头的焦距
    :param baseline: 双目相机基线距离
    :return: 生成的点云
    """
    h, w = depth_map.shape
    # 生成像素坐标网格
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # 计算真实世界中的坐标
    z = depth_map / 1000.0  # 假设深度图单位为毫米，将其转换为米
    x = (x - w / 2) * z / focal_length
    y = (y - h / 2) * z / focal_length
    
    # 将x, y, z合并为点云
    points = np.dstack((x, y, z)).reshape(-1, 3)
    
    # 移除无效点（z <= 0 的点）
    valid_points = points[points[:, 2] > 0]

    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(valid_points)
    return point_cloud

# 点云的三维重建与体积计算
def estimate_volume_from_point_cloud(point_cloud):
    """
    使用凸包算法估算点云的体积
    :param point_cloud: 点云数据
    :return: 估算的体积
    """
    # 使用 Alpha Shape 进行表面重建，alpha 值决定了表面细节的捕捉程度
    alpha = 0.05
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    
    # 计算三角网格的体积
    volume = mesh.get_volume()
    return volume

# 主函数
def process_fish_volume_estimation(left_image_path,right_image_path, focal_length, baseline):
    # 读取双目图像
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    # 计算深度图
    depth_map = StereoSGBM(left_image, right_image)

    if depth_map is None:
        print(f"无法读取深度图像")
        return
    
    # 生成点云
    point_cloud = generate_point_cloud(depth_map, focal_length, baseline)
    
    # 估算体积
    volume = estimate_volume_from_point_cloud(point_cloud)
    print(f"估算的体积为: {volume:.2f} 立方米")

# 示例调用
process_fish_volume_estimation('1_left.jpg','1_right.jpg', focal_length=1000, baseline=0.1)