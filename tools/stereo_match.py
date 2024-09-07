import cv2
import numpy as np

# 图像预处理
def preprocess_image_SGBM(image):
    """
    预处理图像，适用于水下鱼类图像，减少噪声并增强对比度。
    
    参数:
    - image: 输入的图像 (BGR格式)

    返回:
    - preprocessed_image: 预处理后的图像
    """

    # 应用自适应直方图均衡化 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)

    # 双边滤波来减少噪声
    filtered_image = cv2.bilateralFilter(enhanced_image, d=9, sigmaColor=75, sigmaSpace=75)

    return filtered_image

def stereo_correct(left_img, right_img):
    # 相机内参和畸变系数
    left_camera_matrix = np.array([[1435.4, 0, 1080.0], [0, 1436.0, 603.8241], [0, 0, 1]])
    left_distortion = np.array([0.0732, -0.5106, 0.0001, -0.0087, 0])
    right_camera_matrix = np.array([[1433.4, 0, 1088.4], [0, 1436.2, 623.4802], [0, 0, 1]])
    right_distortion = np.array([0.0260, -0.4456, -0.0011, -0.0074, 0])

    # 相机外参
    R = np.array([[1.0000, -0.0017, -0.0037],
                [0.0017, 1.0000, -0.0089],
                [0.0037, 0.0089, 1.0000]])
    T = np.array([-120.5981, 0.1648, 2.4893])

    # 去畸变
    left_img_undistorted = cv2.undistort(left_img, left_camera_matrix, left_distortion)
    right_img_undistorted = cv2.undistort(right_img, right_camera_matrix, right_distortion)

    # 立体校正
    h, w = left_img.shape[:2]
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, (w, h), R, T)
    map1x, map1y = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, (w, h), cv2.CV_32FC1)
    left_img_rectified = cv2.remap(left_img_undistorted, map1x, map1y, cv2.INTER_LINEAR)
    right_img_rectified = cv2.remap(right_img_undistorted, map2x, map2y, cv2.INTER_LINEAR)

    return left_img_rectified, right_img_rectified

# 立体匹配
def StereoSGBM(left_img, right_img):
    # 创建立体匹配器（基于半全局匹配）
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*30,  # 确保是16的倍数
        blockSize=15,
        P1= 8 * 3 * 15^2,
        P2= 32 * 3 * 15^2,
        disp12MaxDiff=1,
        # uniquenessRatio=15,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 计算视差图
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    disparity = np.clip(disparity, 1e-6, None)  # 避免除以零
    
    disparity = postprocess_disparity(disparity) # 后处理

    # 深度计算参数
    focal_length = 2.1
    baseline = 120

    # 计算深度图
    depth_map = (focal_length * baseline) / (disparity + 1e-6)  # 避免除以零

    # 归一化深度图以便于显示
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)

    return disparity, depth_map_normalized

# 图像后处理
def postprocess_disparity(disparity):
    """
    后处理视差图，通过中值滤波和平滑去除噪声，并进行孔洞填充。
    
    参数:
    - disparity: 输入的视差图 (float32 格式)

    返回:
    - processed_disparity: 后处理后的视差图
    """
    # 中值滤波减少噪声
    filtered_disparity = cv2.medianBlur(disparity, 5)

    # 孔洞填充，填补由于噪声和误匹配造成的视差图中的空洞
    processed_disparity = cv2.morphologyEx(filtered_disparity, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))

    return processed_disparity  