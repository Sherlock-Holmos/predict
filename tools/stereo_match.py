import cv2
import numpy as np

# 立体匹配
def StereoSGBM(left_img, right_img):
    # 创建立体匹配器（基于半全局匹配）
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*30,  # 确保是16的倍数
        blockSize=11,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        # uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 计算视差图
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    disparity = np.clip(disparity, 1e-6, None)  # 避免除以零

    # 深度计算参数
    focal_length = 2.1
    baseline = 120

    # 计算深度图
    depth_map = (focal_length * baseline) / (disparity + 1e-6)  # 避免除以零

    # 归一化深度图以便于显示
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)

    return disparity, depth_map_normalized