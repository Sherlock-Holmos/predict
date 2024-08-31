import cv2
import numpy as np

# # 加载相机矩阵和畸变系数（从校准步骤中获得）
# # 这些数据应从相机标定中获得
# camera_matrix_left = np.load('camera_matrix_left.npy')
# dist_coeffs_left = np.load('dist_coeffs_left.npy')
# camera_matrix_right = np.load('camera_matrix_right.npy')
# dist_coeffs_right = np.load('dist_coeffs_right.npy')

# # 加载畸变校正图像
# img_left = cv2.imread('left_image.jpg')
# img_right = cv2.imread('right_image.jpg')

# # 进行校正
# h_left, w_left = img_left.shape[:2]
# h_right, w_right = img_right.shape[:2]

# # 计算校正矩阵
# map1_left, map2_left = cv2.initUndistortRectifyMap(camera_matrix_left, dist_coeffs_left, None, camera_matrix_left, (w_left, h_left), 5)
# map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, None, camera_matrix_right, (w_right, h_right), 5)

# # 校正图像
# rectified_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
# rectified_right = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

# # 保存校正后的图像
# cv2.imwrite('rectified_left.jpg', rectified_left)
# cv2.imwrite('rectified_right.jpg', rectified_right)

# 加载校正后的图像
img_left = cv2.imread('1_left.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('1_right.jpg', cv2.IMREAD_GRAYSCALE)

# 创建 StereoBM 对象
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# 计算视差图
disparity = stereo.compute(img_left, img_right)

# 保存视差图
cv2.imwrite('disparity.png', disparity)
