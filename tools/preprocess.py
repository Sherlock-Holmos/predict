import cv2
import os
import numpy as np
from utils.input_expend import input_expend

# 图像预处理，输入可为图像或路径
def preprocess_image(image_path):
    """
    对输入图像进行预处理，包括降噪、直方图均衡化等
    :param image_path: 图像路径
    :return: 预处理后的图像
    """
    image = input_expend(image_path)

    # 高斯滤波：减少高频噪声
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 双边滤波：降噪同时保留边缘细节
    image = cv2.bilateralFilter(image, 9, 75, 75)

    # 中值滤波：有效去除椒盐噪声
    image = cv2.medianBlur(image, 5)

    # 自适应直方图均衡化：增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    return image


# 图像校正
def correct(left_img, right_img):
    # 畸变校正
    return left_img, right_img


# 双目图像分割并保存
def split_and_save_image(image_path, output_folder):
    # 读取图像
    img = cv2.imread(image_path)
    
    if img is None:
        print("无法读取图像文件")
        return
    
    # 获取图像尺寸
    height, width, _ = img.shape
    
    # 计算中间分割位置
    mid_x = width // 2
    
    # 分割图像
    left_img = img[:, :mid_x]
    right_img = img[:, mid_x:]
    
    # 构建输出路径
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    left_output_path = os.path.join(output_folder, f"{base_name}_left.jpg")
    right_output_path = os.path.join(output_folder, f"{base_name}_right.jpg")
    
    # 保存图像
    cv2.imwrite(left_output_path, left_img)
    cv2.imwrite(right_output_path, right_img)

# 双目图像分割不保存
def split_image(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    
    if img is None:
        print("无法读取图像文件")
        return
    
    # 获取图像尺寸
    height, width, _ = img.shape
    
    # 计算中间分割位置
    mid_x = width // 2
    
    # 分割图像
    left_img = img[:, :mid_x]
    right_img = img[:, mid_x:]
    
    return left_img, right_img