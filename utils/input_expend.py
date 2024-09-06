import cv2
import numpy as np

# 图像输入扩展
def input_expend(input):
    if isinstance(input, str):
        # 如果是字符串类型，假设这是一个图像路径
        image = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图像文件: {input}")
    elif isinstance(input, np.ndarray):
        # 如果是 numpy 数组，假设这是一个图像矩阵
        image = input
        if len(image.shape) not in [2, 3]:
            raise ValueError("输入的图像矩阵的维度不正确")
    else:
        raise TypeError("输入必须是图像路径字符串或图像矩阵")
    
    # 如果图像是彩色图像（3 通道），则将其转换为灰度图像
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # 如果已经是灰度图像，则直接使用
        image = image
    
    return image