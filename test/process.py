import cv2
import numpy as np

def process(image):
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
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

    return morph