from model import YOLO_model
from image_processing import process_image
import cv2

# 主函数
def process_fish(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    result = process_image(image)

    # 使用字典访问
    print(f'长度：{result["length"]}')
    print(f'宽度：{result["width"]}')
    print(f'质量：{result["mass"]}')
    print(f'长宽比：{result["aspect_ratio"]}')

# 调用
process_fish('fish_image.jpg')
