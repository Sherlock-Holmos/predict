# 双目
import cv2
from tools.preprocess import preprocess_image, correct, split_image
from tools.regression import volume_to_mass
from tools.stereo_match import StereoSGBM, preprocess_image_SGBM, postprocess_disparity, stereo_correct
from tools.point_cloud import generate_point_cloud, estimate_volume_from_point_cloud
from tools.model_load import load_model, detect_fish

# 检测深度图像
def map(left_image_path,right_image_path):
    left_image = preprocess_image(left_image_path)
    right_image = preprocess_image(right_image_path)

    left_image, right_image = stereo_correct(left_image, right_image)

    disparity, depth_map = StereoSGBM(left_image, right_image)

    disparity_out_path = 'out/视差图.png'
    depth_map_out_path = 'out/深度图像.png'
    cv2.imwrite(disparity_out_path, disparity)
    cv2.imwrite(depth_map_out_path, depth_map)


# 完整估算体积
def test(left_image_path,right_image_path, focal_length, baseline):
    left_image = preprocess_image(left_image_path)
    right_image = preprocess_image(right_image_path)

    # 检测鱼类
    detections = detect_fish(left_image_path)

    disparity, depth_map = StereoSGBM(left_image, right_image)

    # 假设仅处理第一个检测到的鱼类
    if len(detections) > 0:
        bbox = detections[0]  # 取第一个检测框
        print(f"检测到的鱼类边界框: {bbox}")

        # 生成点云
        point_cloud = generate_point_cloud(depth_map, focal_length, baseline)

        # 裁剪点云并计算体积
        # cropped_pcd = crop_fish_region_from_point_cloud(point_cloud, bbox)
        volume = estimate_volume_from_point_cloud(point_cloud)

        print(f"检测到鱼类的估算体积: {volume:.2f} 立方单位")
    else:
        print("未检测到鱼类。")

# test('1_left.jpg','1_right.jpg', focal_length=2.1, baseline=120)
map('input/1_left_corrected.jpg','input/1_right_corrected.jpg')