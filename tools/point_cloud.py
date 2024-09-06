import numpy as np
import open3d as o3d

# 生成点云
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
    o3d.visualization.draw_geometries([point_cloud], window_name="Generated Point Cloud")
    return point_cloud

# 裁剪点云数据
def crop_fish_region_from_point_cloud(point_cloud, bbox):
    # 将 Tensor 转换为 ndarray
    bbox = bbox.numpy()
    xmin, ymin, xmax, ymax = bbox

    # 将点云转换为 numpy 数组
    points = np.asarray(point_cloud.points)
    
    # 假设相机深度图对应的点云，Z值作为深度，进行简单筛选
    mask = (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & \
           (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
    
    # 提取满足条件的点
    cropped_points = points[mask]
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
    return cropped_pcd
# # 使用Alpha Shape重建裁剪后的点云并估算体积
# def estimate_volume_from_point_cloud(point_cloud):
#     alpha = 0.05  # Alpha值调节表面重建的精细度
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
#     volume = mesh.get_volume()  # 计算体积
#     return volume

# 点云的三维重建与体积计算
def estimate_volume_from_point_cloud(point_cloud):
    """
    使用不同的方法估算点云的体积。
    
    :param point_cloud: 点云数据
    :return: 估算的体积
    """
    # 检查点云是否为空
    if not point_cloud.has_points():
        print("点云为空或无效，无法进行体积估算。")
        return 0

    print(f"点云包含 {len(point_cloud.points)} 个点。")
    
    # 计算法线以便 Poisson 重建
    try:
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        print("法线估算成功。")
    except Exception as e:
        print(f"法线估算失败: {e}")
        return 0
    
    # 尝试使用 Poisson 重建网格
    try:
        print("使用 Poisson 方法进行三维重建...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=9
        )
        
        # 过滤掉低密度的顶点，清理网格
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # 计算并返回三角网格的体积
        volume = mesh.get_volume()
        print(f"估算的体积: {volume:.2f}")
        return volume
    except Exception as e:
        print(f"Poisson 方法失败: {e}")
    
    # 如果 Poisson 重建失败，则使用 Alpha Shape 作为备选方案
    try:
        print("尝试使用 Alpha Shape 进行重建...")
        alpha = 0.05
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
        
        # 计算并返回三角网格的体积
        volume = mesh.get_volume()
        print(f"Alpha Shape 估算的体积: {volume:.2f}")
        return volume
    except Exception as e:
        print(f"Alpha Shape 方法失败: {e}")
    
    # 使用凸包计算体积作为最后的备选方案
    try:
        print("使用 Convex Hull 进行体积估算...")
        hull, _ = point_cloud.compute_convex_hull()
        hull_volume = hull.get_volume()
        print(f"凸包估算的体积: {hull_volume:.2f}")
        return hull_volume
    except Exception as e:
        print(f"凸包方法失败: {e}")

    # 如果所有方法均失败，返回体积为零
    print("无法估算体积。")
    return 0