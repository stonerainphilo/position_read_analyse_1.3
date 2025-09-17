import numpy as np
import time

# 棱台顶点
SHiPvertices = np.array([
    [750, 2150, 45000],   # 0
    [750, -2150, 45000],  # 1
    [-750, -2150, 45000], # 2
    [-750, 2150, 45000],  # 3
    [2500, 5000, 95000],  # 4
    [2500, -5000, 95000], # 5
    [-2500, -5000, 95000],# 6
    [-2500, 5000, 95000]  # 7
])

def compute_affine_transform(vertices):
    """
    计算将棱台映射到单位立方体的仿射变换矩阵
    
    思路：
    1. 找到棱台的三个主轴方向
    2. 计算沿这些方向的缩放因子
    3. 计算平移使棱台位于第一象限
    """
    # 计算棱台的三个主轴方向
    # 底面中心
    bottom_center = np.mean(vertices[:4], axis=0)
    # 顶面中心
    top_center = np.mean(vertices[4:], axis=0)
    
    # Z轴方向 (从底面向顶面)
    z_axis = top_center - bottom_center
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # X轴方向 (底面的一条边)
    x_axis = vertices[0] - vertices[3]
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Y轴方向 (与X和Z轴正交)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 重新计算X轴以确保正交
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 计算沿各轴的范围
    # 将顶点投影到各轴上
    x_coords = np.dot(vertices, x_axis)
    y_coords = np.dot(vertices, y_axis)
    z_coords = np.dot(vertices, z_axis)
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    
    # 计算缩放矩阵
    scale_x = 1.0 / (x_max - x_min)
    scale_y = 1.0 / (y_max - y_min)
    scale_z = 1.0 / (z_max - z_min)
    
    # 计算平移矩阵
    translate_x = -x_min
    translate_y = -y_min
    translate_z = -z_min
    
    # 构建旋转矩阵 (从世界坐标到棱台坐标)
    R = np.array([x_axis, y_axis, z_axis]).T
    
    # 构建完整的仿射变换矩阵 (4x4)
    # 先旋转，再缩放，最后平移
    M = np.eye(4)
    M[:3, :3] = np.diag([scale_x, scale_y, scale_z]) @ R.T
    M[:3, 3] = [-translate_x*scale_x, -translate_y*scale_y, -translate_z*scale_z]
    
    return M

def is_point_inside_using_affine(point, vertices):
    """
    使用仿射变换判断点是否在棱台内部
    """
    # 计算仿射变换矩阵
    M = compute_affine_transform(vertices)
    
    # 将点转换为齐次坐标
    point_homogeneous = np.append(point, 1.0)
    
    # 应用变换
    transformed_point = M @ point_homogeneous
    
    # 检查变换后的点是否在单位立方体内
    return (0 <= transformed_point[0] <= 1 and 
            0 <= transformed_point[1] <= 1 and 
            0 <= transformed_point[2] <= 1)




# 预先计算变换矩阵
M = compute_affine_transform(SHiPvertices)

def is_point_inside_using_precomputed_affine(point, M):
    """
    使用预先计算的仿射变换判断点是否在棱台内部
    """
    # 将点转换为齐次坐标
    point_homogeneous = np.append(point, 1.0)
    
    # 应用变换
    transformed_point = M @ point_homogeneous
    
    # 检查变换后的点是否在单位立方体内
    return (0 <= transformed_point[0] <= 1 and 
            0 <= transformed_point[1] <= 1 and 
            0 <= transformed_point[2] <= 1)

# 批量检查多个点
def are_points_inside_using_affine(points, M):
    """
    批量检查多个点是否在棱台内部
    points: (N, 3) 数组
    """
    # 转换为齐次坐标
    homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
    
    # 应用变换
    transformed_points = (M @ homogeneous_points.T).T
    
    # 检查哪些点在单位立方体内
    inside_x = (transformed_points[:, 0] >= 0) & (transformed_points[:, 0] <= 1)
    inside_y = (transformed_points[:, 1] >= 0) & (transformed_points[:, 1] <= 1)
    inside_z = (transformed_points[:, 2] >= 0) & (transformed_points[:, 2] <= 1)
    
    return inside_x & inside_y & inside_z


# 示例使用
test_point = np.array([1000, 2500, 56000])
result = is_point_inside_using_affine(test_point, SHiPvertices)
print(f"点 {test_point} 在棱台内部: {result}")

# test_points = np.random.rand(10000000, 3) * 10000  # 生成测试点

# start_time = time.time()
# results = are_points_inside_using_affine(test_points, M)
# end_time = time.time()

# print(f"处理 {len(test_points)} 个点耗时: {end_time - start_time:.4f} 秒")
# print(f"平均每个点耗时: {(end_time - start_time) / len(test_points) * 1e6:.4f} 微秒")