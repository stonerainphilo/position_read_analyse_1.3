from numba import jit
import numpy as np
import time



import numpy as np

SHiPvertices = np.array([
    [750, 2150, 45000],  # Bottom Vertices
    [750, -2150, 45000],
    [-750, -2150, 45000],
    [-750, 2150, 45000],
    [2500, 5000, 95000],   # Top Vertices
    [2500, -5000, 95000],
    [-2500, -5000, 95000],
    [-2500, 5000, 95000]]) 
SHiPfaces = [
    [0, 1, 2, 3],  # Bottom face
    [4, 5, 6, 7],  # Top face
    [0, 1, 5, 4],  # Side faces
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7]]

# 计算棱台中心点
center = np.mean(SHiPvertices, axis=0)  # 大约为 [0, 0, 70000]

# 预处理：计算所有面的平面方程，确保法向量指向外部
planes = []
for face in SHiPfaces:
    # 取面上的三个点计算法向量
    p0 = SHiPvertices[face[0]]
    p1 = SHiPvertices[face[1]]
    p2 = SHiPvertices[face[2]]
    
    # 计算两个向量
    v1 = p1 - p0
    v2 = p2 - p0
    
    # 计算法向量并单位化
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # 计算平面方程的 D 值
    D = -np.dot(normal, p0)
    
    # 检查法向量方向：如果指向内部，则反转
    # 计算中心点到平面的有符号距离
    distance_to_center = np.dot(normal, center) + D
    if distance_to_center > 0:
        # 法向量指向中心点（内部），需要反转
        normal = -normal
        D = -np.dot(normal, p0)
    
    planes.append((normal, D))

# 后续函数保持不变
def is_point_inside_frustum_optimized(point, planes):
    for normal, D in planes:
        distance = normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] + D
        if distance > 1e-6:
            return False
    return True

# 使用 Numba 加速的函数也需要使用修正后的 planes
# 注意：需要将 planes 转换为 Numba 兼容格式
planes_numba = [(np.array(normal, dtype=np.float64), np.float64(D)) for normal, D in planes]

from numba import jit, prange

# 修改函数以处理点数组
@jit(nopython=True, parallel=True)
def is_point_inside_frustum_numba_array(points, planes_numba):
    """
    处理点数组版本的函数
    points: (N, 3)数组，N个点的坐标
    """
    results = np.empty(len(points), dtype=np.bool_)
    
    # 使用并行循环
    for i in prange(len(points)):
        point = points[i]
        inside = True
        for j in range(len(planes_numba)):
            normal, D = planes_numba[j]
            distance = normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] + D
            if distance > 1e-6:
                inside = False
                break
        results[i] = inside
    
    return results

# 测试代码
# test_points = np.random.rand(320000, 3) * 10000  # 生成测试点

# start_time = time.time()
# results = is_point_inside_frustum_numba_array(test_points, planes_numba)
# end_time = time.time()

# print(f"处理 {len(test_points)} 个点耗时: {end_time - start_time:.4f} 秒")
# print(f"平均每个点耗时: {(end_time - start_time) / len(test_points) * 1e6:.4f} 微秒")
# print(f"内部点的数量: {np.sum(results)}")

# point = [1000, 2500, 56000]
# result = is_point_inside_frustum_numba(point, planes)
# print(f"点 {point} 在棱台内部: {result}")