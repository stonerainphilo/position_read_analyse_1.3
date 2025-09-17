# from numba import jit
import numpy as np
import time


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

# 预处理：计算所有面的平面方程
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
    
    # 计算平面方程的D值
    D = -np.dot(normal, p0)
    
    planes.append((normal, D))

# 保存planes供后续使用

def is_point_inside_frustum_optimized(point, planes):
    """
    优化版的点是否在棱台内部判断
    """
    # 使用预计算的平面方程
    for normal, D in planes:
        # 计算有符号距离
        distance = normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] + D
        
        # 如果距离为正，点在外部
        if distance > 1e-6:  # 使用小的容差值
            return False
    
    return True

def are_points_inside_frustum(points, planes):
    """
    批量检查多个点是否在棱台内部
    points: (N, 3)数组，N个点的坐标
    """
    results = np.ones(len(points), dtype=bool)
    
    for normal, D in planes:
        # 计算所有点到当前面的有符号距离
        distances = np.dot(points, normal) + D
        
        # 标记距离为正的点（在外部）
        outside_mask = distances > 1e-6
        results[outside_mask] = False
        
        # 如果所有点都已标记为外部，提前终止
        if not np.any(results):
            break
    
    return results



# @jit(nopython=True)
def is_point_inside_frustum_numba(point, planes):
    """
    使用Numba加速的点是否在棱台内部判断
    """
    for i in range(len(planes)):
        normal, D = planes[i]
        distance = normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] + D
        
        if distance > 1e-6:
            return False
    
    return True

# 性能测试
test_points = np.random.rand(10000000, 3) * 10000  # 生成测试点

start_time = time.time()
results = are_points_inside_frustum(test_points, planes)
end_time = time.time()

print(f"处理 {len(test_points)} 个点耗时: {end_time - start_time:.4f} 秒")
print(f"平均每个点耗时: {(end_time - start_time) / len(test_points) * 1e6:.4f} 微秒")
