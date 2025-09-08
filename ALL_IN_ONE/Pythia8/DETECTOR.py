import numpy as np


# SHiP Paras
# Vertices are defined by their coordinates
SHiPvertices = [
    [750, 2150, 45000],  # Bottom Vertices
    [750, -2150, 45000],
    [-750, -2150, 45000],
    [-750, 2150, 45000],
    [2500, 5000, 95000],   # Top Vertices
    [2500, -5000, 95000],
    [-2500, -5000, 95000],
    [-2500, 5000, 95000]
    
]

# Similar to the cube, faces are defined by the indices of the vertices
SHiPfaces = [
    [0, 1, 2, 3],  # Bottom face
    [4, 5, 6, 7],  # Top face
    [0, 1, 5, 4],  # Side faces
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7]
]


def moller_trumbore(v0, v1, v2, orig, dir):
        S = orig - v0
        E1 = v1 - v0
        E2 = v2 - v0
        S1 = np.cross(dir, E2)
        S2 = np.cross(S, E1)

        S1E1 = np.dot(S1, E1)
        if S1E1 == 0:
            return False, None, None, None

        t = np.dot(S2, E2) / S1E1
        b1 = np.dot(S1, S) / S1E1
        b2 = np.dot(S2, dir) / S1E1

        if t >= 0.0 and b1 >= 0.0 and b2 >= 0.0 and (1 - b1 - b2) >= 0.0:
            return True, t, b1, b2

        return False, None, None, None

def judge_paralle(ray, v0, v1, v2):
    # True if ray is parallel to the plane defined by v0, v1, v2
    h = np.cross(v1 - v0, v2 - v0)
    len_h = np.linalg.norm(h)
    len_ray = np.linalg.norm(ray)
    if np.dot(ray, h) == 0:
        return True 
    
    else:
        return False
    

def is_point_in_polyhedron(point, vertices, faces):
    
    ray_direction = np.random.rand(3) - 0.5
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Normalize the ray direction
    intersections = 0

    for face in faces:
        for i in range(1, len(face) - 1):
            v0 = np.array(vertices[face[0]])
            v1 = np.array(vertices[face[i]])
            v2 = np.array(vertices[face[i + 1]])
            while judge_paralle(ray_direction, v0, v1, v2):
                ray_direction = np.random.rand(3) - 0.5
                ray_direction = ray_direction / np.linalg.norm(ray_direction)
                # find a new ray direction that is not parallel to any of the plane
    for face in faces:
        for i in range(1, len(face) - 1):
            v0 = np.array(vertices[face[0]])
            v1 = np.array(vertices[face[i]])
            v2 = np.array(vertices[face[i + 1]])
            intersect, t, b1, b2 = moller_trumbore(v0, v1, v2, np.array(point), ray_direction)
            if intersect:
                intersections += 1

    return intersections % 2 == 1, intersections

# (True/False, number of intersections) 
# For Points inside/outside the Polyhedron

# Example:
# point = [1, 1, 1]
# vertices = [
#     [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
#     [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]
# ]

# faces = [
#     [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
#     [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
# ]
# Faces are defined by the indices of the vertices

# print(is_point_in_polyhedron(point, vertices, faces))  



    """
    - True: in
    - False: out
    """
    
    # get the coordinates of the point and the base center
    px, py, pz = point
    cx, cy, cz = base_center

    # judge if the point is within the height of the cylinder
    if cz <= pz <= cz + height:
        # judge if the point is within the radius of the cylinder
        distance_to_axis = np.sqrt((px - cx)**2 + (py - cy)**2)
        if distance_to_axis <= radius:
            return True

    return False

# # example usage
# point = [1, 1, 3]  # Example point
# base_center = [0, 0, 0]  # Center of the base of the cylinder
# radius = 2  # Radius of the cylinder
# height = 5  # Height of the cylinder

# result = is_point_in_cylinder(point, base_center, radius, height)
# print(f"the Point {point} is in the Cylinder: {result}")

# void Vec4::bst(const Vec4& pIn) {

#   if (abs(pIn.tt) < Vec4::TINY) return;
#   double betaX = pIn.xx / pIn.tt;
#   double betaY = pIn.yy / pIn.tt;
#   double betaZ = pIn.zz / pIn.tt;
#   double beta2 = betaX*betaX + betaY*betaY + betaZ*betaZ;
#   if (beta2 >= 1.) return;
#   double gamma = 1. / sqrt(1. - beta2);
#   double prod1 = betaX * xx + betaY * yy + betaZ * zz;
#   double prod2 = gamma * (gamma * prod1 / (1. + gamma) + tt);
#   xx          += prod2 * betaX;
#   yy          += prod2 * betaY;
#   zz          += prod2 * betaZ;
#   tt           = gamma * (tt + prod1);

# }

def bst_SHiP(self, beta_z=-0.9977):
    """
    zboost to lab frame
    self: [x, y, z, t]
    """
    if abs(beta_z) >= 1.0:
        print("Warning: beta_z >= 1, boost not performed.")
        return self
    
    # 计算gamma因子
    gamma = 1.0 / np.sqrt(1.0 - beta_z * beta_z)
    
    # 保存原始值
    t_old = self[3]
    z_old = self[2]
    
    self[3] = gamma * (t_old + beta_z * z_old)
    self[2] = gamma * (z_old + beta_z * t_old)
    
    
    return [self[0], self[1], self[2]]


def SHiP(point):

    ray_direction = np.random.rand(3) - 0.5
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Normalize the ray direction
    intersections = 0

    for face in SHiPfaces:
        for i in range(1, len(face) - 1):
            v0 = np.array(SHiPvertices[face[0]])
            v1 = np.array(SHiPvertices[face[i]])
            v2 = np.array(SHiPvertices[face[i + 1]])
            while judge_paralle(ray_direction, v0, v1, v2):
                ray_direction = np.random.rand(3) - 0.5
                ray_direction = ray_direction / np.linalg.norm(ray_direction)
                # find a new ray direction that is not parallel to any of the plane
    # for face in SHiPfaces:
    #     for i in range(1, len(face) - 1):
            v0 = np.array(SHiPvertices[face[0]])
            v1 = np.array(SHiPvertices[face[i]])
            v2 = np.array(SHiPvertices[face[i + 1]])
            intersect, t, b1, b2 = moller_trumbore(v0, v1, v2, np.array(point), ray_direction)
            if intersect:
                intersections += 1

    return intersections % 2 == 1, intersections

def SHiP_unbstted(point0):
    point = bst_SHiP(point0)
    ray_direction = np.random.rand(3) - 0.5
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Normalize the ray direction
    intersections = 0

    for face in SHiPfaces:
        for i in range(1, len(face) - 1):
            v0 = np.array(SHiPvertices[face[0]])
            v1 = np.array(SHiPvertices[face[i]])
            v2 = np.array(SHiPvertices[face[i + 1]])
            while judge_paralle(ray_direction, v0, v1, v2):
                ray_direction = np.random.rand(3) - 0.5
                ray_direction = ray_direction / np.linalg.norm(ray_direction)
                # find a new ray direction that is not parallel to any of the plane
    # for face in SHiPfaces:
    #     for i in range(1, len(face) - 1):
            v0 = np.array(SHiPvertices[face[0]])
            v1 = np.array(SHiPvertices[face[i]])
            v2 = np.array(SHiPvertices[face[i + 1]])
            intersect, t, b1, b2 = moller_trumbore(v0, v1, v2, np.array(point), ray_direction)
            if intersect:
                intersections += 1

    return intersections % 2 == 1, intersections
# print(SHiP([0, 0, 4000, 40000]))  # True

