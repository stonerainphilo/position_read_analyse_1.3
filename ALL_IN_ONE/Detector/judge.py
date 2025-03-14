import numpy as np
from Moller_Trumbore import moller_trumbore

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