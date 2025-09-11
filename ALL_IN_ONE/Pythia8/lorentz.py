import math
import numpy as np

class Vec4:
    TINY = 1e-10
    
    def __init__(self, t=0.0, x=0.0, y=0.0, z=0.0):
        self.tt = t
        self.xx = x
        self.yy = y
        self.zz = z
    
    def boost_to_lab_frame(self, beta_x=0.0, beta_y=0.0, beta_z=0.9977):
        """
        通用boost函数，可以沿任意方向boost到lab系
        
        参数:
        beta_x: x方向的beta值
        beta_y: y方向的beta值
        beta_z: z方向的beta值 (默认0.9977)
        """
        beta2 = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z
        
        if beta2 >= 1.0:
            print("Warning: beta^2 >= 1, boost not performed.")
            return self
        
        gamma = 1.0 / math.sqrt(1.0 - beta2)
        prod = beta_x * self.xx + beta_y * self.yy + beta_z * self.zz
        factor = gamma * (gamma * prod / (1.0 + gamma) + self.tt)
        
        self.tt = gamma * (self.tt + prod)
        self.xx += factor * beta_x
        self.yy += factor * beta_y
        self.zz += factor * beta_z
        
        return self
    
    def boost_to_lab_frame_z(self, beta_z=0.9977):
        """便捷函数：专门沿z轴boost"""
        return self.boost_to_lab_frame(0.0, 0.0, beta_z)
    
    def __str__(self):
        return f"(t={self.tt:.6f}, x={self.xx:.6f}, y={self.yy:.6f}, z={self.zz:.6f})"
    
    def copy(self):
        return Vec4(self.tt, self.xx, self.yy, self.zz)

# 使用示例
if __name__ == "__main__":
    # 示例1: 沿z轴boost
    particle1 = Vec4(1.0, 0.5, 0.3, 2.0)
    print("原始位置:", particle1)
    
    particle1.boost_to_lab_frame_z(0.9977)
    print("沿z轴boost后:", particle1)
    
    # 示例2: 沿任意方向boost
    particle2 = Vec4(1.0, 0.5, 0.3, 2.0)
    particle2.boost_to_lab_frame(0.1, 0.2, 0.8)
    print("沿(0.1,0.2,0.8)方向boost后:", particle2)