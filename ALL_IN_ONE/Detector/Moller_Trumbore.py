import numpy as np

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
