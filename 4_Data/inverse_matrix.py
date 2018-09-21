import numpy as np

def myinv(mat):
    s = np.shape(mat)
    if s[0] != s[1]:
        print("Not a square matrix!")
        return None
    elif np.linalg.det(mat) == 0.0:
        print("Determinant is zero!")
        return None
    else:
        return np.linalg.inv(mat)

