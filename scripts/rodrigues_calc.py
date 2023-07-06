import numpy as np
import cv2

def euler_to_rotation_matrix(z, y, x):
    """
    Convert a Z-Y-X Euler angle to a rotation matrix.
    """
    cz = np.cos(z)
    sz = np.sin(z)
    cy = np.cos(y)
    sy = np.sin(y)
    cx = np.cos(x)
    sx = np.sin(x)

    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])

    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])

    R = np.dot(Rz, np.dot(Ry, Rx))

    return R

def get_rodrigues(axis, angle_rad):
    return axis / np.tan(angle_rad / 2.0)

R = euler_to_rotation_matrix(np.pi / 2, 0, np.pi)
# r1 = get_rodrigues(np.array([0, 0, 1]), np.pi / 2)
# R1, _ = cv2.Rodrigues(r1)
# r2 = get_rodrigues(np.array([1, 0, 0]), np.pi)
# R2, _ = cv2.Rodrigues(r2)

# print(R2.shape, R1.shape)

# R = np.matmul(R2, R1)
print(R)
r, _ = cv2.Rodrigues(R)

print(r)