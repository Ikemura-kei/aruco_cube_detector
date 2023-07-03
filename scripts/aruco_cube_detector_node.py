import cv2
import numpy as np
import rospy

dist = np.array([-0.464986, 0.153405, -0.008499, -0.001134, 0.000000])
cam_mat = np.array([[2217.98361,    0.     ,  530.32626],
            [0.     , 2201.45526,  429.69331],
            [0.     ,    0.     ,    1.     ]])

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

# errornous!!!
def get_rotation_matrix_from_rodrigues(axis, angle_rad):
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    
    u = axis[0]
    v = axis[1]
    w = axis[2]
    
    r11 = cos + u**2 * (1 - cos)
    r12 = u * v * (1 - cos) + w * sin
    r13 = u * w * (1 - cos) - v * sin
    r21 = u * v * (1 - cos) - w * sin
    r22 = cos + v**2 * (1 - cos)
    r23 = v * w * (1 - cos) - u * sin
    r31 = u * w * (1 - cos) + v * sin
    r32 = v * w * (1 - cos) - u * sin
    r33 = cos + w**2 * (1 - cos)
    
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def get_rotation_matrix_from_quaternion(w, i, j, k):
    r11 = 1 - 2 * (j**2 + k**2)
    r12 = 2 * (i * j - k * w)
    r13 = 2 * (i * k + j * w)
    r21 = 2 * (i * j + k * w)
    r22 = 1 - 2 * (i**2 + k**2)
    r23 = 2 * (j * k - i * w)
    r31 = 2 * (i * k - j * w)
    r32 = 2 * (j * k + i * w)
    r33 = 1 - 2 * (i**2 + j**2)
    
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def get_quaternion(axis, angle_rad):
    x = axis[0] * np.sin(angle_rad/2)
    y = axis[1] * np.sin(angle_rad/2)
    z = axis[2] * np.sin(angle_rad/2)
    w = np.cos(angle_rad/2)
    
    return w, x, y, z

def map_surface(source_id, target_id):
    if (source_id == 0):
        if (target_id == 0):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 1):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 2):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 3):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 4):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 5):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
    elif (source_id == 1):
        if (target_id == 0):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 1):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 2):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 3):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 4):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 5):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
    elif (source_id == 2):
        if (target_id == 0):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 1):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 2):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 3):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 4):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 5):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
    elif (source_id == 3):
        if (target_id == 0):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 1):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 2):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 3):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 4):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 5):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
    elif (source_id == 4):
        if (target_id == 0):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 1):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 2):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 3):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 4):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 5):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
    elif (source_id == 5):
        if (target_id == 0):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 1):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 2):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 3):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 4):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])
        elif (target_id == 5):
            return np.array([1, 0, 0]), np.pi / 2.0, np.array([[0], [-1], [1]])

def main():
    R_1 = euler_to_rotation_matrix(0, np.pi / 2.0, 0)

    cap = cv2.VideoCapture(2)
    ret, frame = cap.read()
    new_cam_mat = cam_mat.copy()
    
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    
    outer_size = 25
    half_outer_size = outer_size / 2
    size = 20
    half_size = size / 2.0
    obj_pnts = np.array([[-half_size, -half_size, 0], [half_size, -half_size, 0], [half_size, half_size, 0], [-half_size, half_size, 0]]).astype(np.float32)
    target_id = 0
    canvas = np.zeros((200, 200), np.uint8)
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        
        if not ret:
            cv2.imshow("markers", canvas)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            continue
        
        dst = frame.copy()
        dst = cv2.undistort(frame, cam_mat, dist, dst, new_cam_mat)
        
        (corners, ids, rejected) = cv2.aruco.detectMarkers(dst, arucoDict, parameters=arucoParams)
        canvas = dst.copy()
        source_id = ids[0][0] if ids is not None else -1
        
        if len(corners) <= 0 or source_id not in [0, 1, 2, 3, 4, 5]:
            cv2.imshow("markers", canvas)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            continue

        print("--> Source id: {}, Target id: {}".format(source_id, target_id))
        
        # -- estimate the pose of each aruco marker --
        R, R_new, rvec, tvec, rvec_fin = None, None, None, None, None
        for i in range(len(corners)):
            ok, rvec, tvec = cv2.solvePnP(obj_pnts, corners[i],cam_mat, dist, rvec, tvec)
            cv2.drawFrameAxes(canvas, cam_mat, dist, rvec, tvec, 5.5)
        
        cv2.aruco.drawDetectedMarkers(canvas, corners, ids)
        
        # -- setup a relative rotation between the observed face and the original face --
        axis, angle_rad, translation = map_surface(source_id, target_id)
        print(axis, " ", angle_rad, " ", translation)
        R_rot = get_rotation_matrix_from_rodrigues(axis, angle_rad)
        R_to_orig = get_rotation_matrix_from_quaternion(*get_quaternion(axis, angle_rad))
        # why don't they equal!!!!
        # print(R_rot - R_to_orig)
        
        # -- conversion to other plane, currently not working --
        R, _ = cv2.Rodrigues(rvec)
        R_new = np.matmul(R, R_to_orig.T)
        rvec_fin, _ = cv2.Rodrigues(R_new, rvec_fin)
        tvec_fin = np.matmul(R, np.array(translation * half_outer_size, tvec.dtype))
        cv2.drawFrameAxes(canvas, cam_mat, dist, rvec_fin, tvec + tvec_fin, 7.5)
        
        # -- verify your idea about rotation matrix and rodrigues vector here --
        # vec = np.matmul(R_to_orig.T, np.array([1, 0, 0]))
        # vec[vec <= 1e-10] = 0
        # print(vec)
        
        cv2.imshow("markers", canvas)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('c'):
            target_id = ids[0]

if __name__ == "__main__":
    main()