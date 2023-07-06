import cv2
import numpy as np
import rospy

dist = np.array([-0.464986, 0.153405, -0.008499, -0.001134, 0.000000])
cam_mat = np.array([[2217.98361,    0.     ,  530.32626],
            [0.     , 2201.45526,  429.69331],
            [0.     ,    0.     ,    1.     ]])

def euler_to_rotation_matrix(z, y, x):
    """Converts an euler angle (Z-Y-X convention) into an equivalent rotation matrix

    Args:
        z (float): yaw angle in radians
        y (float): pitch angle in radians
        x (float): roll angle in radians

    Returns:
        numpy.ndarray: the 3x3 rotation matrix
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
    """Get a Rodrigues vector from axis-angle pair

    Args:
        axis (numpy.ndarray): a 1X3 or 3X1 vector marking the rotational axis
        angle_rad (float): the angle to rotate around the axis in radians

    Returns:
        numpy.ndarray: the 1x3 or 3X1 (depending on the shape of input) Rodrigues vector
    """
    return axis / np.tan(angle_rad / 2.0)

def get_rotation_matrix_from_quaternion(w, i, j, k):
    """Computes a 3x3 rotation matrix from an unit quaternion

    Args:
        w (float): the real part of the quaternion
        i (float): the x-axis value
        j (float): the y-axis value
        k (float): the z-axis value

    Returns:
        numpy.ndarray: the 3x3 rotation matrix converted from the quaternion
    """
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
    """Get a quaternion from axis-angle pair

    Args:
        axis (numpy.ndarray): a 1x3 array indicating the rotation axis
        angle_rad (float): the angle to turn around the axis in radian

    Returns:
        float, float, float, float: the quaternion elements, in the order of w, i, j, k
    """
    x = axis[0] * np.sin(angle_rad/2)
    y = axis[1] * np.sin(angle_rad/2)
    z = axis[2] * np.sin(angle_rad/2)
    w = np.cos(angle_rad/2)
    
    return w, x, y, z

def get_surface_transform(source_id, target_id):
    """Returns the relative transformation between the surface of the source id and the surface of the target id

    Args:
        source_id (int): the index of the source surface, between 0-5
        target_id (int): the index of the target surface, between 0-5

    Returns:
        numpy.ndarray, numpy.ndarray: the 3x3 rotation matrix as well as the 1x3 translation vector from source surface coordinates to target surface coordinates
    """
    if (source_id == 0):
        if (target_id == 0):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([1, 0, 0]), 0)), np.array([[0], [0], [0]]) # ok
        elif (target_id == 1):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), -np.pi / 2.0)), np.array([[-0.5], [0], [0.5]]) # ok
        elif (target_id == 2):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), -np.pi)), np.array([[0], [0], [1]]) # ok
        elif (target_id == 3):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), np.pi / 2.0)), np.array([[0.5], [0], [0.5]]) # ok
        elif (target_id == 4):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([1, 0, 0]), np.pi / 2.0)), np.array([[0], [-0.5], [0.5]]) # ok
        elif (target_id == 5):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([1, 0, 0]), -np.pi / 2.0)), np.array([[0], [0.5], [0.5]]) # ok
    elif (source_id == 1):
        if (target_id == 0):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), np.pi / 2.0)), np.array([[0.5], [0], [0.5]]) # ok
        elif (target_id == 1):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([1, 0, 0]), 0)), np.array([[0], [0], [0]]) # ok
        elif (target_id == 2):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), -np.pi / 2.0)), np.array([[-0.5], [0], [0.5]]) # ok
        elif (target_id == 3):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), -np.pi)), np.array([[0], [0], [1]]) # ok
        elif (target_id == 4):
            return euler_to_rotation_matrix(-np.pi/2, -np.pi/2, 0).T, np.array([[0], [-0.5], [0.5]]) # ok
        elif (target_id == 5):
            return euler_to_rotation_matrix(np.pi/2, -np.pi/2, 0).T, np.array([[0], [0.5], [0.5]]) # ok
    elif (source_id == 2):
        if (target_id == 0):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), -np.pi)), np.array([[0], [0], [1]]) # ok
        elif (target_id == 1):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), np.pi / 2.0)), np.array([[0.5], [0], [0.5]]) # ok
        elif (target_id == 2):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([1, 0, 0]), 0)), np.array([[0], [0], [0]]) # ok
        elif (target_id == 3):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), -np.pi / 2.0)), np.array([[-0.5], [0], [0.5]]) # ok
        elif (target_id == 4):
            return euler_to_rotation_matrix(-np.pi, 0, np.pi/2).T, np.array([[0], [-0.5], [0.5]]) # ok
        elif (target_id == 5):
            return euler_to_rotation_matrix(-np.pi, 0, -np.pi/2).T, np.array([[0], [0.5], [0.5]]) # ok
    elif (source_id == 3):
        if (target_id == 0):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), -np.pi / 2.0)), np.array([[-0.5], [0], [0.5]]) # ok
        elif (target_id == 1):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), -np.pi)), np.array([[0], [0], [1]]) # ok
        elif (target_id == 2):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([0, 1, 0]), np.pi / 2.0)), np.array([[0.5], [0], [0.5]]) # ok
        elif (target_id == 3):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([1, 0, 0]), 0)), np.array([[0], [0], [0]]) # ok
        elif (target_id == 4):
            return euler_to_rotation_matrix(np.pi/2, np.pi/2, 0).T, np.array([[0], [-0.5], [0.5]])
        elif (target_id == 5):
            return euler_to_rotation_matrix(-np.pi/2, np.pi/2, 0).T, np.array([[0], [0.5], [0.5]])
    elif (source_id == 4):
        if (target_id == 0):
            return euler_to_rotation_matrix(0, 0, np.pi/2).T, np.array([[0], [0.5], [0.5]]) # ok
        elif (target_id == 1):
            return euler_to_rotation_matrix(np.pi/2, 0, np.pi/2).T, np.array([[-0.5], [0], [0.5]]) # ok
        elif (target_id == 2):
            return euler_to_rotation_matrix(np.pi, 0, np.pi/2).T, np.array([[0], [-0.5], [0.5]]) # ok
        elif (target_id == 3):
            return euler_to_rotation_matrix(-np.pi/2, 0, np.pi/2).T, np.array([[0.5], [0], [0.5]]) # ok
        elif (target_id == 4):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([1, 0, 0]), 0)), np.array([[0], [0], [0]]) # ok
        elif (target_id == 5):
            return euler_to_rotation_matrix(0, 0, np.pi).T, np.array([[0], [0], [1]]) # ok
    elif (source_id == 5):
        if (target_id == 0):
            return euler_to_rotation_matrix(0, 0, -np.pi/2).T, np.array([[0], [-0.5], [0.5]]) # ok
        elif (target_id == 1):
            return euler_to_rotation_matrix(-np.pi/2, 0, -np.pi/2).T, np.array([[-0.5], [0], [0.5]]) # ok
        elif (target_id == 2):
            return euler_to_rotation_matrix(np.pi, 0, -np.pi/2).T, np.array([[0], [0.5], [0.5]]) # ok
        elif (target_id == 3):
            return euler_to_rotation_matrix(np.pi/2, 0, -np.pi/2).T, np.array([[0.5], [0], [0.5]]) # ok
        elif (target_id == 4):
            return euler_to_rotation_matrix(0, 0, np.pi).T, np.array([[0], [0], [1]]) # ok
        elif (target_id == 5):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([1, 0, 0]), 0)), np.array([[0], [0], [0]]) # ok

def image_preproc(original_frame, mode=0):
    """Image preprocessing utility

    Args:
        original_frame (numpy.ndarray): the input image
        mode (int, optional): the mode of pre-processing. Defaults to 0.

    Returns:
        numpy.ndarray: the pre-processed image
    """
    ret = original_frame.copy()
    
    if mode == 0:
        lab= cv2.cvtColor(original_frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        # -- applying CLAHE to L-channel --
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        # -- merge the CLAHE enhanced L-channel with the a and b channel --
        limg = cv2.merge((cl,a,b))
        # -- converting image from LAB Color model to BGR color spcae --
        ret = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    elif mode == 1:
        ret = cv2.GaussianBlur(original_frame, (15, 13), 0)
    elif mode == 2:
        THRESHOLD = 185
        mask = np.where(((ret[...,0]>THRESHOLD) * (ret[...,1]>THRESHOLD) * (ret[...,2]>THRESHOLD)), 255, 0).astype(np.uint8)
        ret[np.tile(mask[...,None], (1,1,3))>0] = 255
    
    return ret

def main():
    # -- initialize frame getter --
    cap = cv2.VideoCapture(0)
    ret, test = cap.read()
    if not ret:
        print("--> Camera initialization failed!")
        return
    
    # -- later used for undistort --
    new_cam_mat = cam_mat.copy()
    
    # -- initialize aruco detector --
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    
    # -- constant definitions --
    OUTER_SIZE = 23.5
    HALF_OUTER_SIZE = OUTER_SIZE / 2.0
    ARUCO_SIZE = 20
    HALF_ARUCO_SIZE = ARUCO_SIZE / 2.0
    
    # -- object points for solvePnP --
    obj_pnts = np.array([[-HALF_ARUCO_SIZE, -HALF_ARUCO_SIZE, 0], [HALF_ARUCO_SIZE, -HALF_ARUCO_SIZE, 0], [HALF_ARUCO_SIZE, HALF_ARUCO_SIZE, 0], [-HALF_ARUCO_SIZE, HALF_ARUCO_SIZE, 0]]).astype(np.float32)
   
    target_id = 0
    debug_source_id = 5
    rec_tvec = None
    rec_R = None
   
    canvas = np.zeros((200, 200), np.uint8) # to hold visualization markers
    
    while not rospy.is_shutdown():
        # -- read frame in --
        ret, frame = cap.read()
        
        if not ret:
            cv2.imshow("markers", canvas)
            cv2.imshow("raw", frame)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            continue
        
        # -- indistortion --
        dst = frame.copy()
        dst = cv2.undistort(frame, cam_mat, dist, dst, new_cam_mat)
        
        # -- pre-processing --
        dst = image_preproc(dst, mode=1)
        
        # -- detect aruco marker --
        (corners, ids, rejected) = cv2.aruco.detectMarkers(dst, arucoDict, parameters=arucoParams)
        canvas = dst.copy()
        source_id = ids[0][0] if ids is not None else -1
        
        # -- check if the detection results are valid --
        if len(corners) <= 0 or source_id not in [0, 1, 2, 3, 4, 5]:
            cv2.imshow("markers", canvas)
            cv2.imshow("raw", frame)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            continue

        # print("--> Source id: {}, Target id: {}".format(source_id, target_id))
        # -- estimate the pose of each aruco marker --
        R, R_new, rvec, tvec, rvec_fin = None, None, None, None, None
        for i in range(len(corners)):
            if ids[i][0] == source_id:
                source_index = i
        ok, rvec, tvec = cv2.solvePnP(obj_pnts, corners[source_index],cam_mat, dist, rvec, tvec)

        # -- draw aruco marker visualization --
        # cv2.aruco.drawDetectedMarkers(canvas, corners, ids)
        
        # -- setup a relative rotation and translation between the observed face and the original face --
        R_to_source_surface, translation = get_surface_transform(source_id, target_id)
        
        # -- conversion to other plane --
        R, _ = cv2.Rodrigues(rvec) # get rotation matrix
        R_new = np.matmul(R, R_to_source_surface.T) # apply relative rotation
        rvec_fin, _ = cv2.Rodrigues(R_new, rvec_fin) # get back Rodrigues (since OpenCV prefers working with it)
        tvec_fin = np.matmul(R, np.array(translation * OUTER_SIZE, tvec.dtype)) # apply relative translation
        cv2.drawFrameAxes(canvas, cam_mat, dist, rvec_fin, tvec + tvec_fin, 7.5) # visualization of the inferred coordinates
        
        # -- get command (w, i, j, k, x, y, z) --
        if rec_tvec is not None and rec_R is not None:
            translation = (tvec + tvec_fin) - rec_tvec
            x = translation[0]
            y = translation[1]
            z = translation[2]
            w = 1
            i, j, k = 0, 0, 0
            print("Command is x:{}, y:{}, z:{}, w:{}, i:{}, j:{}, k:{}".format(x, y, z, w, i, j, k))
        else:
            print("Command not started")
        
        # -- visualization --
        cv2.imshow("markers", canvas)
        cv2.imshow("raw", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('c'):
            target_id = (target_id + 1) % 6
        elif k == ord('p'):
            rec_tvec = tvec
            rec_R = R
            target_id = source_id

if __name__ == "__main__":
    main()
    
# errornous!!!
# def get_rotation_matrix_from_rodrigues(axis, angle_rad):
#     cos = np.cos(angle_rad)
#     sin = np.sin(angle_rad)
    
#     u = axis[0]
#     v = axis[1]
#     w = axis[2]
    
#     r11 = cos + u**2 * (1 - cos)
#     r12 = u * v * (1 - cos) + w * sin
#     r13 = u * w * (1 - cos) - v * sin
#     r21 = u * v * (1 - cos) - w * sin
#     r22 = cos + v**2 * (1 - cos)
#     r23 = v * w * (1 - cos) - u * sin
#     r31 = u * w * (1 - cos) + v * sin
#     r32 = v * w * (1 - cos) - u * sin
#     r33 = cos + w**2 * (1 - cos)
    
#     return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
