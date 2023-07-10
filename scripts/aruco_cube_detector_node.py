import cv2
import numpy as np
import rospy
import os
from geometry_msgs.msg import PoseStamped
import struct
import serial
import crc16
import time
import subprocess

# struct Pose
# {
#     // Position
#     float x;
#     float y;
#     float z;
#     // Quaternion
#     float q;
#     float i;
#     float j;
#     float k;
# } __attribute__((packed));

# struct CVMessage
# {
#     Pose headPose;
#     Pose cubePose;
#     uint16_t crc;
# } __attribute__((packed));

def crc16(data: bytes):
    '''
    CRC-16 (CCITT) implemented with a precomputed lookup table
    '''
    table = [ 
    0x0000, 0x1189, 0x2312, 0x329b, 0x4624, 0x57ad, 0x6536, 0x74bf, 0x8c48, 0x9dc1, 0xaf5a, 0xbed3, 0xca6c, 0xdbe5, 0xe97e, 0xf8f7, 0x1081, 0x0108,
    0x3393, 0x221a, 0x56a5, 0x472c, 0x75b7, 0x643e, 0x9cc9, 0x8d40, 0xbfdb, 0xae52, 0xdaed, 0xcb64, 0xf9ff, 0xe876, 0x2102, 0x308b, 0x0210, 0x1399,
    0x6726, 0x76af, 0x4434, 0x55bd, 0xad4a, 0xbcc3, 0x8e58, 0x9fd1, 0xeb6e, 0xfae7, 0xc87c, 0xd9f5, 0x3183, 0x200a, 0x1291, 0x0318, 0x77a7, 0x662e,
    0x54b5, 0x453c, 0xbdcb, 0xac42, 0x9ed9, 0x8f50, 0xfbef, 0xea66, 0xd8fd, 0xc974, 0x4204, 0x538d, 0x6116, 0x709f, 0x0420, 0x15a9, 0x2732, 0x36bb,
    0xce4c, 0xdfc5, 0xed5e, 0xfcd7, 0x8868, 0x99e1, 0xab7a, 0xbaf3, 0x5285, 0x430c, 0x7197, 0x601e, 0x14a1, 0x0528, 0x37b3, 0x263a, 0xdecd, 0xcf44,
    0xfddf, 0xec56, 0x98e9, 0x8960, 0xbbfb, 0xaa72, 0x6306, 0x728f, 0x4014, 0x519d, 0x2522, 0x34ab, 0x0630, 0x17b9, 0xef4e, 0xfec7, 0xcc5c, 0xddd5,
    0xa96a, 0xb8e3, 0x8a78, 0x9bf1, 0x7387, 0x620e, 0x5095, 0x411c, 0x35a3, 0x242a, 0x16b1, 0x0738, 0xffcf, 0xee46, 0xdcdd, 0xcd54, 0xb9eb, 0xa862,
    0x9af9, 0x8b70, 0x8408, 0x9581, 0xa71a, 0xb693, 0xc22c, 0xd3a5, 0xe13e, 0xf0b7, 0x0840, 0x19c9, 0x2b52, 0x3adb, 0x4e64, 0x5fed, 0x6d76, 0x7cff,
    0x9489, 0x8500, 0xb79b, 0xa612, 0xd2ad, 0xc324, 0xf1bf, 0xe036, 0x18c1, 0x0948, 0x3bd3, 0x2a5a, 0x5ee5, 0x4f6c, 0x7df7, 0x6c7e, 0xa50a, 0xb483,
    0x8618, 0x9791, 0xe32e, 0xf2a7, 0xc03c, 0xd1b5, 0x2942, 0x38cb, 0x0a50, 0x1bd9, 0x6f66, 0x7eef, 0x4c74, 0x5dfd, 0xb58b, 0xa402, 0x9699, 0x8710,
    0xf3af, 0xe226, 0xd0bd, 0xc134, 0x39c3, 0x284a, 0x1ad1, 0x0b58, 0x7fe7, 0x6e6e, 0x5cf5, 0x4d7c, 0xc60c, 0xd785, 0xe51e, 0xf497, 0x8028, 0x91a1,
    0xa33a, 0xb2b3, 0x4a44, 0x5bcd, 0x6956, 0x78df, 0x0c60, 0x1de9, 0x2f72, 0x3efb, 0xd68d, 0xc704, 0xf59f, 0xe416, 0x90a9, 0x8120, 0xb3bb, 0xa232,
    0x5ac5, 0x4b4c, 0x79d7, 0x685e, 0x1ce1, 0x0d68, 0x3ff3, 0x2e7a, 0xe70e, 0xf687, 0xc41c, 0xd595, 0xa12a, 0xb0a3, 0x8238, 0x93b1, 0x6b46, 0x7acf,
    0x4854, 0x59dd, 0x2d62, 0x3ceb, 0x0e70, 0x1ff9, 0xf78f, 0xe606, 0xd49d, 0xc514, 0xb1ab, 0xa022, 0x92b9, 0x8330, 0x7bc7, 0x6a4e, 0x58d5, 0x495c,
    0x3de3, 0x2c6a, 0x1ef1, 0x0f78
    ]
    
    crc = 0x0000
    # print("\nstart")
    for byte in data:
        # print(hex(crc), hex(np.ushort(byte)))
        crc = (np.ushort(crc) >> 8) ^ table[(np.ushort(crc) ^ np.ushort(byte)) & 0x00ff]
        crc &= 0xffff      
    # print("end\n")                             
    return crc

def get_quaternion_from_rotation_matrix(R):
    """Get quaternion representation from a 3x3 rotation matrix representation

    Args:
        R (numpy.ndarray): A 3x3 rotation matrix

    Returns:
        float, float, float, float: quaternion elements as in w, i, j, k
    """
    m00 = R[0,0]
    m01 = R[0,1]
    m02 = R[0,2]
    m10 = R[1,0]
    m11 = R[1,1]
    m12 = R[1,2]
    m20 = R[2,0]
    m21 = R[2,1]
    m22 = R[2,2]
    
    tr = m00 + m11 + m22

    if (tr > 0):
        S = np.sqrt(tr+1.0) * 2 # S=4*qw 
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S 
        qz = (m10 - m01) / S 
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2 # S=4*qx 
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S 
        qz = (m02 + m20) / S 
    elif (m11 > m22):
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2 # S=4*qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S 
        qy = 0.25 * S
        qz = (m12 + m21) / S 
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2 # S=4*qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
        
    return qw, qx, qy, qz

def get_euler_zyx_from_rotation_matrix(z, y, x):
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
        numpy.ndarray, numpy.ndarray: the 3x3 rotation matrix as well as the 1x3 translation vector from source surface coordinates to target surface coordinates. Note that the translation is in the unit of cube side length. For example, a translation of 0.5 means half of the cube side length.
    """
    # Please don't get overwhelmed by these, they are derived per surface and they should well document itself.
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
            return get_euler_zyx_from_rotation_matrix(-np.pi/2, -np.pi/2, 0).T, np.array([[0], [-0.5], [0.5]]) # ok
        elif (target_id == 5):
            return get_euler_zyx_from_rotation_matrix(np.pi/2, -np.pi/2, 0).T, np.array([[0], [0.5], [0.5]]) # ok
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
            return get_euler_zyx_from_rotation_matrix(-np.pi, 0, np.pi/2).T, np.array([[0], [-0.5], [0.5]]) # ok
        elif (target_id == 5):
            return get_euler_zyx_from_rotation_matrix(-np.pi, 0, -np.pi/2).T, np.array([[0], [0.5], [0.5]]) # ok
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
            return get_euler_zyx_from_rotation_matrix(np.pi/2, np.pi/2, 0).T, np.array([[0], [-0.5], [0.5]])
        elif (target_id == 5):
            return get_euler_zyx_from_rotation_matrix(-np.pi/2, np.pi/2, 0).T, np.array([[0], [0.5], [0.5]])
    elif (source_id == 4):
        if (target_id == 0):
            return get_euler_zyx_from_rotation_matrix(0, 0, np.pi/2).T, np.array([[0], [0.5], [0.5]]) # ok
        elif (target_id == 1):
            return get_euler_zyx_from_rotation_matrix(np.pi/2, 0, np.pi/2).T, np.array([[-0.5], [0], [0.5]]) # ok
        elif (target_id == 2):
            return get_euler_zyx_from_rotation_matrix(np.pi, 0, np.pi/2).T, np.array([[0], [-0.5], [0.5]]) # ok
        elif (target_id == 3):
            return get_euler_zyx_from_rotation_matrix(-np.pi/2, 0, np.pi/2).T, np.array([[0.5], [0], [0.5]]) # ok
        elif (target_id == 4):
            return get_rotation_matrix_from_quaternion(*get_quaternion(np.array([1, 0, 0]), 0)), np.array([[0], [0], [0]]) # ok
        elif (target_id == 5):
            return get_euler_zyx_from_rotation_matrix(0, 0, np.pi).T, np.array([[0], [0], [1]]) # ok
    elif (source_id == 5):
        if (target_id == 0):
            return get_euler_zyx_from_rotation_matrix(0, 0, -np.pi/2).T, np.array([[0], [-0.5], [0.5]]) # ok
        elif (target_id == 1):
            return get_euler_zyx_from_rotation_matrix(-np.pi/2, 0, -np.pi/2).T, np.array([[-0.5], [0], [0.5]]) # ok
        elif (target_id == 2):
            return get_euler_zyx_from_rotation_matrix(np.pi, 0, -np.pi/2).T, np.array([[0], [0.5], [0.5]]) # ok
        elif (target_id == 3):
            return get_euler_zyx_from_rotation_matrix(np.pi/2, 0, -np.pi/2).T, np.array([[0.5], [0], [0.5]]) # ok
        elif (target_id == 4):
            return get_euler_zyx_from_rotation_matrix(0, 0, np.pi).T, np.array([[0], [0], [1]]) # ok
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

CAMERA_SERIAL_ID = "usb-xhci-hcd.0.auto-1.4"
def find_camera_index():
    cam_idx = -1
    result = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE) # get camera index by hardware identifier
    cmd_out = result.stdout.decode('UTF-8')
    substr_idx = cmd_out.find(CAMERA_SERIAL_ID)

    if substr_idx != -1:
        start_idx = substr_idx+len(CAMERA_SERIAL_ID)+len(":   ")
        camera_device = cmd_out[start_idx:(start_idx+len("/dev/videoX"))] # assumes the camera index is less than 10
        cam_idx = int(camera_device[-1])

    return cam_idx

def find_serial_device():
    device_name = ""
    sub_proc_result = subprocess.run(['ls', '/dev'], stdout=subprocess.PIPE)
    cmd_out = sub_proc_result.stdout.decode('UTF-8')
    substr_idx = cmd_out.find("ttyACM")

    if substr_idx != -1:
        start_idx = substr_idx
        device_name = cmd_out[start_idx:start_idx+len("ttyACMX")]

    return device_name
        
def main():
    # -- initialize ROS stuff --
    rospy.init_node("~aruco_cube_detector_node")
    print("--> Node initialized")
    aruco_pose_pub = rospy.Publisher("/aruco_cube_pose", PoseStamped)
    head_pose_pub = rospy.Publisher("/head_pose", PoseStamped)
    
    # -- open serial port --
    serial_device_name = find_serial_device()
    if serial_device_name == "":
        print("--> Serial device not found!!!")
        while serial_device_name == "" and not rospy.is_shutdown():
            print("--> Attempting to connect to serial device...")
            serial_device_name = find_serial_device()
            time.sleep(1)
        print("--> Serial device connected!!!")
            
    ser = serial.Serial('/dev/'+serial_device_name, baudrate=460800)  # open serial port
    print(ser.name)         # check which port was really used
    
    # -- initialize default cube pose --
    aruco_cube_pose = PoseStamped()
    aruco_cube_pose.header.seq = 0
    aruco_cube_pose.header.frame_id = "cube"
    aruco_cube_pose.header.stamp = rospy.Time.now()
    aruco_cube_pose.pose.position.x = aruco_cube_pose.pose.position.y = aruco_cube_pose.pose.position.z = 0
    aruco_cube_pose.pose.orientation.w = 1
    aruco_cube_pose.pose.orientation.x = aruco_cube_pose.pose.orientation.y = aruco_cube_pose.pose.orientation.z = 0
    
    # -- initialize default head pose --
    head_pose = PoseStamped()
    head_pose.header.seq = 0
    head_pose.header.frame_id = "cube"
    head_pose.header.stamp = rospy.Time.now()
    head_pose.pose.position.x = head_pose.pose.position.y = head_pose.pose.position.z = 0
    head_pose.pose.orientation.w = 1
    head_pose.pose.orientation.x = head_pose.pose.orientation.y = head_pose.pose.orientation.z = 0
    
    # -- get external parameters --
    config_file_path = rospy.get_param("~config_file", default="")
    if config_file_path == "" or ".yaml" not in config_file_path or not os.path.exists(config_file_path):
        print("--> Invalid configuration file path {}!".format(config_file_path))
        return
    print("--> Configuration file: {}".format(config_file_path))
    
    # -- read camera parameters --
    fs = cv2.FileStorage(config_file_path, cv2.FILE_STORAGE_READ)
    cam_mat_node = fs.getNode("camera_matrix")
    cam_mat = cam_mat_node.mat()
    dist_node = fs.getNode("distortion_coefficient")
    dist = dist_node.mat()
    
    print("--> Distortion coefficient\n{}".format(dist))
    print("--> Camera matrix\n{}".format(cam_mat))
    
    # -- initialize frame getter --
    # camera_id = find_camera_index()
    # print("--> Camera index {}".format(camera_id))
    # if camera_id < 0:
    #     print("--> Camera not found !!!!!")
    #     attempt_cnt = 0
    #     while camera_id < 0 and not rospy.is_shutdown():
    #         attempt_cnt += 1
    #         print("--> Trying to re-locate camera, attempt {}".format(attempt_cnt))
    #         camera_id = find_camera_index()
    #         time.sleep(1)
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)
    # ret, test = cap.read()
    # if not ret:
    #     print("--> Camera initialization failed!")
    #     attempt_cnt = 0
        
    #     while not ret and not rospy.is_shutdown():
    #         attempt_cnt += 1
    #         print("--> Trying to re-initialize camera, attempt {}".format(attempt_cnt))
    #         camera_id = find_camera_index()
    #         if camera_id >= 0:
    #             cap = cv2.VideoCapture(camera_id)
    #             ret, test = cap.read()
    #         time.sleep(1)
    
    # -- later used for undistort --
    new_cam_mat = cam_mat.copy()
    
    # -- initialize aruco detector --
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    
    # -- constant definitions --
    ARUCO_CUBE_OUTER_SIZE = 23.5
    ARUCO_CUBE_ARUCO_SIZE = 20
    HALF_ARUCO_CUBE_ARUCO_SIZE = ARUCO_CUBE_ARUCO_SIZE / 2.0
    HEAD_ARUCO_ID = 6
    HEAD_ARUCO_SIZE = 20
    HALF_HEAD_ARUCO_SIZE = HEAD_ARUCO_SIZE / 2.0
    CAMERA_RECONNECTION_PERIOD = 2.75 # seconds
    SERIAL_DEVICE_RECONNECTION_PERIOD = 1.5 # seconds
    
    # -- static variables --
    # object points for solvePnP 
    aruco_cube_aruco_obj_pnts = np.array([[-HALF_ARUCO_CUBE_ARUCO_SIZE, -HALF_ARUCO_CUBE_ARUCO_SIZE, 0], [HALF_ARUCO_CUBE_ARUCO_SIZE, -HALF_ARUCO_CUBE_ARUCO_SIZE, 0], [HALF_ARUCO_CUBE_ARUCO_SIZE, HALF_ARUCO_CUBE_ARUCO_SIZE, 0], [-HALF_ARUCO_CUBE_ARUCO_SIZE, HALF_ARUCO_CUBE_ARUCO_SIZE, 0]]).astype(np.float32)
    head_aruco_obj_pnts= np.array([[-HALF_HEAD_ARUCO_SIZE, -HALF_HEAD_ARUCO_SIZE, 0], [HALF_HEAD_ARUCO_SIZE, -HALF_HEAD_ARUCO_SIZE, 0], [HALF_HEAD_ARUCO_SIZE, HALF_HEAD_ARUCO_SIZE, 0], [-HALF_HEAD_ARUCO_SIZE, HALF_HEAD_ARUCO_SIZE, 0]]).astype(np.float32)
    head_aruco_index = -1
    target_id = 0
    debug_source_id = 5
    canvas = np.zeros((200, 200), np.uint8) # to hold visualization markers
    head_aruco_rvec, head_aruco_tvec, rvec_for_vis, tvec_fin, rvec_for_vis, tvec_for_vis, head_aruco_vis_tvec = None, None, None, None, None, None, None
    camera_disconnection_counter = 0
    last_camera_reconnection_attempt_time = time.time()
    serial_device_disconnection_counter = 0
    last_serial_device_reconnection_attempt_time = time.time()
    
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        rate.sleep()
        
        # -- publish data to ros topic for logging --
        aruco_cube_pose.header.stamp = rospy.Time.now()
        aruco_cube_pose.header.seq += 1
        aruco_pose_pub.publish(aruco_cube_pose)
        head_pose.header.stamp = rospy.Time.now()
        head_pose.header.seq += 1
        head_pose_pub.publish(head_pose)
        
        # -- send information --
        pose_data = struct.pack('ffffffffffffff', head_pose.pose.position.x, head_pose.pose.position.y, head_pose.pose.position.z \
            , head_pose.pose.orientation.w, head_pose.pose.orientation.x, head_pose.pose.orientation.y, head_pose.pose.orientation.z \
                , aruco_cube_pose.pose.position.x, aruco_cube_pose.pose.position.y, aruco_cube_pose.pose.position.z \
            , aruco_cube_pose.pose.orientation.w, aruco_cube_pose.pose.orientation.x, aruco_cube_pose.pose.orientation.y, aruco_cube_pose.pose.orientation.z)
        crc = crc16(pose_data)
        data = struct.pack('ffffffffffffffH', head_pose.pose.position.x, head_pose.pose.position.y, head_pose.pose.position.z \
            , head_pose.pose.orientation.w, head_pose.pose.orientation.x, head_pose.pose.orientation.y, head_pose.pose.orientation.z \
                , aruco_cube_pose.pose.position.x, aruco_cube_pose.pose.position.y, aruco_cube_pose.pose.position.z \
            , aruco_cube_pose.pose.orientation.w, aruco_cube_pose.pose.orientation.x, aruco_cube_pose.pose.orientation.y, aruco_cube_pose.pose.orientation.z \
                , crc)
        
        # -- send serial data, in line with re-connection attempt if disconnected --
        try:
            ser.write(data)
            if serial_device_disconnection_counter > 0:
                print("--> Serial device reconnected!!")
                serial_device_disconnection_counter = 0 # device connected
        except serial.SerialException as e:
            if (time.time() - last_serial_device_reconnection_attempt_time) > SERIAL_DEVICE_RECONNECTION_PERIOD:
                last_serial_device_reconnection_attempt_time = time.time()
                print("--> Serial device disconnected, trying to re-initialize serial device...")
                serial_device_name = find_serial_device()
                if serial_device_name != "":
                    ser = serial.Serial('/dev/'+serial_device_name, baudrate=460800)  # open serial port
            serial_device_disconnection_counter += 1
        
        # -- read frame in --
        ret, frame = cap.read()
        
        if not ret:
            if rvec_for_vis is not None and tvec_fin is not None and rvec_for_vis is not None and tvec_for_vis is not None:
                cv2.drawFrameAxes(canvas, cam_mat, dist, rvec_for_vis, tvec_fin, 1.5) # visualization of the inferred coordinates
                cv2.drawFrameAxes(canvas, cam_mat, dist, rvec_for_vis, tvec_for_vis, 3.5) # visualization of the orientation of the cube only, at the bottom-left
            if head_aruco_rvec is not None and head_aruco_vis_tvec is not None:
                cv2.drawFrameAxes(canvas, cam_mat, dist, head_aruco_rvec, head_aruco_vis_tvec, 2.5) # visualization of the orientation of the head only, at the top-left
            
            if (time.time() - last_camera_reconnection_attempt_time) > CAMERA_RECONNECTION_PERIOD:
                last_camera_reconnection_attempt_time = time.time()
                print("--> Camera disconnected, trying to re-initialize camera...")
                cap.release()
                camera_id = find_camera_index()
                cap = cv2.VideoCapture(camera_id)
            
            camera_disconnection_counter += 1
            
            cv2.imshow("markers", canvas)
            cv2.imshow("raw", np.zeros_like(canvas))
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            continue
        
        if camera_disconnection_counter > 0:
            print("--> Camera reconnected!!")
            camera_disconnection_counter = 0 # frame received, connection is normal
        
        # -- undistortion --
        dst = frame.copy()
        dst = cv2.undistort(frame, cam_mat, dist, dst, new_cam_mat)
        
        # -- pre-processing --
        dst = image_preproc(dst, mode=1)
        
        # -- detect aruco marker --
        (corners, ids, rejected) = cv2.aruco.detectMarkers(dst, arucoDict, parameters=arucoParams)
        canvas = dst.copy()
        source_id = -1 # initialize as -1
        head_aruco_index = -1 # initialize as -1
        for i in range(len(corners)):
            if ids[i] == HEAD_ARUCO_ID:
                head_aruco_index = i
            else:
                source_id = ids[i][0] # use the first found aruco as the source
        
        # TODO: maybe multi-processing at here is a required optimization (depends on the actual performance)
        # -- do aruco cube processing --
        if len(corners) >= 0 and source_id in [0, 1, 2, 3, 4, 5]:
            # print("--> Source id: {}, Target id: {}".format(source_id, target_id))
            # -- estimate the pose of each aruco marker --
            R, R_new, rvec, tvec, rvec_fin = None, None, None, None, None
            for i in range(len(corners)):
                if ids[i][0] == source_id:
                    source_index = i
            ok, rvec, tvec = cv2.solvePnP(aruco_cube_aruco_obj_pnts, corners[source_index],cam_mat, dist, rvec, tvec)

            # -- draw aruco marker visualization --
            # cv2.aruco.drawDetectedMarkers(canvas, corners, ids)
            
            # -- setup a relative rotation and translation between the observed face and the original face --
            R_to_source_surface, translation = get_surface_transform(source_id, target_id)
            
            # -- conversion to other plane --
            R, _ = cv2.Rodrigues(rvec) # get rotation matrix
            R_new = np.matmul(R, R_to_source_surface.T) # apply relative rotation
            rvec_fin, _ = cv2.Rodrigues(R_new, rvec_fin) # get back Rodrigues (since OpenCV prefers working with it)
            tvec_rel = np.matmul(R, np.array(translation * ARUCO_CUBE_OUTER_SIZE, tvec.dtype)) # apply relative translation
            to_cube_center_translation = np.matmul(R_new, np.array([0, 0, ARUCO_CUBE_OUTER_SIZE*0.5], tvec.dtype))[...,None]
            tvec_fin = tvec + tvec_rel + to_cube_center_translation
            
            tvec_for_vis = np.array([[-20.5], [-2.4], [100]], tvec.dtype)
            rvec_for_vis = rvec_fin.copy()
            
            # -- get pose (w, i, j, k, x, y, z) --
            w, i, j, k = get_quaternion_from_rotation_matrix(R_new)
            x = tvec_fin[0][0]
            y = tvec_fin[1][0]
            z = tvec_fin[2][0]
            # print("Command is x:{}, y:{}, z:{}, w:{}, i:{}, j:{}, k:{}, quaternion norm:{:.3f}".format(x, y, z, w, i, j, k, np.linalg.norm(np.array([w,i,j,k]))))
            
            # -- update aruco cube pose --
            aruco_cube_pose.pose.position.x = x
            aruco_cube_pose.pose.position.y = y
            aruco_cube_pose.pose.position.z = z
            aruco_cube_pose.pose.orientation.w = w
            aruco_cube_pose.pose.orientation.x = i
            aruco_cube_pose.pose.orientation.y = j
            aruco_cube_pose.pose.orientation.z = k
        # -- do head aruco processing --
        if head_aruco_index >= 0:
            ok, head_aruco_rvec, head_aruco_tvec = cv2.solvePnP(head_aruco_obj_pnts, corners[head_aruco_index],cam_mat, dist, head_aruco_rvec, head_aruco_tvec)
            
            head_R, _ = cv2.Rodrigues(head_aruco_rvec)
            head_w, head_i, head_j, head_k = get_quaternion_from_rotation_matrix(head_R)
            head_x =head_aruco_tvec[0][0]
            head_y =head_aruco_tvec[1][0]
            head_z =head_aruco_tvec[2][0]
            
            # -- update head pose --
            head_pose.pose.position.x = head_x
            head_pose.pose.position.y = head_y
            head_pose.pose.position.z = head_z
            head_pose.pose.orientation.w = head_w
            head_pose.pose.orientation.x = head_i
            head_pose.pose.orientation.y = head_j
            head_pose.pose.orientation.z = head_k
            
            head_aruco_vis_tvec = np.array([[-20.5], [-16.5], [100]], head_aruco_tvec.dtype)
            
        # -- visualization --
        if rvec_for_vis is not None and tvec_fin is not None and rvec_for_vis is not None and tvec_for_vis is not None:
            cv2.drawFrameAxes(canvas, cam_mat, dist, rvec_for_vis, tvec_fin, 1.5) # visualization of the inferred coordinates
            cv2.drawFrameAxes(canvas, cam_mat, dist, rvec_for_vis, tvec_for_vis, 3.5) # visualization of the orientation of the cube only, at the bottom-left
        if head_aruco_rvec is not None and head_aruco_vis_tvec is not None:
            cv2.drawFrameAxes(canvas, cam_mat, dist, head_aruco_rvec, head_aruco_vis_tvec, 2.5) # visualization of the orientation of the head only, at the top-left
        cv2.imshow("markers", canvas)
        cv2.imshow("raw", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('c'):
            target_id = (target_id + 1) % 6
        elif k == ord('p'):
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
