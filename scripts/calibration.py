'''
This file allows performing intrinsic calibration for the left and right camera, as well as stereo calibration.
'''

from argparse import ArgumentParser
from ast import arg
import cv2
import os
import numpy as np

def load_image_paths(parent_folder_of_calib_images):
    """Get the paths to all calibration images, including the left and right camera images.

    Args:
        parent_folder_of_calib_images (str): the path to the folder containing the calibration images, there should be two subfolders named "left" and "right" containing the respective calibration images.
    
    Returns:
        list of str, list of str: the lists of paths to the left and right calibration images, respectively.
    """
    left_paths, right_paths = [], []
    l_path, r_path = os.path.join(parent_folder_of_calib_images, 'left'), os.path.join(parent_folder_of_calib_images, 'right')

    for idx, f in enumerate(os.listdir(l_path)):
        l_file = os.path.join(l_path, f)
        r_file = os.path.join(r_path, f)

        # -- check file existance on the right folder --
        if not os.path.exists(r_file):
            print("--> The image '%s' exists in the left folder but not in the right folder!" % (f))

        left_paths.append(l_file)
        right_paths.append(r_file)

    return left_paths, right_paths

def get_obj_pnts(row, col):
    """Get the object points of the corners of the calibration pattern (i.e. points in the world coordinate) for calibration.

    Args:
        row (int): the number of rows on the chessboard pattern
        col (int): the number of columns on the chessboard pattern

    Returns:
        numpy.ndarray: the object points, of shape (row*col, 3).
    """
    obj_pnts = np.zeros((row * col, 3), np.float32)

    x, y = 0, 0
    for i in range(col):
        x = i
        for j in range(row):
            y = j
            pnt = np.array([x, y, 0])
            obj_pnts[x * row + y] = pnt

    return obj_pnts

def get_img_pnts(img, row, col, winSize):
    """Get the image points from a calibration image.

    Args:
        img (numpy.ndarray): the input calibration image
        row (int): the number of rows on the chessboard pattern
        col (int): the number of columns on the chessboard pattern
        winSize (tuple of two int): the window size used for cv2.cornerSubPix() function
        
    Returns:
        refine_corner (numpy.ndarray): the refined corner points
        image_with_corners (numpy.ndarray): the caliration image with the detected corners drawn
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # we use grayscale image for finding corners

    # -- step 1, apply chessboard corner detection --
    ret, corners = cv2.findChessboardCorners(gray_img, (row, col), None)
    if ret is False:
        return None, img

    # -- step 2, apply refinement to the output from step 1 --
    # NOTE: (criteria type, max num iterations, epsilon)
    subpix_refine_term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.001)
    refine_corner = cv2.cornerSubPix(gray_img, corners, winSize, (-1, -1), subpix_refine_term_criteria) 

    # -- step 3, show the chessboard corners to user for sanity check --
    image_with_corners = np.copy(img)
    cv2.drawChessboardCorners(image_with_corners, (row, col), refine_corner, ret)
    cv2.putText(image_with_corners, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.52, (33,100,255), 1)

    return refine_corner, image_with_corners

def single_camera_calibration(row, col, img_folder, tile_size, verbose):
    """Calibrate a single camera's intrinsic.

    Args:
        row (int): the number of rows on the chessboard
        col (int): the number of columns on the chessboard
        img_folder (str): the path to the parent folder containing calibration images (specifically, the 'left' and 'right' subfolder)
        tile_size (int): the size of the tile on the chessboard pattern, specified in mm
        verbose (bool): True to show detected corners, False otherwise

    Returns:
        cmtx (numpy.ndarray): the 3x3 intrinsic matrix
        dist (numpy.ndarray): the 5-element distortion coefficient (k1, k2, p1, p2, k3)
        img_p (numpy.ndarray): the detected chessboard corners
        obj_p (numpy.ndarray): the object points (in the world coordinates) of the chessboard corners
    """
    img_size = cv2.imread(img_folder[0]).shape
    print("--> Calibration image size:", img_size)

    img_p, obj_p = [], []
    for idx, l_p in enumerate(img_folder):
        obj_pnts = get_obj_pnts(row, col) * tile_size
        l_im = cv2.imread(l_p)
        img_pnts, img_with_img_pnts = get_img_pnts(l_im, row, col, (5, 5))
        
        if verbose:
            cv2.imshow('corners', img_with_img_pnts)
            
            # -- press 's' to discard a sample --
            if cv2.waitKey(0) == ord('s'):
                print('--> Skipped image sample')
                continue
            
        if img_pnts is None:
            print("--> Find corner failed for this image")
            continue

        img_p.append(img_pnts)
        obj_p.append(obj_pnts)

    cv2.destroyAllWindows()
    print("--> Obj points shape:", obj_p[0].shape, "img points shape:", img_p[0].shape)
    ret, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_p, img_p, (img_size[1], img_size[0]), None, None)

    print("\n#####################################################################\nRMSE re-projection error: %.8f\n" % (ret))
    print("intrinsic camera matrix:\n", cmtx)
    print('#####################################################################\n')

    return cmtx, dist, img_p, obj_p

if __name__ == "__main__":
    parser = ArgumentParser(description="stereo calibration")

    parser.add_argument('--img_folder', type=str, required=True, help="the parent folder to the calibration images, this folder should contain two subfolders named 'left' and 'right' containing the respective calibration images.")
    parser.add_argument('-r', type=int, required=True, dest='row', help="the number of rows on the chessboard pattern.")
    parser.add_argument('-c', type=int, required=True, dest='col', help="the number of columns on the chessboard pattern.")
    parser.add_argument('--tile_size', type=float, required=True, help="the size of each tile on the chessboard patter, specified in mm.")
    parser.add_argument("--inspect", action="store_true", default=False, help="True to inspect the corner detecton picture by picture, False otherwise, that is, skip the inspection process")

    args = parser.parse_args()
    
    # -- get image paths --
    img_paths = []

    for idx, f in enumerate(os.listdir(args.img_folder)):
        file = os.path.join(args.img_folder, f)
        img_paths.append(file)
        
    # -- perform calibration --
    result_dict = single_camera_calibration(args.row, args.col, img_paths, args.tile_size, verbose=args.inspect)
    print(result_dict[1]) # distortion coeffs
    
    for im in img_paths:
        pic = cv2.imread(im, -1)
        undistorted = cv2.undistort(pic, result_dict[0], result_dict[1])
        cv2.imshow("orig", pic)
        cv2.imshow("undistorted", undistorted)
        k = cv2.waitKey(2500)
        if k == ord('q'):
            break
        
    