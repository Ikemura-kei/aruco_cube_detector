import cv2
from cv2 import aruco
import os

save_dir = "../arucos"
num_mark = 7 # number of markers
size_mark = 500 # size of markers

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)

for count in range(num_mark) :
    id_mark = count
    img_mark = aruco.drawMarker(dict_aruco, id_mark, size_mark)

    if count < 10 :
        img_name_mark = 'mark_id_0' + str(count) + '.jpg'
    else :
        img_name_mark = 'mark_id_' + str(count) + '.jpg'
    path_mark = os.path.join(save_dir, img_name_mark)

    cv2.imwrite(path_mark, img_mark)