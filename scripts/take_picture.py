import cv2
import numpy as np
import time
import os

cap = cv2.VideoCapture(0)

video_frame = np.ones((500, 700), dtype=np.uint8)
last_taken_time = time.time()
save_path = "../calibration_images/3"
os.makedirs(save_path, exist_ok=True)
cnt = 0
started = False
printed = False
while 1:
    ret, frame = cap.read()
    if not printed:
        print("--> Frame size: ", frame.shape)
        printed = True
        
    if not ret:
        cv2.imshow("video", video_frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        continue
    
    video_frame = frame.copy()
    
    cv2.imshow("video", video_frame)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('k'):
        time.sleep(2)
        started = True
    
    if (time.time() - last_taken_time) >= 0.85 and started:
        file = os.path.join(save_path, str(cnt)+".png")
        cnt += 1
        last_taken_time = time.time()
        cv2.imwrite(file, video_frame)