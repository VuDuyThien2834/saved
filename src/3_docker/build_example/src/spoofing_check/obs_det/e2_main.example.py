import os

import shutil
import cv2
import numpy as np

from glob import glob

from IDCard_Obstructed_Detection import IDCard_Obstructed_Detection

#---------------------------CONFIG-1---------------------------
HOME = '/home/shared/FPT/projects/z_06_SHB/data/raw/testcase'
# HOME = '/home/shared/FPT/projects/z_06_SHB/data/raw/card/test_raw'
# HOME = '/home/shared/FPT/projects/z_06_SHB/data/raw/card/other'
#---------------------------END CONFIG-1---------------------------

#---------------------------CONFIG-2---------------------------
front_cd_w_p = '/home/shared/FPT/projects/recaptured_detection/models/yolov8/src/runs/detect/train/weights/best.pt'     # Front corners detection weight
back_cd_w_p = '/home/shared/FPT/projects/recaptured_detection/models/yolov8/src/runs/detect/train9/weights/best.pt'     # Back corners detection weight
od_w_p = '/home/shared/FPT/projects/z_06_SHB/src/models/runs/detect/train/weights/best.pt'                              # Obstruction object detection weight
#---------------------------END CONFIG-2---------------------------


# TODO: phan loai giay to truoc va sau truoc khi khoi tao doi tuong
ofc=IDCard_Obstructed_Detection(front_cd_w_p, back_cd_w_p, od_w_p) 


im_ps=glob(os.path.join(HOME,'*'))
reses=[]
_id=1

_0_count, _1_count = 0, 0
thrs_list = []

b,g,r = [],[],[]
i_type = 4

for im_p in im_ps:
    # print(f'{_id}/{len(im_ps)}  :   {im_p}')
    _id+=1

    #------------- INFER -------------------
    im_bgr = cv2.imread(im_p) # Input image is original-full frame
    res, conf=ofc.infer(im_bgr, i_type, device='cuda')
    #------------- END INFER -------------------



    # ------------ RESULT REPORT------------
    # print(res)
    if res!=0: 
        _1_count += 1
        # print(res)
    else: _0_count +=1
print(f" 0 : {_0_count}, 1 : {_1_count}")
    # ------------ END RESULT REPORT------------