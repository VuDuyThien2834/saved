import os

import shutil
import cv2
import numpy as np

from glob import glob

from c1_IDCardAlignment import IDCardAlignment

#---------------------------CONFIG-1---------------------------
# home = './test_data'
home = '/home/shared/FPT/projects/000_saved/src/4_feature/card_align/test_data'
#---------------------------END CONFIG-1---------------------------

#---------------------------CONFIG-2---------------------------
weight_path ='./m1_cbb_detection.pt' #Front and back
#---------------------------END CONFIG-2---------------------------

rd =IDCardAlignment(weight_path)

im_ps=glob(os.path.join(home,'*'))
reses=[]
_id=1
fail = 0

for im_p in im_ps:
    print(f'{_id}/{len(im_ps)}  :   {im_p}')
    
    im_bgr = cv2.imread(im_p)

    res=rd.align_card(im_bgr,device='cuda')

    cv2.imwrite(f'./test_results/result{_id}.jpg', res)
    _id+=1

    if type(res)!=list:
        continue
    fail += 1
    

print(f'success: {len(im_ps) - fail}')
print(f'fail: {fail}')


