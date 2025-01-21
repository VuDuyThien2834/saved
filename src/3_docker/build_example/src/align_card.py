import os

import numpy as np
from ultralytics import YOLO
import cv2 as cv


class AlignCard:
    def __init__(self, front_cd_w_p, back_cd_w_p):
        self.front_cd_model = YOLO(front_cd_w_p, verbose=False)
        self.back_cd_model = YOLO(back_cd_w_p, verbose=False)

    def align_card(self, im_bgr, i_type, debug=False, im_p='', tf=True, device='cuda', device_type='0'):
        if device_type == '1':
            return im_bgr
        if i_type == 5:
            res = self.front_cd_model(im_bgr, imgsz=640, save=False, verbose=False, device=device)[0]
        else:
            res = self.back_cd_model(im_bgr, imgsz=640, save=False, verbose=False, device=device)[0]

        cps = [None] * 4  # [NOTE]: Corner points.

        if debug:
            _im_bgr = im_bgr.copy()

        for i in range(len(res.boxes.xywh)):
            bb = res.boxes.xywh[i]
            cls_id = int(res.boxes.cls.tolist()[i])
            conf = round(res.boxes.conf.tolist()[i], 4)
            cp = bb[:2]

            if debug:
                cc = tuple(np.array(cp.tolist()).astype('int'))
                _im_bgr = cv.circle(_im_bgr, cc, 5, (0, 0, 255), 3)

            if cls_id in [1, 2, 3,
                          4]:  # [NOTE]: Khi một corner class có nhiều bounding boxes, không thể biết nên chọn bounding box nào.
                if cps[cls_id - 1] != None:
                    # [TODO]: Optimize.
                    if debug:
                        im_name = os.path.basename(im_p)
                        o_im_p = os.path.join(self.fc_001_home, im_name)
                        cv.imwrite(o_im_p, _im_bgr)
                    return 1
                # [TODO]: Cuda?
                cps[cls_id - 1] = cp.cpu().tolist()

        for cp in cps:
            if cp == None:
                if debug:
                    im_name = os.path.basename(im_p)
                    o_im_p = os.path.join(self.fc_002_home, im_name)
                    cv.imwrite(o_im_p, _im_bgr)
                return 1

        if debug:
            im_name = os.path.basename(im_p)
            o_im_p = os.path.join(self.tc_home, im_name)
            cv.imwrite(o_im_p, _im_bgr)

        ID_CARD_W = 856
        ID_CARD_H = 540
        O_CPS = np.float32([[0, 0], [ID_CARD_W, 0], [ID_CARD_W, ID_CARD_H], [0, ID_CARD_H]])

        # Perform perspective transformation
        cps = np.float32(cps)
        tf_m = cv.getPerspectiveTransform(cps, O_CPS)
        tf_im = cv.warpPerspective(im_bgr, tf_m, (ID_CARD_W, ID_CARD_H))

        if debug:
            im_name = os.path.basename(im_p)
            o_im_p = os.path.join(self.aligned_home, im_name)
            cv.imwrite(o_im_p, tf_im)

        return tf_im
