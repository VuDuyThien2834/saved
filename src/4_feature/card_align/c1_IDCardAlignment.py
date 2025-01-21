import os

import numpy as np
from ultralytics import YOLO
import cv2 as cv


class IDCardAlignment:
    def __init__(self, cbb_w_p):
        self.cbb_detection_model = YOLO(cbb_w_p)

    def align_card(self, im_bgr, device='cuda'):
        
        res = self.cbb_detection_model(im_bgr, imgsz=640, save=False, verbose=False, device=device)[0]

        cps = [None] * 4  # [NOTE]: Corner points.

        for i in range(len(res.boxes.xywh)):
            bb = res.boxes.xywh[i]
            cls_id = int(res.boxes.cls.tolist()[i])
            cp = bb[:2]

            if cls_id in [1, 2, 3, 4]:  # [NOTE]: Khi một corner class có nhiều bounding boxes, không thể biết nên chọn bounding box nào.
                if cps[cls_id - 1] != None:
                    return 1
                cps[cls_id - 1] = cp.cpu().tolist()

        for cp in cps:
            if cp == None:
                return 1

        ID_CARD_W = 856
        ID_CARD_H = 540
        O_CPS = np.float32([[0, 0], [ID_CARD_W, 0], [ID_CARD_W, ID_CARD_H], [0, ID_CARD_H]])

        # Perform perspective transformation
        cps = np.float32(cps)
        tf_m = cv.getPerspectiveTransform(cps, O_CPS)
        tf_im = cv.warpPerspective(im_bgr, tf_m, (ID_CARD_W, ID_CARD_H))

        return tf_im
