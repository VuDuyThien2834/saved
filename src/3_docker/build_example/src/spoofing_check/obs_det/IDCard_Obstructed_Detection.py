from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os


class IDCard_Obstructed_Detection:
    def __init__(self, od_w_p):  # corner detection and obstruction detection
        self.od_model = YOLO(od_w_p, verbose=False)

    def detect_obstruciton(self, transformed_card):
        res = self.od_model.predict(transformed_card, save=False, verbose=False, imgsz=640)
        if len(res[0].boxes.xywh) != 0:
            return 1, round(res[0].boxes.conf.tolist()[0], 4)
        else:
            return 0, 1.0

    def infer(self, im_bgr_aligned):

        tf_m = im_bgr_aligned

        if isinstance(tf_m, int) == True:
            return 0, 1.0
        else:
            return (self.detect_obstruciton(tf_m))

    def calculate_overlap(sefl, rect1, rect2):
        """
        Calculate the overlap percentage between two rectangles. (to use at infer fuction)

        Args:
        - rect1, rect2: Tuples representing rectangles in the format (x, y, width, height).

        Returns:
        - overlap_percent: Percentage of overlap between the two rectangles.
        """
        # Calculate the coordinates of the intersection rectangle
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
        y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])

        # Calculate area of intersection rectangle
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate area of both rectangles
        rect1_area = rect1[2] * rect1[3]
        rect2_area = rect2[2] * rect2[3]

        # Calculate overlap percentage
        overlap_percent = (intersection_area / min(rect1_area, rect2_area)) * 100

        return overlap_percent

    def infer_check_if_obstruct_fingerprint(self, im_bgr_aligned):

        tf_m = im_bgr_aligned

        if isinstance(tf_m, int):
            return [(0, 1.0), (0, 1.0)]
        else:
            res = self.od_model.predict(tf_m, save=False, verbose=False, imgsz=640)
            if len(res[0].boxes.xywh) != 0:
                boxes_xywh_pixel = res[
                    0].boxes.xywh.tolist()  # [[759.0170288085938, 125.66754913330078, 186.225830078125, 221.5996551513672] [....]]
                fp_001 = [457, 0, 198, 252]
                fp_002 = [658, 0, 198, 252]
                fps_obstruct_result = [(0, 1.0), (0, 1.0)]

                for box in boxes_xywh_pixel:
                    box_xywh = [box[0] - box[2] // 2, box[1] - box[3] // 2, box[2], box[3]]
                    fp_001_obstruct = self.calculate_overlap(box_xywh, fp_001)
                    fp_002_obstruct = self.calculate_overlap(box_xywh, fp_002)

                    if fp_001_obstruct != 0:
                        fps_obstruct_result[0] = (1, 1.0)
                    if fp_002_obstruct != 0:
                        fps_obstruct_result[1] = (1, 1.0)

                return fps_obstruct_result
            else:
                return [(0, 1.0), (0, 1.0)]
