from ultralytics import YOLO

import numpy as np
import cv2 as cv

import os
import shutil


class DeviceDetection:

    def __init__(self):
        self.model = YOLO('./models/idcard_models/spoofing_check/yolov8x.pt')

    def predict_with_COCO_model(sefl, img):
        """
        Detect devices on image using COCO model (to use at detect_device() fuction)
        
        Args:
        - image: openCV format of photo (using cv2.imread func )
        
        Returns:
        - detection result 
        """
        # model = YOLO('yolov8x.pt')

        # Inference COCO model
        #  62: 'tv',
        #  63: 'laptop',
        #  67: 'cell phone'
        return (sefl.model(img, save=False, imgsz=640, classes=[62, 63, 67], verbose=False, conf=0.6, device='cuda'))

    #TODO: Find a way to use both model (currently I am only using the COCO model)
    def predict_with_OIV7_model(sefl, img):
        """
        Detect devices on image using OIV7 model (to use at detect_device() fuction)
        
        Args:
        - image: openCV format of photo (using cv2.imread func )
        
        Returns:
        - detection result 
        """
        model = YOLO('yolov8x-oiv7.pt')

        # Inference OIV7 model
        #   128: Computer monitor
        #   339: Mobile phone
        #   526: Telephone
        #   527: Television
        #   516: Tablet computer
        #   304: Laptop
        return (model(img, save=False, imgsz=640, classes=[128, 339, 526, 527, 516, 304], verbose=False))

    def crop_image(sefl, image, x, y, crop_rate):
        """
        Crop an image based on specified coordinates. (to use at detect_device() fuction)
        
        Args:
        - image_path: Path to the input image.
        - x, y: Coordinates of the center of the region to be cropped.
        - crop_rate: to calculate width and height of the region to be cropped. (VD: 0.75, 0.8)
        
        Returns:
        - cropped_image: Cropped image.
        """
        # Read the image and prepare parameter for cropping
        org_height, org_width, _ = image.shape
        crop_height = round(org_height * crop_rate)
        crop_width = round(org_width * crop_rate)

        # Crop the region of interest
        cropped_image = image[round(y - crop_height / 2):round(y + crop_height / 2),
                        round(x - crop_width / 2):round(x + crop_width / 2)]
        return cropped_image

    def detect_device(self, image, cropTime_threshold, crop_rate):
        """
        Detect devices on image (to use at calculate_overlap() fuction)
        
        Args:
        - image: openCV format of photo (using cv2.imread func )
        
        Returns:
        - result_COCO[0].boxes: boxes object of devices detection result, atleast contain xywh information of detected devices
        """

        if image.shape[0] > 640 and image.shape[
            1] > 640:  #check if any image's edge is smaller than 640 (according to yolov8 resized shape)
            result_COCO = self.predict_with_COCO_model(image)

            if len(result_COCO[0].boxes.cls.tolist()) == 0:
                image = self.crop_image(image, int(image.shape[1] / 2), int(image.shape[0] / 2), crop_rate)
                return (self.detect_device(image, cropTime_threshold, crop_rate))

            return result_COCO
        else:
            return None

    def define_card_position(self, img):
        height, width, _ = img.shape
        if height == 1920:  # height = 1920 for IOS
            b_w = width * 270 / 1080
            b_h = height * 170 / 1920
            card_box = (0.50462963 * width - b_w / 2, height * 0.471354167 - b_h / 2, b_w, b_h)
        else:  # Height ~ 4000 for android
            b_w = width * 784 / 3000
            b_h = height * 495 / 4000
            card_box = (width * 0.500666667 - b_w / 2, height * 0.416875 - b_h / 2, b_w, b_h)
        return card_box

    def overlap_percentage(self, rect1, rect2):
        """
        Calculate the overlap percentage between two rectangles. (to use at infer fuction)
        
        Args:
        - rect1, rect2: Tuples representing rectangles in the format (x_tl, y_tl, width, height).
        
        Returns:
        - overlap_percent: Percentage of overlap between the two rectangles. 
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Calculate the coordinates of the bottom-right corners
        rect1_bottom_right = (x1 + w1, y1 + h1)
        rect2_bottom_right = (x2 + w2, y2 + h2)

        # Calculate the coordinates of the intersection rectangle
        intersect_x1 = max(x1, x2)
        intersect_y1 = max(y1, y2)
        intersect_x2 = min(rect1_bottom_right[0], rect2_bottom_right[0])
        intersect_y2 = min(rect1_bottom_right[1], rect2_bottom_right[1])

        # Calculate the width and height of the intersection rectangle
        intersect_width = max(0, intersect_x2 - intersect_x1)
        intersect_height = max(0, intersect_y2 - intersect_y1)

        # Calculate the area of the intersection rectangle
        intersect_area = intersect_width * intersect_height

        # Calculate the area of rect1
        rect2_area = w2 * h2

        # Calculate the overlap percentage relative to rect1
        overlap_percentage = (intersect_area / rect2_area) if rect2_area > 0 else 0

        # Determine the relationship between the rectangles
        if intersect_width > 0 and intersect_height > 0:
            if x1 >= x2 and y1 >= y2 and rect1_bottom_right[0] <= rect2_bottom_right[0] and rect1_bottom_right[1] <= \
                    rect2_bottom_right[1]:
                relationship = "inside"
            else:
                relationship = "overlaps"
        else:
            relationship = "none"

        return relationship, overlap_percentage

    def infer(self, image, cropTime_threshold, crop_rate, i_type):
        """
        Calculate the overlap percentage between two rectangles. (to use at infer fuction)
        
        Args:
        - rect1, rect2: Tuples representing rectangles in the format (x, y, width, height).
        
        Returns:
        - overlap_percent: Percentage of overlap between the two rectangles. 
        """
        detected_result = self.detect_device(image, cropTime_threshold, crop_rate)
        rect2 = self.define_card_position(image)

        if detected_result == None:
            return [int(0), 0.0]  # Anh hop le voi RecaptureObs = 0.0
        else:
            detected_devices = detected_result[0].boxes

        for i in range(0, len(detected_devices.cls.tolist())):
            bbox = detected_devices[i].xywh[0].tolist()

            rect1 = [bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[2], bbox[3]]

            relationship, overlap_percentage = self.overlap_percentage(rect1, rect2)

            if overlap_percentage > 0.5:
                return [int(1), 1.0]  # Anh recapture voi RecaptureObs = 1.0
        return [int(0), 0.0]  # Anh hop le voi RecaptureObs = 0.0
