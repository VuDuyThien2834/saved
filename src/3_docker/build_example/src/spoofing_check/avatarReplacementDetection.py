from ultralytics import YOLO

import numpy as np
import cv2 as cv



class avatarReplacementDetection:
	def __init__(self, avatar_replacement_detection_w_p):
		self.avatar_replacement_detection_model	=	YOLO(avatar_replacement_detection_w_p)    
	
	def infer(self, im_aligned__card, device='cuda'):
		result = self.avatar_replacement_detection_model(im_aligned__card, imgsz=640, device=device, save=True, verbose = True)
		if result[0].boxes.xywh.tolist() == []:
			return (0,1.0) #Can't detect => Auth
		else:
			cls = int(result[0].boxes.cls.tolist()[0])
			conf = result[0].boxes.conf.tolist()[0]
			return (cls, conf)