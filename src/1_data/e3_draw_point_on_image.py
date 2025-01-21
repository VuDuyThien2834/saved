import os
import cv2 as cv

image_path    = '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0_Computer_Monitor.v3i.yolov11/images/05ersXu1oMXozYJa66i9GEo-40-fit_lim-size_1050x_jpg.rf.1ead64d61d8c0d4537c06eeefcbaad74.jpg'
label_path      = '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0_Computer_Monitor.v3i.yolov11/labels/05ersXu1oMXozYJa66i9GEo-40-fit_lim-size_1050x_jpg.rf.1ead64d61d8c0d4537c06eeefcbaad74.txt'

image_path = '/home/shared/FPT/projects/z_16_screen_detection/data/raw/1_BlackAndWhite-laptop-screen-detection.v1i.yolov11/images/Screenshot-2024-04-30-120112_png.rf.a4977866833e9bf80dbe941caad4f642.jpg'
label_path = '/home/shared/FPT/projects/z_16_screen_detection/data/raw/1_BlackAndWhite-laptop-screen-detection.v1i.yolov11/labels/Screenshot-2024-04-30-120112_png.rf.a4977866833e9bf80dbe941caad4f642.txt'

image_path = '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0_Computer_Monitor.v3i.yolov11/images/Laptop_monitor_393_jpg.rf.86d16677a5757f6a951b3c31110f53e7.jpg'
label_path = '/home/shared/FPT/projects/z_16_screen_detection/src/data/new_label.txt'

image = cv.imread(image_path)
image_h, image_w, _ = image.shape
f = open(label_path, "r")
label = f.read().split(' ')

number_of_points = (len(label) - 1)//2

for i in range (0, number_of_points):
    point = (int((float(label[i*2+1])*image_w)), int(float(label[i*2+2])*image_h))
    image = cv.circle(image, point, radius=2, color=(0, 0, 255), thickness=2)

cv.imwrite('pointed_image.jpg', image)

