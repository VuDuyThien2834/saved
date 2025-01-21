import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import random

# Function to draw polygons on an image
def visualize_coco_segmentation(image_path, txt_path):
    # Load the image
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (Matplotlib)
    img_h, img_w = image.shape[:2]
    image = cv2.resize(image, (500, int(img_h*500/img_w)))
    
    # Read segmentation data from the .txt file
    if os.path.isfile(txt_path):
        with open(txt_path, 'r') as file:
            lines = file.readlines()

        img_h, img_w = image.shape[:2]
        
        for line in lines:
            # Each line might contain: class_id, x1, y1, x2, y2, ..., xn, yn
            data = list(map(float, line.strip().split()))
            class_id = int(data[0])  # First element is the class ID
            for i in range(0, len(data)):
                if i%2==0:
                    data[i] *= img_h
                else:
                    data[i] *= img_w

            polygon_points = np.array(data[1:]).reshape(-1, 2)  # Rest are polygon points (x, y pairs)

            # Draw the polygon on the image (use green color)
            polygon_points = polygon_points.astype(int)
            cv2.polylines(image, [polygon_points], isClosed=True, color=(125, 255, 0), thickness=2)

    # Display the image with segmentation
    cv2.imshow('YOLO Annotations', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    subset = 'train'
    folder = '/home/shared/FPT/projects/z_16_screen_detection/data/raw/Computer_Monitor.v3i.yolov11'
    
    image_names = os.listdir(os.path.join(folder, subset, 'images'))
    number_of_choice = 20
    image_choices = random.choices(image_names, k=number_of_choice)
    for file in image_choices:
        image_path = os.path.join(folder, subset, 'images',  file)
        annotation_path = os.path.join(folder, 'labels', subset, file.rsplit('.', 1)[0] + '.txt')
        visualize_coco_segmentation(image_path, annotation_path)

    # image_folder_path = '/home/shared/FPT/projects/z_13_iBeta/data/interim/split_images/2_Smartphone'
    # label_folder_path = '/home/shared/FPT/projects/z_13_iBeta/data/interim/split_images/2_Smartphone_segment_label'
    # image_names = os.listdir(label_folder_path)
    
    # number_of_choice = 20
    # image_choices = random.choices(image_names, k=number_of_choice)
    # for file in image_choices:
    #     file_name = file.rsplit('.',1)[0]
    #     image_path = os.path.join(image_folder_path, file_name  + '.jpg')
    #     annotation_path = os.path.join(label_folder_path, file_name + '.txt')
    #     visualize_coco_segmentation(image_path, annotation_path)
