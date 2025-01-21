import os
import shutil

_0_laptop_data_path = [
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0_BlackAndWhite-laptop-screen-detection.v1i.yolov11/labels',
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0123_Screen Detection v2.v1i.yolov11/split_by_class/class_0_laptop/labels'
]

_1_smartphone_data_path = [
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/1_Mobile Phone Dataset.v2i.yolov11/labels',
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/1_mobile_detector.v2i.yolov11/labels',
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0123_Screen Detection v2.v1i.yolov11/split_by_class/class_1_smartphone/labels'
]

_2_monitor_data_path = [
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/2_Computer_Monitor.v3i.yolov11/labels',
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/2_MonitorDetection.v4i.yolov11/labels',
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0123_Screen Detection v2.v1i.yolov11/split_by_class/class_2_monitor/labels'
]

_3_tablet_data_path = [
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0123_Screen Detection v2.v1i.yolov11/split_by_class/class_3_tablet/labels'
]

import os

def replace_first_character(folder_path, class_id):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r+') as file:
                contents = file.readlines()
                if contents:  # Check if file is not empty
                    for content in contents:
                        content = str(class_id) + content[1:]  # Replace first character with '0'
                        file.seek(0)
                        file.write(content)
                        file.truncate()

# Replace 'folder_path' with the path of your folder containing .txt files
for folder_path in _0_laptop_data_path:
    class_id = 0
    replace_first_character(folder_path, class_id)

for folder_path in _1_smartphone_data_path:
    class_id = 1
    replace_first_character(folder_path, class_id)

for folder_path in _2_monitor_data_path:
    class_id = 2
    replace_first_character(folder_path, class_id)

for folder_path in _3_tablet_data_path:
    class_id = 3
    replace_first_character(folder_path, class_id)
