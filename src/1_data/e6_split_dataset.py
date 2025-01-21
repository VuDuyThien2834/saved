import os
import random
import shutil

#1. create final_data_folder_path and sub_folder
out_home = '/home/shared/FPT/projects/z_16_screen_detection/data/processed/final_screen_detection_v1'

os.makedirs(out_home, exist_ok=True)

for subset in ['train', 'val', 'test']:
    os.makedirs(os.path.join(out_home, subset), exist_ok=True)
    for folder in ['images', 'labels']:
        os.makedirs(os.path.join(out_home, subset, folder), exist_ok=True)


#- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - 
# COPY OBJECT IMAGES TO THE SUBSETS
# 2. List of folder contain images and folder contain labels
folders = [
    #obj
    #CLASS 0: LAPTOP
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0_BlackAndWhite-laptop-screen-detection.v1i.yolov11',
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0123_Screen Detection v2.v1i.yolov11/split_by_class/class_0_laptop',

    #CLASS 1: SMARTPHONE
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/1_Mobile Phone Dataset.v2i.yolov11',
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/1_mobile_detector.v2i.yolov11',
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0123_Screen Detection v2.v1i.yolov11/split_by_class/class_1_smartphone',

    #CLASS 2: MONITOR
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/2_Computer_Monitor.v3i.yolov11',
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/2_MonitorDetection.v4i.yolov11',
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0123_Screen Detection v2.v1i.yolov11/split_by_class/class_2_monitor',

    #CLASS 3: TABLET
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0123_Screen Detection v2.v1i.yolov11/split_by_class/class_3_tablet'
]

#3. Count number of each subset, later to calculate number of background will be added to the dataset
num_of_train_object = 0
num_of_val_object = 0
num_of_test_object = 0

# 4. Copy object image to train, val, test subset
for folder in folders:
    #4.1 list image names: if the file don't has '.json' extension
    image_folder = os.path.join(folder, "images")
    label_folder = os.path.join(folder, "labels")
    images = []
    for file in os.listdir(image_folder):
        images.append(file)
    
    # 4.2 calculate subset size and shuffle data
    count_images = len(images)
    train_ratio = 0.7
    val_ratio = 0.2
    trainset_size = int(train_ratio * count_images)
    valset_size = int(val_ratio * count_images)
    random.shuffle(images)

    #4.3 Copy image and label to make train set
    for i in range (0, trainset_size):
        image_name = images[i]
        label_name = image_name[0:-4] + '.txt'
        shutil.copy(os.path.join(image_folder, image_name), os.path.join(out_home, 'train', 'images', 'obj_' + image_name))
        if os.path.isfile(os.path.join(label_folder, label_name)):
            shutil.copy(os.path.join(label_folder, label_name), os.path.join(out_home, 'train', 'labels', 'obj_' + label_name))
    
    #4.4 Copy image and label to make val set
    for i in range (trainset_size, trainset_size + valset_size):
        image_name = images[i]
        label_name = image_name[0:-4] + '.txt'
        shutil.copy(os.path.join(image_folder, image_name), os.path.join(out_home, 'val', 'images', 'obj_' + image_name))
        if os.path.isfile(os.path.join(label_folder, label_name)):
            shutil.copy(os.path.join(label_folder, label_name), os.path.join(out_home, 'val', 'labels', 'obj_' + label_name))
    
    #4.5 Copy image and label to make test set
    for i in range (trainset_size + valset_size, count_images):
        image_name = images[i]
        label_name = image_name[0:-4] + '.txt'
        shutil.copy(os.path.join(image_folder, image_name), os.path.join(out_home, 'test', 'images', 'obj_' + image_name))
        if os.path.isfile(os.path.join(label_folder, label_name)):
            shutil.copy(os.path.join(label_folder, label_name), os.path.join(out_home, 'test', 'labels', 'obj_' + label_name))

    #4.6 Count number of each subset
    num_of_train_object += trainset_size
    num_of_val_object += valset_size
    num_of_test_object += count_images - (trainset_size + valset_size)

#- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - 

# ADD BACKGROUND TO THE SUBSETS
# We recommend about 0-10% background images  [https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/]
# 5. List of folder contain images and folder contain labels
background_fs = [
    '/home/shared/FPT/projects/z_10_CBB_detection_new/data/interim/background_4/train/images',
    '/home/shared/FPT/projects/z_13_iBeta/data/interim/split_images_700/0_Real_Person',
    '/home/shared/FPT/projects/000_background_dataset/Indoor Object Detection.v3i.yolov11/images',
    '/home/shared/FPT/projects/000_background_dataset/interior item detection.v3i.yolov11/images',
    '/home/shared/FPT/projects/000_background_dataset/KP-SS Indoor Wall Segmentation.v1i.yolov11/images'
]
num_noOb_train, num_noOb_val, num_noOb_test = 0, 0, 0

backgroud_ratio = 0.2 # CHATGPT: While there’s no strict percentage, a recommended approach is to aim for around 10-30% background-only images in your dataset.


#7. For each folder
for j in range(0, len(background_fs)):
    folder = background_fs[j]
    
    #7.1 List image names and shuffle them
    images = []
    tmp_ps = os.listdir(folder)
    for p in tmp_ps:
        images.append(p)
    count_images = len(images)
    random.shuffle(images)

    # 6. calculate each subset's number of backgrounds
    num_train_background = int(num_of_train_object * backgroud_ratio/len(background_fs))
    num_val_background = int(num_of_val_object * backgroud_ratio/len(background_fs))
    num_test_background = int(num_of_test_object * backgroud_ratio/len(background_fs))
    if (num_train_background + num_val_background + num_test_background)> count_images:
        num_train_background =  int(count_images * train_ratio)
        num_val_background = int(count_images * val_ratio)
        num_test_background = count_images - int(count_images * (train_ratio+val_ratio))
    #7.2 Copy background imagesto each subset
    for i in range (0, num_train_background):
        image_name = images[i]
        shutil.copy(os.path.join(folder, image_name), os.path.join(out_home, 'train', 'images', f'bg{j}_' + image_name))
    for i in range (num_train_background, num_train_background + num_val_background):
        image_name = images[i]
        shutil.copy(os.path.join(folder, image_name), os.path.join(out_home, 'val', 'images', f'bg{j}_' + image_name))
    for i in range (num_train_background + num_val_background,num_train_background + num_val_background + num_test_background):
        image_name = images[i]
        shutil.copy(os.path.join(folder, image_name), os.path.join(out_home, 'test', 'images', f'bg{j}_' + image_name))
    num_noOb_train += num_train_background
    num_noOb_val += num_val_background
    num_noOb_test += num_test_background

#- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - #- - - - - - - - - - - - - - - - - - - - - - 
# LOGS
print('OBJECT')
print(f'Num of train object: {num_of_train_object}')
print(f'Num of val object: {num_of_val_object}')
print(f'Num of test object: {num_of_test_object}')
print('BACKGROUND')
print(f'Num of train background: {num_noOb_train}')
print(f'Num of val background: {num_noOb_val}')
print(f'Num of test background: {num_noOb_test}')