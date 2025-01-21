import os
import uuid

import cv2

# from readmrz import MrzDetector, MrzReader
from ultralytics import YOLO

# from infor_detector.infor_box_all_2024 import INFO_BOX_DET_2024

# mrz_detector = MrzDetector()
# mrz_reader = MrzReader()

# text_box_det_2024 = INFO_BOX_DET_2024(weight_path='./models/idcard_models/textbox/textbox_cc2024.pt')
# for root, dirs, files in os.walk('/home/thienbd90/PycharmProjects/yolov8_train/datasets/train_cc_cccd_2_mat/images'):
#     for file in files:
#         if file.endswith('.jpg'):
#
#             image = cv2.imread(os.path.join(root, file))
#
#             cropped = mrz_detector.crop_area(mrz_detector.read(os.path.join(root, file)))
#             result = mrz_reader.read_mrz(cropped)
#
#             lines = result.split('\n')
#             print(lines)
#             if len(lines) != 3:
#                 continue
#
#             map_boxes = text_box_det_2024.predict_back(image, 0.5
#                                                        )
#             mrz_lines = map_boxes['mrz_line']
#             if len(mrz_lines) != 3:
#                 continue
#             for idx, mrz_line in enumerate(mrz_lines):
#                 xmin, ymin, xmax, ymax = mrz_line[0]
#                 img_mrz = image[ymin:ymax, xmin:xmax]
#                 filename = str(uuid.uuid4()).replace('-', '_')
#                 if len(lines[idx]) !=30:
#                     continue
#                 cv2.imwrite(f'/media/thienbd90/df6f6696-1524-408b-b4e5-09bd861d318a/data/cccd_cmt/train_mrz/reals/{filename}.jpg', img_mrz)
#                 with open(f'/media/thienbd90/df6f6696-1524-408b-b4e5-09bd861d318a/data/cccd_cmt/train_mrz/reals.txt',
#                       'a') as f:
#                     f.write(f'reals/{filename}.jpg\t{lines[idx]}\n')
#                 # input('.............')

model = YOLO('/home/thienbd90/PycharmProjects/yolov8_train/runs/detect/train6/weights/best.pt', verbose=False)
for root, dirs, files in os.walk('/media/thienbd90/df6f6696-1524-408b-b4e5-09bd861d318a/data/cccd_cmt/data_cccd_cmt/8/cc2024_mt'):
    for file in files:
        if file.endswith('.jpg'):
            image = cv2.imread('/home/thienbd90/Desktop/448856618_2142421929463334_5179930703831429284_n.jpg')
            # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            results = model(image)
            img_plot = results[0].plot()
            cv2.imshow('image', img_plot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
