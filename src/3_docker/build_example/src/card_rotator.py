import cv2
from PIL import Image
from matplotlib import pyplot as plt
from ultralytics import YOLO


class CardRotator:
    def __init__(self, model_file):
        self.model = YOLO(model_file, verbose=False, task='classify')
        print(' * Loading CARD ROTATOR model weight ', model_file)


    def rotate(self, img, threshold=0.5):
        results = self.model(img, verbose=False, imgsz=640)
        label = results[0].probs.top1
        prob = results[0].probs.top1conf.cpu().numpy()
        if prob < 0.6:
            return 0, prob, img
        if label == 0:
            return label, prob, img
        elif label == 1:
            return -90, prob, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif label == 2:
            return 90, prob, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif label == 3:
            return 180, prob, cv2.rotate(img, cv2.ROTATE_180)
        return 0, 0.0, img


if __name__ == '__main__':
    model = YOLO('/home/thienbd90/PycharmProjects/yolov8_train/runs/classify/train/weights/best.pt')
    image_path = '/home/thienbd90/Pictures/Screenshots/Screenshot from 2024-05-14 22-42-14.png'
    img = cv2.imread(image_path)
    results = model(img)
    print(results[0].probs)
    label = results[0].probs.top1
    prob = results[0].probs.top1conf.cpu().numpy()
    print(label, ' - ',prob)