from ultralytics import YOLO

# Load a model
model = YOLO('/home/shared/FPT/projects/z_16_screen_detection/src/models/runs/detect/train3/weights/best.pt')  # load a pretrained model

HOME = '/home/shared/FPT/projects/z_16_screen_detection/data/raw/test_wideview'
model.predict(HOME, save=True, imgsz=640)
