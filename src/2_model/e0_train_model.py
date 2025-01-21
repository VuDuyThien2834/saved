from ultralytics import YOLO

cfg_p='/home/shared/FPT/projects/z_16_screen_detection/src/models/cbb.yaml'
# Load a model
model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=cfg_p, epochs=500, imgsz=640,batch=128, fliplr=0)
