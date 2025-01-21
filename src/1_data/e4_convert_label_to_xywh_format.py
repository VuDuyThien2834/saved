import os 

paths = [
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/0_Computer_Monitor.v3i.yolov11/labels',
    '/home/shared/FPT/projects/z_16_screen_detection/data/raw/1_BlackAndWhite-laptop-screen-detection.v1i.yolov11/labels'
]

for path in paths:
    os.makedirs(os.path.join(path.rsplit('/',1)[0], 'new_labels'), exist_ok=True)
    for label_name in os.listdir(path):
        label_path = os.path.join(path, label_name)
        f = open(label_path, "r")
        lines = f.readlines()

        new_label = ''
        for line in lines:
            xs = []
            ys = []
            label = [float(a) for a in line.split(' ')]
            for i in range (1, len(label)):
                if i%2==1: 
                    xs.append(label[i])
                else:
                    ys.append(label[i])
            
            class_ID = int(label[0])
            x = min(xs) + (max(xs) - min(xs))/2
            y = min(ys) + (max(ys) - min(ys))/2
            w = (max(xs) - min(xs))
            h = (max(ys) - min(ys))

            new_label += f'{class_ID} {x} {y} {w} {h}\n'
        
        f = open(os.path.join(path.rsplit('/',1)[0], 'new_labels', label_name), "a")
        f.write(new_label)
        f.close()