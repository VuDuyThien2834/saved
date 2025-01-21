import cv2
from ultralytics import YOLO



class RECAPTURE_CHECK:
    def __init__(self, model_1):
        self.thienbd_recap = YOLO(model_1)
        # self.hungvs_recap = HVS_RECAPTURE()
        print(' * Loading RECAPTURE CHECKER model weight ', model_1)

    def yl_predict(self, cv_image):
        results = self.thienbd_recap.predict(cv_image, imgsz=224, verbose=False)

        print(' * Top5 predict recapture: ', results[0].probs.top5, ' - Conf: ', results[0].probs.top5conf.cpu().numpy().tolist())

        probs = results[0].probs.top5conf.cpu().numpy().tolist()
        labels = results[0].probs.top5
        return labels, probs

    def check_recapture(self, cv_image):
        labels, probs = remove_photo_result(self.thienbd_recap.yl_predict(cv_image))
        return labels, probs


def remove_photo_result(labels, confs):
    index = -1
    for i in labels:
        if i == 2:
            index = i
            break
    if index != -1:
        labels.remove(index)
        confs.remove(index)

    return labels, confs

