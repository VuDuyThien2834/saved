import math
import os
import shutil
import time
import traceback
import uuid

from PIL import ImageDraw, Image
from ultralytics import YOLO
import cv2
import numpy as np
import sympy

from src import utils
from src.card_rotator import CardRotator
from src.utils import DEBUG


def appx_best_fit_ngon(polygon):
    hull = cv2.convexHull(np.array(polygon))
    hull = np.array(hull).reshape((len(hull), 2))
    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]

    # for pt in hull:
    #     print(pt)

    tick = time.time()
    # run until we cut down to n vertices
    while len(hull) > 4:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # trước tiên chúng ta cần đảm bảo rằng tổng các góc trong mà cạnh tạo với hai cạnh liền kề lớn hơn 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # Tìm đỉnh mới nếu xóa cạnh này
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # diện tích của tam giác chúng ta sẽ thêm vào
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # xóa cạnh và thêm giao điểm của các cạnh liền kề
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)
        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    # back to python land
    hull = [(int(x), int(y)) for x, y in hull]
    print(' * Time crop =', time.time() - tick)
    return hull


def distance_2p(p1, p2):
    print(p1, p2)
    return int(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def padding_img(img, ratio=1.2, is_resize=False, hori=True):
    """

    :param img:
    :param ratio:
    :return:
    """
    ht, wd, cc = img.shape
    try:
        # create new image of desired size and color (blue) for padding
        if hori:
            hh = int(ht * ratio)
        else:
            hh = int(ht)  # int(ht * ratio)
        ww = int(hh * wd / ht)
        color = (255, 255, 255)
        result = np.full((hh, ww, cc), color, dtype=np.uint8)

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2

        # copy img image into center of result image
        result[yy:yy + ht, xx:xx + wd:] = img
        # result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return result
    except:
        print('lỗi ảnh padding...')

    return img


def order_points(pts):
    try:
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        if len(pts) < 4:
            return None, None, None, None
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        tl = rect[0]
        tr = rect[1]
        bl = rect[3]
        br = rect[2]

        return tl, tr, bl, br
    except:
        return None, None, None, None


def get_center_point(box):
    x_min, y_min, x_max, y_max = box
    return (x_min + x_max) // 2, (y_min + y_max) // 2


def perspective_transoform(image, source_points, trans_type=0):
    if trans_type == 0:
        w, h = 1000, 600
    else:
        w, h = 600, 1000
    dest_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, matrix, (w, h))
    return dst


def get_4point_in_polygon(polygon, points):
    # x_coordinates, y_coordinates = zip(*polygon)
    #
    # bl, tr = [min(x_coordinates), min(y_coordinates)], [max(x_coordinates), max(y_coordinates)]
    # padding = int(math.fabs(bl[1] - tr[1]) * 0.1)
    #
    # bl = [bl[0] - padding, bl[1] - padding]
    # tr = [tr[0] + padding, tr[1] + padding]
    # print(points)
    # four_points = []
    # for p in points:
    #     if bl[0] <= p[0] <= tr[0] and bl[1] <= p[1] <= tr[1]:
    #         four_points.append(p)
    #         if len(four_points) == 4:
    #             return four_points
    # print(four_points)
    return points


def gettype(label):
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx= ', label)
    labels = {'9s_mt': 1, '9s_ms': 2, '12s_mt': 3, '12s_ms': 4, 'chip_mt': 5, 'chip_ms': 6, 'ppvn': 7, 'cc_2024_mt': 8, 'cc_2024_ms': 9}
    try:
        return labels[str(label)]
    except KeyError:
        return -1


class CardCropper:
    def __init__(self, n_ngon_model_file=None, corner_model_file=None, rotator_model_file=None):
        self.n_gon_model = YOLO(n_ngon_model_file, task='segment')
        self.corner_model = YOLO(corner_model_file, verbose=False, task='detect')
        self.names = self.corner_model.names
        self.card_rotator = CardRotator(rotator_model_file)
        print(' * Loading CARD CROPPER model  weight ', n_ngon_model_file, '\n\t - ', corner_model_file,
              '\n\t - ', rotator_model_file)


    def crop_idcard(self, img, threshold=0.7, device_type='0'):
        crop, (tl, br) = self.crop_corner(img)
        if device_type == '1':
            print(' * Device type: ', device_type, ' --> crop = raw_img')
            _, _, rotated_img = self.card_rotator.rotate(img)
            return rotated_img, (tl, br)
        
        img = padding_img(img)
        crop, (tl, br) = self.crop_corner(img)
        # crop, label, score, (tl, br) = self.crop_corner(img)
        if crop is None:
            print(' * Crop with polygon')
            crop, (tl, br) = None, (None, None)  #self.crop_n_gon(img, threshold)
        else:
            print(' * Crop with 4-cornres')
            _, _, rotated_img = self.card_rotator.rotate(crop)
            return rotated_img, (tl, br)

    def get_type_polygon(self, cv_img, threshold=0.5, device_type='0'):
        try:
            results = self.n_gon_model.predict(cv_img, verbose=False, imgsz=480)
            polygon = results[0].masks[0].xy[0]
            label = int(results[0].boxes.cls.cpu().numpy()[0]) + 1

            score = results[0].boxes.conf.cpu().numpy()[0]
            if DEBUG:
                pil_img = Image.fromarray(cv_img)
                draw = ImageDraw.Draw(pil_img)
                draw.polygon(polygon, outline=(0, 255, 0), width=5)
                pil_img.save('./debug/preprocessing/idcard/2_1_cropper_polygon.jpg')
            return label, score, polygon
        except:
            return None, None, None

    def crop_corner(self, cv_img, threshold=0.6):
        try:
            # cv_img = padding_img(cv_img)
            results = self.corner_model.predict(source=cv_img, save=False, verbose=False)  # save plotted images
            boxes = results[0].boxes.xyxy.cpu().tolist()
            probs = results[0].boxes.conf.cpu().tolist()

            if DEBUG:
                plot = results[0].plot()
                utils.save_image('./debug/preprocessing/idcard/2_2_cropper_corners.jpg', plot)
            new_box = non_max_suppression_fast(boxes, 150)

            new_bb = []
            for idx, box in enumerate(new_box):
                if probs[idx] < threshold:
                    continue

                x, y = get_center_point(box)
                new_bb.append((x, y))

            if len(new_bb) >= 4:
                # label, score, polygon = self.get_type_polygon(cv_img)
                # print('==============================================')
                # print(new_bb)
                # new_bb = get_4point_in_polygon(polygon, new_bb)
                # print(new_bb)
                tl, tr, bl, br = order_points(np.array(new_bb))

                ww = distance_2p(tl, tr)
                hh = distance_2p(tl, bl)
                type_trans = 0 if ww > hh else 1
                crop = perspective_transoform(cv_img, np.float32([tl, tr, br, bl]), trans_type=type_trans)

                return crop, (tl, br)
            return None, (None, None)
        except:
            return None, (None, None)

    def crop_n_gon(self, cv_img, threshold=0.5):
        try:
            label, score, polygon = self.get_type_polygon(cv_img)
            hull = appx_best_fit_ngon(polygon)
            tl, tr, bl, br = order_points(np.array(hull))
            ww = distance_2p(tl, tr)
            hh = distance_2p(tl, bl)
            type_trans = 0 if ww > hh else 1
            crop = perspective_transoform(cv_img, np.float32([tl, tr, br, bl]), trans_type=type_trans)
            cv2.imwrite(
                f'/home/thienbd90/debug/9_ngon/{uuid.uuid4()}_ngon.jpg', crop)

            return crop, label, score, (tl, br)
        except:
            traceback.print_exc()
            return cv_img, 5, 0.0, ((0, 0), (cv_img.shape[1], cv_img.shape[0]))

    # def crop_corner_make_box(self, cv_img, threshold=0.6, file=''):
    #     try:
    #         cv_img = padding_img(cv_img)
    #         cv2.imwrite(f'/home/thienbd90/PycharmProjects/yolov8_train/datasets/corners_train_all/images2/{file}', cv_img)
    #         return None
    #         results = self.corner_model.predict(source=cv_img, save=False, verbose=False)  # save plotted images
    #         boxes = results[0].boxes.xyxy.cpu().tolist()
    #         probs = results[0].boxes.conf.cpu().tolist()
    #         classes = results[0].boxes.cls.cpu().tolist()
    #         top_left, top_right, bottom_left, bottom_right = None, None, None, None
    #         # if DEBUG:
    #         #     save_img = results[0].plot()
    #         #     cv2.imwrite('./debug/preprocessing/idcard/2_2_cropper_corners.jpg', save_img)
    #         h, w = cv_img.shape[:2]
    #         content = f'<annotation>	<folder>news</folder>	<filename>{file}</filename>	<path>/media/thienbd/T7 Touch/TRAINING_DATA/IDCARD/front_textbox_dect/news/{file}</path><source>		<database>Unknown</database>	</source>	<size>		<width>{w}</width>		<height>{h}</height><depth>3</depth>	</size>	<segmented>0</segmented>'
    #         flag = False
    #         for idx, box in enumerate(boxes):
    #             cls = self.names[int(classes[idx])]
    #             x_min, y_min, x_max, y_max = box
    #             # if probs[idx] < threshold:
    #             #     continue
    #             if top_left is None and cls == 'Top left':
    #                 flag = True
    #                 top_left = get_center_point(box)
    #                 content += f'<object>   <name>corner</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'
    #
    #             elif top_right is None and cls == 'Top right':
    #                 flag = True
    #                 top_right = get_center_point(box)
    #                 content += f'<object>   <name>corner</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'
    #
    #             elif bottom_left is None and cls == 'Bottom left':
    #                 flag = True
    #                 bottom_left = get_center_point(box)
    #                 content += f'<object>   <name>corner</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'
    #
    #             elif bottom_right is None and cls == 'Bottom right':
    #                 flag = True
    #                 bottom_right = get_center_point(box)
    #                 content += f'<object>   <name>corner</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'
    #
    #         content += '</annotation>'
    #         if flag:
    #
    #             with open(f'/home/thienbd90/PycharmProjects/yolov8_train/datasets/corners_train_all/annotations/{file}'.replace('.jpg','.xml'),
    #                           'w') as f:
    #                     f.write(content)
    #
    #         if top_left is None or top_right is None or bottom_left is None or bottom_right is None:
    #             return None, (None, None)
    #
    #         image_cropped = None #perspective_transoform(cv_img, np.float32([top_left, top_right, bottom_right, bottom_left]), trans_type=0)
    #
    #         return image_cropped, (top_left, bottom_right)
    #     except IndexError:
    #         return None, (None, None)


def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return boxes
    new_boxes = [boxes[0]]

    for idx, box in enumerate(boxes):
        if select_box(new_boxes, box, overlapThresh):
            new_boxes.append(box)
    return new_boxes


def select_box(accept_boxes, selected_box, overlapThresh=100):
    x_s, y_s = get_center_point(selected_box)
    for box in accept_boxes:
        x, y = get_center_point(box)
        distance = distance_2p([x_s, y_s], [x, y])
        if distance < overlapThresh:
            return False

    return True


if __name__ == '__main__':

    cropper = CardCropper(n_ngon_model_file='./models/idcard_models/cropper/type_ngon.pt',
                          corner_model_file='./models/idcard_models/cropper/corners_new.pt',
                          rotator_model_file='./models/idcard_models/rotator/best.pt')

    image = cv2.imread('')
    cropped, (_, _) = cropper.crop_corner(image)
    cv2.imshow('cropped', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
