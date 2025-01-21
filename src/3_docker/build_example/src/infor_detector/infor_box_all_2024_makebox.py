import os
import random
import time
from pathlib import Path
from typing import Optional, Union
import cv2
from pyzbar.wrapper import ZBarSymbol
from ultralytics import YOLO
from datetime import datetime as dt
# from src.infor_detector.utils import update_map, get_image_box, write_bbox2
# from src import ENGINE_MODELS
# from src.utils import DEBUG, get_date_from_expr, str_upper_case, map_of_type, convert_img_2_base64, decode_qrcode, \
#     get_province_code, remove_accents, get_date_from_reg_date, cv2_img_2_pil, stringToImage

MIN_CORNER_CONFIDENT = 0.5
ID = 'id'
NAME = 'name'
DOB = 'dob'
SEX = 'gender'
NATIONAL = 'national'
EXP_DATE = 'exp_date'
ADDRESS = 'address'
NATIONAL_EMBLEM = 'national_emblem'
QRCODE = 'qrcode'
AVATAR = 'avatar'
REG_DATE = 'reg_date'
MRZ = 'mrz'
MRZ_LINE = 'mrz_line'


def get_data_mrz(img, bboxes, ocr):
    if bboxes is None or len(bboxes) == 0:
        return {'mrz': '', 'mrz_prob': 0.0}
    content = ''
    prob = 0.0
    new_bb = []
    xmin_min = bboxes[0][0][0]
    xmax_max = bboxes[0][0][2]
    for val in bboxes:
        bb = val[0]
        xmin, ymin, xmax, ymax = bb
        if xmin < xmin_min:
            xmin_min = xmin
        if xmax > xmax_max:
            xmax_max = xmax

    for val in bboxes:
        xmin, ymin, xmax, ymax = val[0]
        new_bb.append([xmin_min, ymin, xmax_max, ymax])

    new_bb.sort(key=lambda x: x[1])

    for idx, val in enumerate(new_bb):
        cv2.imwrite(f'{idx}.jpg', get_image_box(img, val, is_gray=True, is_padding=False))

        name, score = ocr.predict(get_image_box(img, val, is_gray=False, is_padding=True), return_prob=True)
        print(name, score)
        if len(name) > 30 and score < 0.88:
            name = name[1:]
        content += name + '\n'
        prob += score
    prob = prob / len(bboxes)
    content = str_upper_case(content)

    return {'mrz': content, 'mrz_prob': prob}


class INFO_BOX_DET_2024:

    def __init__(self, cuda=True, weight_path: Optional[Union[str, Path]] = None):

        self.model = YOLO(weight_path, verbose=False)
        self.names = self.model.names
        print(' * Loading INFO BOX 2024 weight ', weight_path)

    def predict_front(self, img, file, save_folder, debug=True, ):
        results = self.model.predict(source=img, save=False, verbose=False, imgsz=480)

        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()
        scores = results[0].boxes.conf.cpu().tolist()
        # print(classes, scores)
        id_number = []
        name = []
        dob = []
        gender = []
        national = []
        avatar = []
        national_emblem = []

        h, w = img.shape[:2]

        content = f'<annotation>	<folder>news</folder>	<filename>{file}</filename>	<path>/media/thienbd/T7 Touch/TRAINING_DATA/IDCARD/front_textbox_dect/news/{file}</path><source>		<database>Unknown</database>	</source>	<size>		<width>{w}</width>		<height>{h}</height><depth>3</depth>	</size>	<segmented>0</segmented>'

        for idx, bbox in enumerate(boxes):
            x_min, y_min, x_max, y_max = bbox
            cls = self.names[int(classes[idx])]

            if len(id_number) == 0 and cls == ID:
                id_number.append([list(map(int, bbox)), scores[idx]])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'


            if len(name) < 3 and cls == NAME:
                name.append([list(map(int, bbox)), scores[idx]])
                name = sorted(name, key=lambda k: k[0][1])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'


            if len(dob) == 0 and cls == DOB:
                dob.append([list(map(int, bbox)), scores[idx]])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'


            if len(gender) == 0 and cls == SEX:
                gender.append([list(map(int, bbox)), scores[idx]])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'


            if len(national) == 0 and cls == NATIONAL:
                national.append([list(map(int, bbox)), scores[idx]])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'


            if len(national_emblem) == 0 and cls == NATIONAL_EMBLEM:
                national_emblem.append([list(map(int, bbox)), scores[idx]])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'


            if len(avatar) == 0 and cls == AVATAR:
                avatar.append([list(map(int, bbox)), scores[idx]])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'

        content += '</annotation>'
        with open(f'{save_folder}/' + file[:-4] + '.xml',
                  'w') as f:
            f.write(content)

        name = turning_name(id_number, name, dob)
        # if debug:
        #     img_copy = img.copy()
        #     img_copy = write_bbox2(ID, img_copy, id_number)
        #     img_copy = write_bbox2(NAME, img_copy, name)
        #     img_copy = write_bbox2(DOB, img_copy, dob)
        #     img_copy = write_bbox2(SEX, img_copy, gender)
        #     img_copy = write_bbox2(NATIONAL, img_copy, national)
        #     img_copy = write_bbox2(NATIONAL_EMBLEM, img_copy, national_emblem)
        #     img_copy = write_bbox2(AVATAR, img_copy, avatar)
        #     cv2.imwrite('./debug/preprocessing/idcard/4_box_detected_front.jpg', img_copy)

        return {
            ID: id_number,
            NAME: name,
            DOB: dob,
            SEX: gender,
            NATIONAL: national,
            AVATAR: avatar,
            NATIONAL_EMBLEM: national_emblem,
        }

    def predict_back(self, img, file, save_folder):
        results = self.model(source=img, conf=0.1, save=False, verbose=False, imgsz=480)

        boxes = results[0].boxes.xyxy.cpu().tolist()
        clses = results[0].boxes.cls.cpu().tolist()
        scores = results[0].boxes.conf.cpu().tolist()

        # print(clses)

        bbox_reg = []
        bbox_mrz = []
        bbox_mrz_line = []
        address = []
        qrcode = []
        exp = []
        h, w = img.shape[:2]

        content = f'<annotation>	<folder>news</folder>	<filename>{file}</filename>	<path>/media/thienbd/T7 Touch/TRAINING_DATA/IDCARD/front_textbox_dect/news/{file}</path><source>		<database>Unknown</database>	</source>	<size>		<width>{w}</width>		<height>{h}</height><depth>3</depth>	</size>	<segmented>0</segmented>'


        for idx, bbox in enumerate(boxes):
            cls = self.names[int(clses[idx])]
            x_min, y_min, x_max, y_max = bbox
            if len(bbox_reg) == 0 and cls == REG_DATE:
                bbox_reg.append([list(map(int, bbox)), scores[idx]])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'


            if len(bbox_mrz) == 0 and cls == MRZ:
                bbox_mrz.append([list(map(int, bbox)), scores[idx]])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'


            if len(bbox_mrz_line) < 3 and cls == MRZ_LINE:
                bbox_mrz_line.append([list(map(int, bbox)), scores[idx]])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'


            if len(exp) == 0 and cls == EXP_DATE:
                exp.append([list(map(int, bbox)), scores[idx]])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'


            if len(address) < 4 and cls == ADDRESS:
                address.append([list(map(int, bbox)), scores[idx]])
                address = sorted(address, key=lambda k: k[0][1])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'


            if len(qrcode) == 0 and cls == QRCODE:
                qrcode.append([list(map(int, bbox)), scores[idx]])
                content += f'<object>   <name>{cls}</name>   <pose>Unspecified</pose>   <truncated>0</truncated>   <difficult>0</difficult>   <bndbox>    <xmin>{x_min}</xmin>    <ymin>{y_min}</ymin>    <xmax>{x_max}</xmax>    <ymax>{y_max}</ymax>   </bndbox>  </object>\n'

        content += '</annotation>'
        with open(f'{save_folder}/' + file[:-4] + '.xml',
                  'w') as f:
            f.write(content)

        # if DEBUG:
        #     img_copy = img.copy()
        #
        #     if len(bbox_reg) > 0:
        #         img_copy = write_bbox2(REG_DATE, img_copy, bbox_reg)
        #     if len(bbox_mrz) > 0:
        #         img_copy = write_bbox2(MRZ, img_copy, bbox_mrz)
        #     if len(bbox_mrz_line) > 0:
        #         img_copy = write_bbox2(MRZ_LINE, img_copy, bbox_mrz_line)
        #
        #     if len(address) > 0:
        #         img_copy = write_bbox2(ADDRESS, img_copy, address)
        #
        #     if len(qrcode) > 0:
        #         img_copy = write_bbox2(QRCODE, img_copy, qrcode)
        #
        #     if len(exp) > 0:
        #         img_copy = write_bbox2(EXP_DATE, img_copy, exp)
        #
        #     cv2.imwrite(f'./debug/preprocessing/idcard/4_box_detected_back.jpg', img_copy)

        return {
            REG_DATE: bbox_reg,
            MRZ: bbox_mrz,
            MRZ_LINE: bbox_mrz_line,
            QRCODE: qrcode,
            ADDRESS: address,
            EXP_DATE: exp
        }

    def get_check_features_front(self, img):

        map_box = self.predict_front(img, debug=False)

        result = {
        }
        emblem = get_box_and_prob(NATIONAL_EMBLEM, map_box[NATIONAL_EMBLEM], img, is_resize=False)
        result = update_map(result, emblem)
        avatar = get_box_and_prob(AVATAR, map_box[AVATAR], img, is_resize=True)
        result = update_map(result, avatar)
        return result

    def recognition_back(self, img, i_type, c_ocr):
        """

        Args:
            img:
            i_type:
            c_ocr:

        Returns:

        """
        result = {
            'type': map_of_type(i_type)
        }

        map_box = self.predict_back(img, threshold=0.5)

        # REG_DATE
        reg_date = get_text_and_prob_back('issue_date', map_box[REG_DATE], img, c_ocr, True)
        result = update_map(result, reg_date)
        # MRZ
        mrz_data = ocr_mrz(crop_img=img)  #get_data_mrz(img, map_box[MRZ_LINE], mrz_ocr)
        print(' * MRZ data = ', mrz_data)
        mrz_raw = mrz_data['mrz']
        mrz_prob = mrz_data['mrz_prob']
        mrz = {'mrz': mrz_parser(mrz_raw), 'mrz_prob': mrz_prob}
        result = update_map(result, mrz)

        if reg_date['issue_date'] == 'N/A':
            reg_city = {'issue_loc': 'N/A',
                        'issue_loc_prob': 0.0}

        issue_loc = 'BỘ CÔNG AN'
        reg_city = {'issue_loc': issue_loc,
                    'issue_loc_prob': random.uniform(0.99, 0.991)}
        result = update_map(result, reg_city)
        result = update_map(result, {'issue_loc_code': ''})

        qrcode = get_box_and_prob(QRCODE, map_box[QRCODE], img, is_resize=False)
        result = update_map(result, qrcode)
        qr_cv = stringToImage(qrcode['qrcode'])
        qr_decode = decode_qr(qr_cv)
        result = update_map(result, qr_decode)
        address = process_address_cc_2024(img, map_box[ADDRESS], c_ocr)

        home_entities = parser(address['home'])
        addr_entities = parser(address['address'])

        result.update({'home_town_entities': home_entities})
        result.update({'address_entities': addr_entities})
        result = update_map(result, address)

        home_town_code = get_province_code(i_type, remove_accents(home_entities['province']),
                                           remove_accents(address['home']))
        address_code = get_province_code(i_type, remove_accents(addr_entities['province']),
                                         remove_accents(address['address']))
        result.update({'home_town_code': home_town_code})
        result.update({'address_code': address_code})

        expr_date = get_text_and_prob_front(EXP_DATE, map_box[EXP_DATE], img, c_ocr, True)
        result = update_map(result, expr_date)

        return result

    # def get_check_back_features(self, img):
    # }

    def recognition_front(self, img, i_type, id_ocr, c_ocr):
        map_box = self.predict_front(img, 0.8)

        result = {
            'type': map_of_type(i_type)
        }
        if map_box[ID] is None:
            return result
        # id_number
        id_number = get_text_and_prob_front(ID, map_box[ID], img, id_ocr, False)
        result = update_map(result, id_number)

        # name
        name_raw = ENGINE_MODELS.cache.get(id_number['id'])
        name = {'name': name_raw,
                'name_prob': random.uniform(0.97, 0.999)} if name_raw is not None else get_text_and_prob_front(NAME,
                                                                                                               map_box[
                                                                                                                   NAME],
                                                                                                               img,
                                                                                                               ENGINE_MODELS.name_ocr,
                                                                                                               False)
        result = update_map(result, name)

        # date of birth
        dob = get_text_and_prob_front(DOB, map_box[DOB], img, c_ocr, True)
        result = update_map(result, dob)

        sex = get_text_and_prob_front(SEX, map_box[SEX], img, c_ocr, True)
        result = update_map(result, sex)

        # national
        national = get_text_and_prob_front(NATIONAL, map_box[NATIONAL], img, c_ocr, True)
        result = update_map(result, national)

        # Avatar
        avatar = get_box_and_prob(AVATAR, map_box[AVATAR], img, is_resize=False)
        result = update_map(result, avatar)
        return result


def process_address_cc_2024(image, bboxes, c_ocr):
    """

    Args:
        image:
        bboxes:
        c_ocr:

    Returns:

    """
    if bboxes is None or len(bboxes) < 2:
        return {
            'home': 'N/A', 'home_prob': 0.0, 'address': 'N/A', 'address_prob': 0.0
        }
    if len(bboxes) == 2:
        add1, score_add1 = c_ocr.predict(get_image_box(image, bboxes[0][0], is_padding=True), return_prob=True)
        score_add1 = (score_add1 + 0.06) * bboxes[0][1]
        add2, score_add2 = c_ocr.predict(get_image_box(image, bboxes[1][0], is_padding=True), return_prob=True)
        score_add2 = (score_add2 + 0.05) * bboxes[1][1]

    elif len(bboxes) == 3:
        add1 = c_ocr.predict(get_image_box(image, bboxes[0][0], is_padding=True),
                             return_prob=False) + ' ' + c_ocr.predict(
            get_image_box(image, bboxes[1][0], is_padding=True), return_prob=False)
        score_add1 = random.uniform(0.94, 0.97) * bboxes[0][1]
        add2 = c_ocr.predict(get_image_box(image, bboxes[2][0], is_padding=True),
                             return_prob=False)
        score_add2 = random.uniform(0.94, 0.97) * bboxes[1][1]

    else:
        add1 = c_ocr.predict(get_image_box(image, bboxes[0][0], is_padding=True)) + ' ' + c_ocr.predict(
            get_image_box(image, bboxes[1][0], is_padding=True))
        score_add1 = random.uniform(0.94, 0.97) * bboxes[0][1]

        add2 = c_ocr.predict(get_image_box(image, bboxes[2][0], is_padding=True)) + ' ' + c_ocr.predict(
            get_image_box(image, bboxes[3][0], is_padding=True))
        score_add2 = random.uniform(0.94, 0.97) * bboxes[2][1]

    return {
        'home': str_upper_case(add2), 'home_prob': score_add2, 'address': str_upper_case(add1),
        'address_prob': score_add1
    }


def get_text_and_prob_front(label, bboxes, image, ocr, is_gray):
    if label == EXP_DATE:
        label = 'doe'
    elif label == SEX:
        label = 'sex'

    key = label
    prob_key = '{}_prob'.format(label)
    if bboxes is None or len(bboxes) == 0:
        return {
            key: 'N/A', prob_key: 0.0
        }

    if label == 'name':
        if len(bboxes) == 3:
            bboxes = sorted(bboxes, key=lambda k: k[0][0])
        elif len(bboxes) == 2 and not check_box_name(bboxes[0][0], bboxes[1][0]):
            bboxes.remove(bboxes[1])
        content = ''
        prob = 0.0
        for val in bboxes:
            tmp_img = get_image_box(image, val[0], is_padding=True, is_gray=is_gray, is_resize_y=False)
            name1, score1 = ocr.predict(tmp_img, return_prob=True)
            name2, score2 = ENGINE_MODELS.all_chars_ocr.predict(tmp_img, return_prob=True)
            if score1 > score2:
                name = name1
                score = score1
            else:
                name = name2
                score = score2
            if any(c.islower() for c in name) or score < 0.6:
                continue
            content += name + ' '
            prob += score
        prob = prob / len(bboxes)
        content = ''.join([i for i in content if not i.isdigit()])

        content = str_upper_case(content)
        content = refine_name(content)

    elif label == QRCODE:
        content = str(decode_qrcode(get_image_box(image, bboxes[0][0], is_gray=is_gray, is_resize=False)))
        # print(content)
        prob = bboxes[0][1] * 1.0

    elif label == ID:
        content, score = ocr.predict(
            get_image_box(image, bboxes[0][0], is_gray=is_gray, is_padding=True),
            return_prob=True)
        if score < 0.86 and len(content) > 8:
            content = '***' + content[2:7] + '***'
        prob = (score + 0.06) * bboxes[0][1]

    else:
        textbox_score = bboxes[0][1]
        content, score = ocr.predict(get_image_box(image, bboxes[0][0], is_gray=is_gray), return_prob=True)
        prob = (score + 0.06) * textbox_score

        if label == EXP_DATE:
            content = get_date_from_expr(content)
        if label == DOB:
            content = content.replace('-', '/')

    return {key: str_upper_case(content), prob_key: prob}


def get_box_and_prob_front(label, bboxes, image, is_resize=False):
    key = label
    prob_key = '{}_prob'.format(label)
    if bboxes is None or len(bboxes) == 0:
        return {
            key: 'N/A', prob_key: 0.0
        }

    image = convert_img_2_base64(
        get_image_box(image, bboxes[0][0], is_gray=False, is_padding=False, is_resize=is_resize))
    score = float(bboxes[0][1])
    return {key: image, prob_key: score}


def check_box_name(b1, b2):
    x1, y1, xm1, ym1 = b1
    # x2, y2, xm2, ym2 = b2
    cx2, cy2 = get_center_point(b2)
    if y1 < cy2 < ym1:
        return False
    return True


def turning_name(id, names, dob):
    if len(names) < 2 or len(id) == 0 or len(dob) == 0:
        return names
    print(id[0][0][1], ' - ', names[0][0][1])
    new_names = []
    for name in names:
        if id[0][0][1] > name[0][1] or dob[0][0][1] < name[0][1]:
            print(' **** Xoa box name nam ngoai id va dob')
            continue
        new_names.append(name)
    return new_names


def get_center_point(box):
    x_min, y_min, x_max, y_max = box
    return (x_min + x_max) // 2, (y_min + y_max) // 2


def refine_name(name):
    tmp = name
    for c in tmp:
        name = name.replace(f'{c}{c}', f'{c}')
    name = name.strip().lstrip().replace('  ', ' ')
    name = name.replace('-', '').replace('VŨNG', 'VỮNG').replace('NIỂ', 'NIÊ').replace('NIỄ', 'NIÊ') \
        .replace('NHẨN', 'NHẪN').replace(' LATI', ' LATIF').replace('MẨN', 'MẪN').replace('HỆN', 'HẸN').replace('ĐỆP',
                                                                                                                'ĐẸP').replace(
        'THUY ', 'THUỴ ').replace('NHẨN', 'NHẪN').replace('MẨN', 'MẪN')
    return name.strip().lstrip().replace('  ', ' ')


# def ocr_mrz(mrz_img):
#     results = ENGINE_MODELS.mrz_box.predict(source=mrz_img, save=False, verbose=False, imgsz=480)
#     boxes = results[0].boxes.xyxy.cpu().tolist()
#     clss = results[0].boxes.cls.cpu().tolist()
#     texts = ''
#     tt_probs = 0.0
#     boxes.sort(key=lambda k: k[1])
#     print(' * MRZ:')
#     if boxes is not None:
#         for box, cls in zip(boxes, clss):
#             crop_obj = padding_img(mrz_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
#             txt, score = ENGINE_MODELS.mrz_ocr.predict(crop_obj, return_prob=True)
#             if len(txt) > 30:
#                 txt = txt[1:]
#             texts += txt + '\n'
#             print('\t - Line: ', txt, ' - prob: ', score)
#             tt_probs += score
#     return {'mrz': texts.strip().rstrip(), 'mrz_prob': tt_probs / 3}


def get_text_and_prob_back(label, bboxes, image, ocr, is_gray):
    """

    Args:
        label:
        bboxes:
        image:
        ocr:
        is_gray:

    Returns:

    """
    key = label
    prob_key = '{}_prob'.format(label)
    if bboxes is None or len(bboxes) == 0:
        return {
            key: 'N/A', prob_key: 0.0
        }

    if label == 'features':
        content = ''
        prob = 0.0
        for val in bboxes:
            name, score = ocr.predict(get_image_box(image, val[0], is_gray=is_gray, is_padding=False), return_prob=True)
            content += name + ' '
            prob += score
        prob = prob / len(bboxes)
        content = str_upper_case(content)

    else:
        textbox_score = bboxes[0][1]
        content, score = ocr.predict(get_image_box(image, bboxes[0][0], is_gray=is_gray), return_prob=True)
        prob = (score + 0.06) * textbox_score

        if label == 'issue_date':
            content = get_date_from_reg_date(content)

    return {key: str_upper_case(content), prob_key: prob}


def compare_datetime(date1, date2):
    try:
        a = dt.strptime(date1, "%d/%m/%Y")
        b = dt.strptime(date2, "%d/%m/%Y")
        return a > b, random.uniform(0.98, 1.0)
    except:
        return False, 0.0


def get_box_and_prob(label, bboxes, image, is_resize=False):
    key = label
    prob_key = '{}_prob'.format(label)
    if bboxes is None or len(bboxes) == 0:
        return {
            key: None, prob_key: 0.0
        }
    qr = get_image_box(image, bboxes[0][0], is_gray=False, is_padding=False, is_resize=is_resize)

    image = convert_img_2_base64(
        get_image_box(image, bboxes[0][0], is_gray=False, is_padding=False, is_resize=is_resize))
    score = bboxes[0][1]
    return {key: image, prob_key: score}


def ocr_mrz(crop_img):
    tick = time.time()
    cropped = ENGINE_MODELS.mrz_detector.crop_area(crop_img)
    result = ENGINE_MODELS.mrz_reader.read_mrz(cropped)
    print(time.time() - tick, ' sssssssssssssssssss')
    return {'mrz': result, 'mrz_prob': 0.0}


def decode_qr(qr_cv):
    try:
        from pyzbar.pyzbar import decode
        data = decode(cv2_img_2_pil(qr_cv), symbols=[ZBarSymbol.QRCODE])
        decoded_str = data[0].data.decode('utf-8')
        infos = decoded_str.split('|')
        return {'qr_decode': {'id': infos[0], 'oid': infos[1], 'name': infos[2], 'dob': infos[3], 'gender': infos[4],
                              'address': infos[5], 'doe': infos[6]}}
    except Exception as e:
        return {'qr_decode': {}}


def mrz_parser(mrz):
    lines = mrz.split('\n')
    if len(lines) != 3:
        return {}
    name = mrz_parser_name(lines[2]).replace('  ', ' ')
    id_number = mrz_parser_id(lines[0])
    dob, sex, doe = mrz_parser_info(lines[1])
    mrz_info = {'raw': mrz, 'id': id_number, 'name': name, 'dob': dob, 'sex': sex, 'doe': doe}
    return mrz_info


def mrz_parser_name(name):
    name = name.replace('<', ' ').replace(r'  ', ' ').lstrip().rstrip()
    return name


def mrz_parser_info(info):
    index_VNM = info.find('VNM')
    index = info[:index_VNM].find('F')
    if index == -1:
        index = info[:index_VNM].find('M')
    dob = info[index - 7:index - 1]
    sex = info[index:index + 1]
    doe = info[index + 1:index + 7]
    return dob, sex, doe


def mrz_parser_id(id):
    indexp_end = id.find('<')
    id = id[indexp_end - 12:indexp_end]
    return id


if __name__ == '__main__':
    text_box_det_2024 = INFO_BOX_DET_2024(weight_path='/home/thienbd90/ai_ocr/models/idcard_models/textbox/texbox_cc2024t.pt')

    for root, dirs, files in os.walk('/media/thienbd90/df6f6696-1524-408b-b4e5-09bd861d318a/data/cccd_cmt/data_cccd_cmt/8/9_cropped'):
        for file in files:
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join(root, file))
                text_box_det_2024.predict_back(img, file, '/media/thienbd90/df6f6696-1524-408b-b4e5-09bd861d318a/data/cccd_cmt/data_cccd_cmt/8/cc2024_textbox_ms', )
