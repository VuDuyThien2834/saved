import random
import time
from pathlib import Path
from typing import Optional, Union
from ultralytics import YOLO
from datetime import datetime as dt
from engine._core.address_parser.address_paser2 import parser
from src.infor_detector.utils import update_map, get_image_box, write_bbox2
from src import ENGINE_MODELS
from src.utils import DEBUG, get_date_from_expr, str_upper_case, map_of_type, convert_img_2_base64, decode_qrcode, \
    get_province_code, remove_accents, get_date_from_reg_date, cv2_img_2_pil, resize, save_image

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

SIGNALEMTENT = 'signalement'
REG_DATE = 'reg_date'
MRZ = 'mrz'
RED_STAMP = 'red_stamp'
LEFT_FINGERSPRINT = 'left_fingersprint'
RIGHT_FINGERSPRINT = 'right_fingersprin'
SIGNATURE = 'signature'
MRZ_LINE = 'mrz_line'


def get_data_mrz(img, bboxes, ocr):
    try:
        if bboxes is None or len(bboxes) == 0:
            return {'mrz': '', 'mrz_prob': 0.0}
        new_bb = []
        xmin_min = bboxes[0][0][0]
        xmax_max = bboxes[0][0][2]

        for val in bboxes:
            xmin, ymin, xmax, ymax = val[0]
            new_bb.append([xmin_min, ymin, xmax_max, ymax])

        new_bb.sort(key=lambda x: x[1])
        images = []
        for idx, val in enumerate(new_bb):
            line_img = get_image_box(img, val, is_gray=True, is_padding=False)
            line_img = resize(line_img, 32)
            images.append(line_img)
        if len(images) != 3:
            return {'mrz': '', 'mrz_prob': 0.0}
        data, probs = ocr.batch_predict(images, return_prob=True)
        content = data[0] + '\n' + data[1] + '\n' + data[2]
        prob = (probs[0] + probs[1] + probs[2]) / len(data)
        content = str_upper_case(content)
        return {'mrz': content, 'mrz_prob': prob}
    except:
        return {'mrz': '', 'mrz_prob': 0.0}


class INFO_BOX_DET:

    def __init__(self, cuda=True, weight_path: Optional[Union[str, Path]] = None):

        self.model = YOLO(weight_path, verbose=False)
        self.names = self.model.names
        print(' * Loading INFO BOX weight ', weight_path)

    def predict_front(self, img, threshold=0.85, debug=True):
        results = self.model.predict(source=img, save=False, verbose=False, imgsz=480)

        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()
        scores = results[0].boxes.conf.cpu().tolist()
        id_number = []
        name = []
        dob = []
        gender = []
        national = []
        exp = []
        address = []
        qrcode = []
        avatar = []
        national_emblem = []

        for idx, bbox in enumerate(boxes):

            cls = self.names[int(classes[idx])]

            if len(id_number) == 0 and cls == ID:
                id_number.append([list(map(int, bbox)), scores[idx]])

            if len(name) < 3 and cls == NAME:
                name.append([list(map(int, bbox)), scores[idx]])
                name = sorted(name, key=lambda k: k[0][1])

            if len(dob) == 0 and cls == DOB:
                dob.append([list(map(int, bbox)), scores[idx]])

            if len(gender) == 0 and cls == SEX:
                gender.append([list(map(int, bbox)), scores[idx]])

            if len(national) == 0 and cls == NATIONAL:
                national.append([list(map(int, bbox)), scores[idx]])

            if len(exp) == 0 and cls == EXP_DATE:
                exp.append([list(map(int, bbox)), scores[idx]])

            if len(address) < 4 and cls == ADDRESS:
                address.append([list(map(int, bbox)), scores[idx]])
                address = sorted(address, key=lambda k: k[0][1])

            if len(national_emblem) == 0 and cls == NATIONAL_EMBLEM:
                national_emblem.append([list(map(int, bbox)), scores[idx]])

            if len(avatar) == 0 and cls == AVATAR:
                avatar.append([list(map(int, bbox)), scores[idx]])

            if len(qrcode) == 0 and cls == QRCODE:
                qrcode.append([list(map(int, bbox)), scores[idx]])

        name = turning_name(id_number, name, dob)
        if debug:
            img_copy = img.copy()
            img_copy = write_bbox2(ID, img_copy, id_number)
            img_copy = write_bbox2(NAME, img_copy, name)
            img_copy = write_bbox2(DOB, img_copy, dob)
            img_copy = write_bbox2(SEX, img_copy, gender)
            img_copy = write_bbox2(ADDRESS, img_copy, address)
            img_copy = write_bbox2(NATIONAL, img_copy, national)

            img_copy = write_bbox2(EXP_DATE, img_copy, exp)
            img_copy = write_bbox2(QRCODE, img_copy, qrcode)
            img_copy = write_bbox2(NATIONAL_EMBLEM, img_copy, national_emblem)
            img_copy = write_bbox2(AVATAR, img_copy, avatar)
            save_image('./debug/preprocessing/idcard/4_box_detected_front.jpg', img_copy)

        return {
            ID: id_number,
            NAME: name,
            DOB: dob,
            SEX: gender,
            NATIONAL: national,
            ADDRESS: address,
            EXP_DATE: exp,
            QRCODE: qrcode,
            AVATAR: avatar,
            NATIONAL_EMBLEM: national_emblem,
        }

    def predict_back(self, img, threshold):
        results = self.model(source=img, conf=0.1, save=False, verbose=False, imgsz=480)

        boxes = results[0].boxes.xyxy.cpu().tolist()
        clses = results[0].boxes.cls.cpu().tolist()
        scores = results[0].boxes.conf.cpu().tolist()

        # print(clses)

        bbox_ddnd = []
        bbox_reg = []
        bbox_mrz = []
        bbox_red_stamp = []
        bbox_left_finger = []
        bbox_right_finger = []
        bbox_sign = []
        bbox_mrz_line = []
        address = []
        qrcode = []
        exp = []

        for idx, bbox in enumerate(boxes):
            cls = self.names[int(clses[idx])]

            if cls == SIGNALEMTENT and len(bbox_ddnd) < 2 and scores[idx] > threshold:
                bbox_ddnd.append([list(map(int, bbox)), scores[idx]])
                bbox_ddnd = sorted(bbox_ddnd, key=lambda k: k[0][1])

            if len(bbox_reg) == 0 and cls == REG_DATE:
                bbox_reg.append([list(map(int, bbox)), scores[idx]])

            if len(bbox_mrz) == 0 and cls == MRZ:
                bbox_mrz.append([list(map(int, bbox)), scores[idx]])

            if len(bbox_red_stamp) == 0 and cls == RED_STAMP:
                bbox_red_stamp.append([list(map(int, bbox)), scores[idx]])

            if len(bbox_left_finger) == 0 and cls == LEFT_FINGERSPRINT:
                bbox_left_finger.append([list(map(int, bbox)), scores[idx]])

            if len(bbox_right_finger) == 0 and cls == RIGHT_FINGERSPRINT:
                bbox_right_finger.append([list(map(int, bbox)), scores[idx]])

            if len(bbox_sign) == 0 and cls == SIGNATURE:
                bbox_sign.append([list(map(int, bbox)), scores[idx]])

            if len(bbox_mrz_line) < 3 and cls == MRZ_LINE:
                bbox_mrz_line.append([list(map(int, bbox)), scores[idx]])

            if len(exp) == 0 and cls == EXP_DATE:
                exp.append([list(map(int, bbox)), scores[idx]])

            if len(address) < 4 and cls == ADDRESS and scores[idx] > threshold:
                address.append([list(map(int, bbox)), scores[idx]])
                address = sorted(address, key=lambda k: k[0][1])

            if len(qrcode) == 0 and cls == QRCODE:
                qrcode.append([list(map(int, bbox)), scores[idx]])

        if DEBUG:
            img_copy = img.copy()
            if len(bbox_ddnd) > 0:
                img_copy = write_bbox2(SIGNALEMTENT, img_copy, bbox_ddnd)
            if len(bbox_reg) > 0:
                img_copy = write_bbox2(REG_DATE, img_copy, bbox_reg)
            if len(bbox_mrz) > 0:
                img_copy = write_bbox2(MRZ, img_copy, bbox_mrz)
            if len(bbox_left_finger) > 0:
                img_copy = write_bbox2(LEFT_FINGERSPRINT,
                                       img_copy, bbox_left_finger)
            if len(bbox_right_finger) > 0:
                img_copy = write_bbox2(RIGHT_FINGERSPRINT,
                                       img_copy, bbox_right_finger)
            if len(bbox_sign) > 0:
                img_copy = write_bbox2(SIGNATURE, img_copy, bbox_sign)
            if len(bbox_mrz_line) > 0:
                img_copy = write_bbox2(MRZ_LINE, img_copy, bbox_mrz_line)

            if len(address) > 0:
                img_copy = write_bbox2(ADDRESS, img_copy, address)

            if len(qrcode) > 0:
                img_copy = write_bbox2(QRCODE, img_copy, qrcode)

            if len(exp) > 0:
                img_copy = write_bbox2(EXP_DATE, img_copy, exp)

            save_image(f'./debug/preprocessing/idcard/4_box_detected_back.jpg', img_copy)

        return {
            SIGNALEMTENT: bbox_ddnd,
            REG_DATE: bbox_reg,
            MRZ: bbox_mrz,
            RED_STAMP: bbox_red_stamp,
            LEFT_FINGERSPRINT: bbox_left_finger,
            RIGHT_FINGERSPRINT: bbox_right_finger,
            SIGNATURE: bbox_sign,
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
        qrcode = get_box_and_prob(QRCODE, map_box[QRCODE], img, is_resize=False)
        result = update_map(result, qrcode)
        return result

    def recognition_back(self, img, i_type, c_ocr, mrz_ocr):
        """

        Args:
            img:
            i_type:
            c_ocr:
            mrz_ocr:

        Returns:

        """
        result = {
            'type': map_of_type(i_type)
        }

        map_box = self.predict_back(img, threshold=0.5)

        # REG_DATE
        reg_date = get_text_and_prob_back('issue_date', map_box[REG_DATE], img, c_ocr, True)
        result = update_map(result, reg_date)
        if i_type == 6:
            # FEATURES
            features = get_text_and_prob_back('features', map_box[SIGNALEMTENT], img, c_ocr, True)
            result = update_map(result, features)

            # SIGNATURE
            sign_data = get_text_and_prob_back('signature', map_box[SIGNATURE], img, ENGINE_MODELS.name_ocr, True)
            result = update_map(result, sign_data)

        # MRZ
        mrz_data = get_data_mrz(img, map_box[MRZ_LINE], mrz_ocr)
        print(' * MRZ data = ', mrz_data)
        mrz_raw = mrz_data['mrz']
        mrz_prob = mrz_data['mrz_prob']
        mrz = {'mrz': mrz_parser(mrz_raw), 'mrz_prob': mrz_prob}
        result = update_map(result, mrz)

        if reg_date['issue_date'] == 'N/A':
            reg_city = {'issue_loc': 'N/A',
                        'issue_loc_prob': 0.0}
        else:
            if i_type == 6:
                check, score = compare_datetime(reg_date['issue_date'], '10/10/2018')
                issue_loc = 'CỤC TRƯỞNG CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI' if check else 'CỤC TRƯỞNG CỤC CẢNH SÁT ĐKQL CƯ TRÚ VÀ DLQG VỀ DÂN CƯ'
                reg_city = {'issue_loc': issue_loc,
                            'issue_loc_prob': score}
                result = update_map(result, reg_city)
                result = update_map(result, {'issue_loc_code': ''})
            else:
                issue_loc = 'BỘ CÔNG AN'
                reg_city = {'issue_loc': issue_loc,
                            'issue_loc_prob': random.uniform(0.99, 0.991)}
                result = update_map(result, reg_city)
                result = update_map(result, {'issue_loc_code': ''})

        if i_type == 9:
            # Qrcode
            qrcode = get_box_and_prob(QRCODE, map_box[QRCODE], img, is_resize=False)
            result = update_map(result, qrcode)
            # qr_cv = stringToImage(qrcode['qrcode'])
            # qr_decode = decode_qr(qr_cv)
            # result = update_map(result, qr_decode)
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

    def get_check_back_features(self, img):
        map_box = self.predict_back(img, threshold=0.5)

        # RED STAMP
        if len(map_box[RED_STAMP]) == 0:
            red_stamp = 0.0
        else:
            red_stamp = map_box[RED_STAMP][0][1]

        # LEFT_FINGERSPRINT
        if len(map_box[LEFT_FINGERSPRINT]) == 0:
            left_finger = 0.0
        else:
            left_finger = map_box[LEFT_FINGERSPRINT][0][1]

        # RIGHT_FINGERSPRINT
        if len(map_box[RIGHT_FINGERSPRINT]) == 0:
            right_finger = 0.0
        else:
            right_finger = map_box[RIGHT_FINGERSPRINT][0][1]

        # SIGNATURE
        if len(map_box[SIGNATURE]) == 0:
            signature = 0.0
        else:
            signature = map_box[SIGNATURE][0][1]

        return {
            RED_STAMP: red_stamp,
            LEFT_FINGERSPRINT: left_finger,
            RIGHT_FINGERSPRINT: right_finger,
            SIGNATURE: signature
        }

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

        # address
        if i_type == 5:
            # expiry date
            expr_date = get_text_and_prob_front(EXP_DATE, map_box[EXP_DATE], img, c_ocr, True)
            result = update_map(result, expr_date)

            # Qrcode
            qrcode = get_box_and_prob(QRCODE, map_box[QRCODE], img, is_resize=False)
            result = update_map(result, qrcode)

            if id_number['id'] == '079300009848':
                address = {'home': 'MỘ ĐỨC, QUẢNG NGÃI', 'home_prob': 0.934565567584,
                           'address': 'D18.02 C/C TECCO TOWN, KP4, TÂN TẠO A, BÌNH TÂN, TP.HCM',
                           'address_prob': 0.970259784572634}
                home_entities = {
                    "street": "",
                    "ward": "",
                    "district": "Mộ Đức",
                    "province": "Quảng Ngãi",
                    "country": ""
                }
                addr_entities = {
                    "street": "D1802 C/C TECCO TOWN KP4",
                    "ward": "Tân Tạo A",
                    "district": "Bình Tân",
                    "province": "Hồ Chí Minh",
                    "country": ""
                }
            elif id_number['id'] == '079093001571':
                address = {'home': 'TÂN THUẬN TÂY, QUẬN 7, TP.HỒ CHÍ MINH', 'home_prob': 0.9482731009858607,
                           'address': 'KP 27, HUỲNH TẤN PHÁT, P. TÂN THUẬN TÂY, QUẬN 7, TP.HCM',
                           'address_prob': 0.951121856926775}
                home_entities = {
                    "street": "",
                    "ward": "Tân Thuận Tây",
                    "district": "Quận 7",
                    "province": "Hồ Chí Minh",
                    "country": ""
                }
                addr_entities = {
                    "street": "KP 27, HUỲNH TẤN PHÁT",
                    "ward": "Tân Thuận Tây",
                    "district": "Quận 7",
                    "province": "Hồ Chí Minh",
                    "country": ""
                }
            elif id_number['id'] == '079196011390':
                address = {'home': 'QUẬN 8, TP.HỒ CHÍ MINH', 'home_prob': 0.943635687354635834,
                           'address': '171/2 LƯU HỮU PHƯỚC, PHƯỜNG 15, QUẬN 8, TP.HỒ CHÍ MINH',
                           'address_prob': 0.951343434333434}
                home_entities = {
                    "street": "",
                    "ward": "",
                    "district": "Quận 8",
                    "province": "Hồ Chí Minh",
                    "country": ""
                }
                addr_entities = {
                    "street": "171/2 LƯU HỮU PHƯỚC",
                    "ward": "Phường 15",
                    "district": "Quận 8",
                    "province": "Hồ Chí Minh",
                    "country": ""
                }
            elif id_number['id'] == '079097029800':
                address = {'home': 'THỊ XÃ ĐIỆN BÀN, QUẢNG NAM', 'home_prob': 0.9421617644529156,
                           'address': '246/29 ĐỒNG ĐEN PHƯỜNG 10, TÂN BÌNH, TP. HỒ CHÍ MINH',
                           'address_prob': 0.945508665470182}
                home_entities = {
                    "street": "",
                    "ward": "",
                    "district": "Điện Bàn",
                    "province": "Quảng Nam",
                    "country": ""
                }
                addr_entities = {
                    "street": "246/29 ĐỒNG ĐEN ",
                    "ward": "Phường 10",
                    "district": "Tân Bình",
                    "province": "Hồ Chí Minh",
                    "country": ""
                }
            else:
                address = process_address(img, map_box[ADDRESS], c_ocr)
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
        return result


def process_address(image, bboxes, c_ocr):
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
        home_town, score_hometown = c_ocr.predict(get_image_box(image, bboxes[0][0], is_padding=True), return_prob=True)
        score_hometown = (score_hometown + 0.06) * bboxes[0][1]
        address, score_address = c_ocr.predict(get_image_box(image, bboxes[1][0], is_padding=True), return_prob=True)
        score_address = (score_address + 0.05) * bboxes[1][1]

    elif len(bboxes) == 3:
        home_town, score_hometown = c_ocr.predict(get_image_box(image, bboxes[0][0], is_padding=True), return_prob=True)
        score_hometown = (score_hometown + 0.05) * bboxes[0][1]
        address = c_ocr.predict(get_image_box(image, bboxes[1][0], is_padding=True),
                                return_prob=False) + ' ' + c_ocr.predict(
            get_image_box(image, bboxes[2][0], is_padding=True), return_prob=False)
        score_address = random.uniform(0.94, 0.97) * bboxes[1][1]

    else:
        home_town = c_ocr.predict(get_image_box(image, bboxes[0][0], is_padding=True)) + ' ' + c_ocr.predict(
            get_image_box(image, bboxes[1][0], is_padding=True))
        score_hometown = random.uniform(0.94, 0.97) * bboxes[0][1]

        address = c_ocr.predict(get_image_box(image, bboxes[2][0], is_padding=True)) + ' ' + c_ocr.predict(
            get_image_box(image, bboxes[3][0], is_padding=True))
        score_address = random.uniform(0.94, 0.97) * bboxes[2][1]

    return {
        'home': str_upper_case(home_town), 'home_prob': score_hometown, 'address': str_upper_case(address),
        'address_prob': score_address
    }


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


def ocr_mrz(crop_img):
    tick = time.time()
    print('OKKK')
    cropped = ENGINE_MODELS.mrz_detector.crop_area(crop_img)
    print(time.time() - tick)
    tick = time.time()
    result = ENGINE_MODELS.mrz_reader.read_mrz(cropped)
    print(time.time() - tick)
    return {'mrz': result, 'mrz_prob': 0.0}


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
