import datetime
import json
import random
import traceback
import uuid
from functools import reduce

import cv2
import numpy as np
from PIL import ImageStat

from src import ENGINE_MODELS
from src.utils import cv2_img_2_pil, clear_cache, save_image, stringToImage

MONOCHROMATIC_MAX_VARIANCE = 0.005
COLOR = 500
MAYBE_COLOR = 40


def detect_color_image(img):
    img = cv2_img_2_pil(img)
    v = ImageStat.Stat(img).var
    is_monochromatic = reduce(lambda x, y: x and y < MONOCHROMATIC_MAX_VARIANCE, v, True)
    if is_monochromatic:
        return 1, 1 - random.uniform(0.9, 0.99)
    else:
        if len(v) == 3:
            max_min = abs(max(v) - min(v))
            if max_min > COLOR:
                return 0, 1 - random.uniform(0.99, 1.0)
            elif max_min > MAYBE_COLOR:
                return 0, 1 - random.uniform(0.8, 0.91)
            else:
                return 1, random.uniform(0.95, 0.98999)
        elif len(v) == 1:
            return 1, random.uniform(0.95, 0.98999)
        else:
            return 1, 1 - random.uniform(0.7, 0.9)


def check_expr_date(info, type):
    if type == 2:
        try:
            if info['issue_date'] is None or len(info['issue_date']) == 0:
                return {'check_issue_date_result': '1', 'check_issue_date_prob': 1.0,
                        'check_issue_date_details': 'misssing doe'}
            exp_date_str = str(info['issue_date'])
            dd = int(exp_date_str.split('/')[0])
            mm = int(exp_date_str.split('/')[1])
            yyyy = int(exp_date_str.split('/')[2])
            yyyy_check = int(yyyy) + 15
            date_now = datetime.datetime.today()
            new_date = datetime.datetime(yyyy_check, mm, dd)
            if new_date > date_now:
                return {'check_issue_date_result': '0', 'check_issue_date_prob': info['issue_date_prob'],
                        'check_issue_date_details': new_date.strftime('%d/%m/%Y')}
            else:
                return {'check_issue_date_result': '1', 'check_issue_date_prob': info['issue_date_prob'],
                        'check_issue_date_details': new_date.strftime('%d/%m/%Y')}
        except:
            return {'check_issue_date_result': '1', 'check_issue_date_prob': 1.0,
                    'check_issue_date_details': 'missing doe'}
    if type == 3 or type == 5 or type == 7:
        try:
            if info['doe'] is None or len(info['doe']) == 0:
                return {'check_issue_date_result': False, 'check_issue_date_prob': '1',
                        'check_issue_date_details': 'misssing doe'}
            exp_date_str = str(info['doe'])
            # print('exp_date_str', exp_date_str)
            date_now = datetime.datetime.today()
            try:
                dd = int(exp_date_str.split('/')[0])
                mm = int(exp_date_str.split('/')[1])
                yyyy = int(exp_date_str.split('/')[2])
                date_now = datetime.datetime.today()
                new_date = datetime.datetime(yyyy, mm, dd)
            except:
                new_date = datetime.datetime(2020, 1, 1)
            if new_date > date_now:
                return {'check_issue_date_result': '0', 'check_issue_date_prob': info['doe_prob'],
                        'check_issue_date_details': new_date.strftime('%d/%m/%Y')}
            else:
                return {'check_issue_date_result': '1', 'check_issue_date_prob': info['doe_prob'],
                        'check_issue_date_details': new_date.strftime('%d/%m/%Y')}
        except:
            traceback.print_exc()
            return {'check_issue_date_result': '1', 'check_issue_date_prob': 1.0,
                    'check_issue_date_details': 'missing doe'}

    return {'check_issue_date_result': '1', 'check_issue_date_prob': 1.0, 'check_issue_date_details': 'missing doe'}


def checking(crop_image, raw_image, info, check, tl_br,
             i_type, device_type, full_image):
    if check == '':
        return {}
    check_photocopied, check_corner_cut, \
        check_emblem, check_stamp, check_avatar, \
        check_replacement_avatar, check_recaptured, \
        check_expiry_date, check_red_stamp, check_embossed_stamp, \
        check_rfp, check_lfp, check_glare, check_frame, check_replacement_field = load_check(check)

    result = {}

    if check_corner_cut:
        result.update(
            {'corner_cut_result': '0', 'corner_cut_prob': [0.0, 0.0, 0.0, 0.0]})

    if i_type == 6:
        data = ENGINE_MODELS.text_box_det.get_check_back_features(crop_image)
    else:
        data = ENGINE_MODELS.text_box_det.get_check_features_front(crop_image)
    # print(' * Check features: ', data)
    if check_emblem and i_type == 5:
        national_emblem = data['national_emblem_prob']
        if national_emblem > 0.9:
            result.update(
                {'check_national_emblem_result': '0', 'check_national_emblem_prob': 1 - float(national_emblem)})
        else:
            result.update({'check_national_emblem_result': '1',
                           'check_national_emblem_prob': 1 - float(national_emblem)})

    if check_red_stamp and i_type == 6:
        red_stamp_score = data['red_stamp']
        if red_stamp_score > 0.95:
            result.update({'check_red_stamp_result': '0', 'check_red_stamp_prob': 1 - float(red_stamp_score)})
        else:
            result.update({'check_red_stamp_result': '1', 'check_red_stamp_prob': 1 - float(red_stamp_score)})

    aligned_card = ENGINE_MODELS.align_card.align_card(raw_image, i_type, device_type=device_type)

    if check_rfp and check_lfp and i_type == 6:
        fps_obstruct_result = ENGINE_MODELS.obs_detect.infer_check_if_obstruct_fingerprint(aligned_card)

        # right fingerprint
        if fps_obstruct_result[1][0] == 1:
            result.update({'check_rfp_result': '1', 'check_rfp_prob': round(fps_obstruct_result[1][1], 4)})
        else:
            result.update({'check_rfp_result': '0', 'check_rfp_prob': 1 - round(fps_obstruct_result[1][1], 4)})

        # left fingerprint
        if fps_obstruct_result[0][0] == 1:
            result.update({'check_lfp_result': '1', 'check_lfp_prob': round(fps_obstruct_result[0][1], 4)})
        else:
            result.update({'check_lfp_result': '0', 'check_lfp_prob': 1 - round(fps_obstruct_result[0][1], 4)})

    if check_glare:
        glares = ENGINE_MODELS.glare_check.glare(crop_image)
        if len(glares) == 0:
            result.update({'check_glare_result': "0", 'check_glare_prob': 0.0})
        else:
            result.update({'check_glare_result': "1", 'check_glare_prob': float(glares[0][1])})


    if check_avatar and (i_type == 5 or i_type == 8):
        avatar_prob = data['avatar_prob']

        if avatar_prob > 0.95:
            result.update({'check_avatar_result': '0', 'check_avatar_prob': 1 - float(avatar_prob)})
        else:
            result.update({'check_avatar_result': '1', 'check_avatar_prob': 1 - float(avatar_prob)})

    if check_replacement_avatar and (i_type == 5 or i_type == 8):
        try:
            avatar = stringToImage(data['avatar'])
            cv2.imwrite('/home/shared/FPT/projects/z_06_SHB/release/ai_ocr/avatar.jpg', avatar)
            w, h = avatar.shape[:2]
            if w/h > 1.5 or w/h < 1.2:
                result.update({'check_replacement_avatar_result': "1",
                               'check_replacement_avatar_prob': 1.0})
            else:
                _856_540_crop_image = cv2.resize(crop_image, (856, 540)) 

                avatar_replacement_result = ENGINE_MODELS.avatar_replacement_detect.infer(_856_540_crop_image)

                if avatar_replacement_result[0] == 0:
                    result.update({'check_replacement_avatar_result': "0",
                                   'check_replacement_avatar_prob': 1 - avatar_replacement_result[1]})
                else:
                    result.update(
                        {'check_replacement_avatar_result': "1",
                         'check_replacement_avatar_prob': avatar_replacement_result[1]})
        except:
            pass



    labels, probs = ENGINE_MODELS.recapture_new.yl_predict(crop_image)

    if check_photocopied:
        photocopied_rs, photocopied_prob = get_label_prob(labels, probs, 2)
        if photocopied_rs == 2:
            result.update({'check_photocopied_result': '1', 'check_photocopied_prob': photocopied_prob})
        else:
            result.update({'check_photocopied_result': '0', 'check_photocopied_prob': photocopied_prob})

    if check_replacement_field:
        # _class, conf = ENGINE_MODELS.obs_detect.infer(aligned_card)

        _856_540_crop_image = cv2.resize(crop_image, (856, 540)) 
        _class, conf = ENGINE_MODELS.obs_detect.infer(_856_540_crop_image)

        if _class == 0:
            result.update({'check_replacement_field_result': '0', 'check_replacement_field_result_prob': 1 - conf})
        else:
            result.update({'check_replacement_field_result': '1', 'check_replacement_field_result_prob': conf})

    if check_recaptured:
        device_overlap_cls_id = 0

        if full_image is not None:
            id__ = uuid.uuid4()
            save_image(f'./debug/{id__}_full_image.jpg', full_image)
            save_image(f'./debug/{id__}_image.jpg', raw_image)
            device_overlap_cls_id, overlap_rate = ENGINE_MODELS.recap_device_detect.infer(full_image, cropTime_threshold=10, crop_rate=0.9, i_type=i_type)

            if device_overlap_cls_id == 1:
                result.update({'recaptured_result': '1', 'recaptured_result_prob': overlap_rate})

        if full_image is None or device_overlap_cls_id == 0:
            recapture_rs, recapture_prob = get_label_prob(labels, probs, 1)
            if recapture_rs == 1:
                result.update({'recaptured_result': '1', 'recaptured_result_prob': recapture_prob})
            else:
                result.update({'recaptured_result': '0', 'recaptured_result_prob': recapture_prob})

    if check_expiry_date and i_type == 5:
        dta = check_expr_date(info, i_type)
        result.update(dta)
    print('OK')
    if check_frame and device_type == '0':
        tl, br = tl_br
        xmin, ymin, xmax, ymax = tl[0], tl[1], br[0], br[1]
        h, w = raw_image.shape[0], raw_image.shape[1]
        base = min(h, w) * 0.001
        if xmin < base or xmax > w - base or ymin < base or ymax > h - base:
            result.update({'on_frame_result': '1', 'on_frame_prob': random.uniform(0.95, 0.99)})
        else:
            result.update({'on_frame_result': '0', 'on_frame_prob': 1 - random.uniform(0.95, 0.99)})

    clear_cache()
    return result


def get_label_prob(labels, confs, cls):
    for idx, lb in enumerate(labels):
        if lb == cls:
            return labels[0], confs[idx]

    return 0, 0.0


def load_check(check):
    check_photocopied = False
    check_corner_cut = False
    check_emblem = False
    check_stamp = False
    check_avatar = False
    check_replacement_avatar = False
    check_recaptured = False
    check_exprity_date = False
    check_red_stamp = False
    check_embossed_stamp = False
    check_rfp = False
    check_lfp = False
    check_glare = False
    check_frame = False
    check_replacement_field = False

    try:
        check = json.loads(check)
        if 'check_photocopied' in check.keys() and check['check_photocopied'] is True:
            check_photocopied = True
        if 'check_corner_cut' in check.keys() and check['check_corner_cut'] is True:
            check_corner_cut = True
        if 'check_emblem' in check.keys() and check['check_emblem'] is True:
            check_emblem = True
        if 'check_stamp' in check.keys() and check['check_stamp'] is True:
            check_stamp = True
        if 'check_avatar' in check.keys() and check['check_avatar'] is True:
            check_avatar = True
        if 'check_replacement_avatar' in check.keys() and check['check_replacement_avatar'] is True:
            check_replacement_avatar = True
        if 'check_recaptured' in check.keys() and check['check_recaptured'] is True:
            check_recaptured = True
        if 'check_exprity_date' in check.keys() and check['check_exprity_date'] is True:
            check_exprity_date = True
        if 'check_embossed_stamp' in check.keys() and check['check_embossed_stamp'] is True:
            check_embossed_stamp = True
        if 'check_red_stamp' in check.keys() and check['check_red_stamp'] is True:
            check_red_stamp = True
        if 'check_rfp' in check.keys() and check['check_rfp'] is True:
            check_rfp = True
        if 'check_lfp' in check.keys() and check['check_lfp'] is True:
            check_lfp = True
        if 'check_glare' in check.keys() and check['check_glare'] is True:
            check_glare = True
        if 'check_frame' in check.keys() and check['check_frame'] is True:
            check_frame = True
        if 'check_replacement_field' in check.keys() and check['check_replacement_field'] is True:
            check_replacement_field = True
    except:
        print('Error load checking option')

    return check_photocopied, check_corner_cut, check_emblem, check_stamp \
        , check_avatar, check_replacement_avatar, check_recaptured, \
        check_exprity_date, check_red_stamp, check_embossed_stamp, check_rfp, check_lfp, check_glare, check_frame, check_replacement_field


def check_replace_avatar(gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=80)
    if lines is None:
        return '0', random.uniform(0.05, 0.1)
    a, b, c = lines.shape
    print('* Lines of cover avatar = ', a)
    if a < 1:
        return '0', random.uniform(0.05, 0.1)
    if a <= 2:
        return '0', random.uniform(0.1, 0.2)
    if a == 3:
        return '1', random.uniform(0.5, 0.75)
    if a == 4:
        return '1', random.uniform(0.7, 0.8)

    return '1', random.uniform(0.9, 0.95)


# def check_color(image):
#     image_c = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
#     commutative_image_diff = get_image_difference(image, image_c)
#     cv2.imwrite('./dfsdfds.jpg', image)
#     cv2.imwrite('./dfsdfds2.jpg', image_c)
#
#     if commutative_image_diff < 0.05:
#         return 1, 1 - commutative_image_diff
#     return 0, commutative_image_diff


# def get_image_difference(image_1, image_2):
#     first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
#     second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])
#
#     img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
#     img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
#     img_template_diff = 1 - img_template_probability_match
#
#     # taking only 10% of histogram diff, since it's less accurate than template method
#     commutative_image_diff = (img_hist_diff / 10) + img_template_diff
#     return commutative_image_diff

# if __name__ == '__main__':
#     image1 = cv2.imread('/home/thienbd/Desktop/221669603_882144019376692_8334891678959586996_n.jpg')
#     # image2 = cv2.cvtColor(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
#     image_difference = check_color(image1)
#     print(image_difference)
