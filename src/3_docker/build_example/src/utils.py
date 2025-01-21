import base64
import gc
import io
import json
import math
import os
import random
import time
import uuid
from builtins import len
from math import sqrt

import cv2
import gdown
import numpy as np
import torch
from PIL import Image
import requests
import re
from datetime import datetime

import scipy.spatial.distance as distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ERROR_NOT_AN_IDCARD = 'Không phải chứng minh thư, CCCD Việt Nam.'
SEPARATOR = ','
DEBUG = True

RESULT_ERR_COD3 = {'errorCode': 3, 'errorMessage': 'Ảnh không chứa cmt/cccd/pp Việt Nam', 'data': [{}],
                   'checking_result': {}}

map_of_idcard_type = {1: 'cmt_9_mt', 2: 'cmt_9_ms', 3: 'cccd/cmt_12_mt', 4: 'cccd/cmt_12_ms', -1: 'Other',
                      5: 'cccd_chip_mt', 6: 'cccd_chip_ms', 7: 'vn_passport', 8: 'cc_2024_mt', 9: 'cc_2024_ms'}
map_of_idcard_type_details = {1: 'Mặt trước CMT 9 số', 2: 'Mặt sau chứng minh thư 9 số', 3: 'Mặt trước CCCD/ CMT 12 số',
                              4: 'Mặt sau CCCD/CMT 12 số', -1: 'Loại khác', 5: 'Mặt trước CCCD chíp',
                              6: 'Mặt sau CCCD chíp', 7: 'Passport'}

map_of_idcard_type_details2 = {1: 'IDCARD_9', 2: 'IDCARD_12', 3: 'IDCARD_CHIP', 4: 'PASSPORT', -1: 'OTHERS'}

map_of_province_new = {'Ha Noi': '001', 'Ha Giang': '002', 'Cao Bang': '004', 'Bac Kan': '006', 'Bac Can': '006',
                       'Tuyen Quang': '008', 'Lao Cai': '010', 'Dien Bien': '011', 'Lai Chau': '012', 'Son La': '014',
                       'Yen Bai': '015', 'Hoa Binh': '017', 'Thai Nguyen': '019', 'Lang Son': '020',
                       'Quang Ninh': '022', 'Bac Giang': '024', 'Phu Tho': '025', 'Vinh Phuc': '026', 'Bac Ninh': '027',
                       'Hai Duong': '030', 'Hai Phong': '031', 'Hung Yen': '033', 'Thai Binh': '034', 'Ha Nam': '035',
                       'Nam Dinh': '036', 'Ninh Binh': '037', 'Thanh Hoa': '038', 'Nghe An': '040', 'Ha Tinh': '042',
                       'Quang Binh': '044', 'Quang Tri': '045', 'Thua Thien Hue': '046', 'T.T.Hue': '046', 'Hue': '046',
                       'T.T Hue': '046', 'T T Hue': '046', 'TT Hue': '046', 'TTHue': '046', 'Da Nang': '048',
                       'Quang Nam': '049', 'Quang Ngai': '051', 'Binh Dinh': '052', 'Phu Yen': '054',
                       'Khanh Hoa': '056', 'Ninh Thuan': '058', 'Binh Thuan': '060', 'Kon Tum': '062', 'Gia Lai': '064',
                       'Dak Lak': '066', 'Dak Nong': '067', 'Lam Dong': '068', 'Binh Phuoc': '070', 'Tay Ninh': '072',
                       'Binh Duong': '074', 'Dong Nai': '075', 'Ba Ria - Vung Tau': '077', 'Ba Ria Vung Tau': '077',
                       'Ho Chi Minh': '079', 'Long An': '080', 'Tien Giang': '082', 'Ben Tre': '083', 'Tra Vinh': '084',
                       'Vinh Long': '086', 'Dong Thap': '087', 'An Giang': '089', 'Kien Giang': '091', 'Can Tho': '092',
                       'Hau Giang': '093', 'Soc Trang': '094', 'Bac Lieu': '095', 'Ca Mau': '096'}
map_of_province_old = {'Lam Dong': '25', 'Lang Son': '08', 'Lao Cai': '06', 'Long An': '30', 'Nam Dinh': '16',
                       'Nghe An': '18', 'Ninh Binh': '16', 'Ninh Thuan': '26', 'Phu Tho': '13', 'Phu Yen': '22',
                       'Quang Binh': '19', 'Quang Nam': '20', 'Q. Nam': '20', 'Q Nam': '20', 'Q.Nam': '20',
                       'Quang Ngai': '21', 'Quang Ninh': '10', 'Quang Tri': '19', 'Soc Trang': '36', 'Son La': '5',
                       'Tay Ninh': '29', 'Thai Binh': '15', 'Thai Nguyen': '090:091 ', 'Thanh Hoa': '17',
                       'Thua Thien Hue': '19', 'T.T.Hue': '19', 'T.T Hue': '19', 'T T Hue': '19', 'Hue': '19',
                       'Tien Giang': '31', 'Tra Vinh': '33', 'Tuyen Quang': '7', 'Vinh Long': '33', 'Vinh Phuc': '13',
                       'Yen Bai': '15', 'Ha Noi': '01', 'Ho Chi Minh': '02', 'An Giang': '35',
                       'Ba Ria - Vung Tau': '27', 'Ba Ria Vung Tau': '27', 'Bac Can': '95', 'Bac Kan': '95',
                       'Bac Lieu': '38', 'Bac Ninh: Bac Giang': '12', 'Ben Tre': '32', 'Binh Dinh': '21',
                       'Binh Thuan': '26', 'Ca Mau': '38', 'Can Tho': '36', 'Cao Bang': '08', 'Da Nang': '20',
                       'Dak Lak': '24', 'Dong Nai': '27', 'Dong Thap': '34', 'Gia Lai': '230:231', 'Ha Giang': '07',
                       'Ha Nam': '16', 'Binh Duong': '280', 'Ha Tay': '11', 'Ha Tinh': '18', 'Hai Duong': '14',
                       'Hai Phong': '3', 'Hau Giang': '36', 'Hoa Binh': '11', 'Hung Yen': '14', 'Kon Tum': '23',
                       'Lai Chau': '04', 'Kien Giang': '37', 'Khanh Hoa': '22', 'Binh Phuoc': '285'}
s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'

map_corners = {'top_left': 0, 'top_right': 1, 'bottom_left': 2, 'bottom_right': 3}


def get_province_code(itype, province_name, addr):
    if itype == 1 or itype == 2:
        if map_of_province_old.keys().__contains__(province_name):
            return map_of_province_old[province_name]
        for k in map_of_province_old.keys():
            if addr.__contains__(k):
                return map_of_province_old[k]
        for k in map_of_province_old.keys():
            if addr.__contains__(k):
                return map_of_province_new[k]

        return ''
    if map_of_province_new.keys().__contains__(province_name):
        return map_of_province_new[province_name]
    for k in map_of_province_new.keys():
        if addr.__contains__(k):
            return map_of_province_new[k]
    return ''


def save_image(file_path, image):
    try:
        cv2.imwrite(file_path, image)
    except:
        pass


def resize(img, base_h):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_w, img_h = img.shape[0], img.shape[1]
    if img_h > base_h:
        wpercent = float(img_h / img_w)
        wsize = int((float(base_h) * float(wpercent)))
        resized = cv2.resize(img, (wsize, base_h), interpolation=cv2.INTER_AREA)
        return resized
    return img


def remove_accents(input_str):
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s


def write_log(image, result):
    dir_log = './debug/logs/{}_{}/'.format(uuid.uuid4(), time.time())
    os.mkdir(dir_log)
    cv2.imwrite(dir_log + 'request.png', image)
    f = open(dir_log + 'result.txt', 'w')
    f.write(str(result))
    f.close()


def str_upper_case(string):
    return string.strip().lstrip().lstrip().upper()


def get_date_from_expr(expr_date):
    if str(expr_date).__contains__('N/A'):
        return 'N/A'
    try:
        match = re.search(r'\d{2}/\d{2}/\d{4}', expr_date)
        if match is None:
            match2 = re.search(r'\d{4}', expr_date)
            if match2 is not None:
                return match2.group()

            return "KHÔNG THỜI HẠN"
        date = datetime.strptime(match.group(), "%d/%m/%Y").strftime("%d/%m/%Y")
        return date
    except:
        return 'N/A'


def get_date_from_reg_date(expr_date):
    # print(expr_date)
    expr_date = re.sub("[.~!@@#$%^&*()_,+_\":]", "", expr_date)
    # print(expr_date)
    if str(expr_date).__contains__('N/A'):
        return 'N/A'
    try:
        match = re.search(r'\w+ *\d{1,2} *\w+ *\d{1,2} *\w+ \d{2,4}', expr_date)
        # print('maatch = ' + match.group())
        if match is None:
            return expr_date
        date = datetime.strptime(match.group(), "Ngày %d tháng %m năm %Y").strftime("%d/%m/%Y")
        return date
    except:
        return 'N/A'


def distance_2points(p1, p2):
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def padding_img(img, ratio=1.1):
    ht, wd, cc = img.shape
    try:
        # create new image of desired size and color (blue) for padding
        hh = int(ht)  # int(ht * ratio)
        ww = int(wd * ratio)  # int(hh * wd / ht)
        color = img[0][0]
        result = np.full((hh, ww, cc), color, dtype=np.uint8)
        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2
        # copy img image into center of result image
        result[yy:yy + ht, xx:xx + wd:] = img
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result
    except:
        print('lỗi ảnh padding...')

    return img


def getface(image, is_get_face=False):
    if is_get_face:
        try:
            filepath = './debug/tmp/{}.jpg'.format(random.random())
            cv2.imwrite(filepath, image)

            url = "http://face-api.aeyes.tech/api/faces_crop"

            payload = {}
            files = [
                ('file',
                 ('079300000037_mt.png', open(filepath, 'rb'), 'image/png'))
            ]
            headers = {}

            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            os.remove(filepath)
            d = json.loads(response.text)
            return str(d['faces'][0])
        except:
            return ''
    else:
        return ''


def prepare_for_recognition(image, points):
    bboxes = []
    cv_images = []
    file_names = []
    id = 0
    imH, imW, _ = image.shape

    mask = np.ones(image.shape, dtype=np.uint8)
    mask.fill(255)
    _, img = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)
    #
    # boundary = []
    #
    # for item in points:
    #     print(item[0])
    #     if len(item[0]) > 0:
    #         boundary.append(np.array(item[0], dtype=np.int32))
    # boundary.sort(key=lambda x: get_contour_precedence(x, img.shape[1]))

    for item in points:
        (x1, y1, x2, y2, x3, y3, x4, y4) = item
        x1 = x1 if x1 > 0 else 0
        y1 = y1 if y1 > 0 else 0
        x2 = x2 if x2 > 0 else 0
        y2 = y2 if y2 > 0 else 0
        x3 = x3 if x3 > 0 else 0
        y3 = y3 if y3 > 0 else 0
        x4 = x4 if x4 > 0 else 0
        y4 = y4 if y4 > 0 else 0

        arr = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        roi_corners = np.array(arr, dtype=np.int32)
        rect = cv2.boundingRect(roi_corners)
        x, y, w, h = rect
        croped = image[y:y + h, x:x + w].copy()
        try:
            id = id + 1
            file_names.append("word_" + str(id) + ".png")
            cv_images.append(croped)
            bboxes.append((x, y, w, h, id))

        except:
            continue

    return cv_images, file_names, imW, imH, bboxes


def map_of_type(i_type):
    return map_of_idcard_type[i_type]


def map_of_type_detail(i_type):
    return map_of_idcard_type_details[i_type]


def map_of_type_detail2(i_type):
    return map_of_idcard_type_details2[i_type]


def save_cv_img(path, img):
    cv2.imwrite(path, img)


def join_address(array):
    if array is None or len(array) == 0:
        return ''
    if len(array) == 1:
        return str(array[0])
    i = 0
    txt = ''
    while i < len(array) - 1:
        txt += str(array[i]) + SEPARATOR
        i += 1
    txt += str(array[len(array) - 1])
    return txt


def cv2_img_2_pil(cv2_img):
    tmp_img = cv2_img  #= cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(tmp_img).convert('RGB')


def download(url: str, save_path: str):
    """
    Downloads file from gdrive, shows progress.
    Example inputs:
        url: 'ftp://smartengines.com/midv-500/dataset/01_alb_id.zip'
        save_path: 'data/file.zip'
    """

    # create save_dir if not present
    create_dir(os.path.dirname(save_path))
    # download file
    gdown.download(url, save_path, quiet=False)


def convert_img_2_base64(image):
    """

    Args:
        image:

    Returns:

    """
    try:
        retval, buffer_img = cv2.imencode('.jpg', image)
        return str(base64.b64encode(buffer_img))[2:].replace('\'', '')
    except:
        # traceback.print_exc()
        return ''


# Take in base64 string and return PIL image
def stringToImage(base64_string):
    try:
        imgdata = base64.b64decode(base64_string)
        imgg = toRGB(Image.open(io.BytesIO(imgdata)))
        return imgg
    except:
        return None


def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def decode_qrcode(cv_image):
    # img = cv2_img_2_pil(cv_image)
    from PIL import Image
    # from pyzbar import pyzbar
    # my_qr_output = pyzbar.decode(img)
    return ''


def sorting_bounding_box(image, pointss, isTable=False):
    # print(pointss)
    points = list(map(lambda x: [x[0], x[1][0], x[1][2]], pointss))
    # print('pointd    = ',points)
    points_sum = list(map(lambda x: [x[0], x[1], sum(x[1]), x[2][1]], points))
    # print('points_sum=',points_sum)
    x_y_cordinate = list(map(lambda x: x[1], points_sum))
    # print('cordinate = ', x_y_cordinate)
    final_sorted_list = []
    while len(points_sum) is not 0:
        try:
            new_sorted_text = []
            new_sorted_bb = []
            initial_value_A = [i for i in sorted(enumerate(points_sum), key=lambda x: x[1][2])][0]
            #         print(initial_value_A)
            threshold_value = abs(initial_value_A[1][1][1] - initial_value_A[1][3])
            # print(threshold_value)
            threshold_value = 5
            del points_sum[initial_value_A[0]]
            del x_y_cordinate[initial_value_A[0]]
            A = [initial_value_A[1][1]]
            K = list(map(lambda x: [x, abs(x[1] - initial_value_A[1][1][1])], x_y_cordinate))
            K = [[count, i] for count, i in enumerate(K)]
            K = [i for i in K if i[1][1] <= threshold_value]
            sorted_K = list(map(lambda x: [x[0], x[1][0]], sorted(K, key=lambda x: x[1][1])))
            B = []
            points_index = []
            for tmp_K in sorted_K:
                points_index.append(tmp_K[0])
                B.append(tmp_K[1])
            # print('A = ', A)
            if len(B) == 0:
                B = A.copy()
            dist = distance.cdist(A, B)[0]
            d_index = [i for i in sorted(zip(dist, points_index), key=lambda x: x[0])]
            new_sorted_text.append(initial_value_A[1][0])
            index = []
            for j in d_index:
                new_sorted_text.append(points_sum[j[1]][0])
                index.append(j[1])
            for n in sorted(index, reverse=True):
                del points_sum[n]
                del x_y_cordinate[n]
            final_sorted_list.append(new_sorted_text)
        except Exception as e:
            print(e.with_traceback())
            break
    final_sorted_bb = []
    for line in final_sorted_list:
        new_sorted_bb = []
        for text in line:
            new_sorted_bb.append(pointss[text][1])
        final_sorted_bb.append(new_sorted_bb)

    # print('==============ss =', final_sorted_bb)

    final_sorted_bb.sort(key=take_min_y)
    # return  final_sorted_bb
    if isTable:
        new_clusters = final_sorted_bb
    else:
        new_clusters = []
        for clus in final_sorted_bb:
            indexes = []
            if len(clus) == 1:
                new_clusters.append(clus)
                continue
            sb = space_base2(image)
            for el in range(len(clus) - 1):
                if distance_x(clus[el + 1][0], clus[el][1]) >= 0.5 * sb:
                    indexes.append(el + 1)
            if len(indexes) == 0:
                new_clusters.append(clus)
            else:
                new_clusters.append(clus[0:indexes[0]])
                j = 0
                while j < len(indexes) - 1:
                    new_clusters.append(clus[indexes[j]:indexes[j + 1]])
                    j += 1
                new_clusters.append(clus[indexes[len(indexes) - 1]:len(clus)])
    # new_clusters.sort(key=top_left)
    # print(new_clusters)
    del points
    del points_sum
    del x_y_cordinate
    del final_sorted_bb
    gc.collect()
    return new_clusters


def top_left(k):
    return k[0][0][1]


def space_base(cluster):
    if len(cluster) == 0:
        return 0
    clus_choice = cluster[0]
    for clus in cluster:
        if len(clus) > 1:
            clus_choice = clus
            break
    return math.fabs(clus_choice[0][0][0] - clus_choice[0][1][0])


# def linebreak(image, polys, threshol, sb):
#     polys


def distance_x(p1, p2):
    return math.fabs(p1[0] - p2[0])


def take_min_y(elem):
    return elem[0][0][1]


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def crop_img_block(img, polys):
    w, h, _ = img.shape
    xes = []
    ies = []
    for poly in polys:
        x1, y1 = int(poly[0][0]), int(poly[0][1])
        x2, y2 = int(poly[1][0]), int(poly[1][1])
        x3, y3 = int(poly[2][0]), int(poly[2][1])
        x4, y4 = int(poly[3][0]), int(poly[3][1])

        xes.append([x1, x2, x3, x4])
        ies.append([y1, y2, y3, y4])

    top_left_x = min(np.array(xes).reshape(-1))
    top_left_y = min(np.array(ies).reshape(-1))
    bot_right_x = max(np.array(xes).reshape(-1))
    bot_right_y = max(np.array(ies).reshape(-1))
    if top_left_x < 0: top_left_x = 0
    if top_left_y < 0: top_left_y = 0

    # if bot_right_x > w: bot_right_x = w
    # if bot_right_y > h: bot_right_y = h
    return img[top_left_y:bot_right_y, top_left_x:bot_right_x]


def space_base2(image):
    return image.shape[1] / 10


# def cv2_img_2_pil(cv2_img):
#     tmp_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
#     return Image.fromarray(tmp_img)

def make_map_bboxes(bboxes):
    """

    :param bboxes:
    :return:
    """
    map_bboxes = []
    i = 0
    for value in bboxes:
        # print('value = ', value)
        try:
            map_bboxes.append([i, value.tolist()])
        except:
            map_bboxes.append([i, value])
        i += 1
    return map_bboxes


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()
