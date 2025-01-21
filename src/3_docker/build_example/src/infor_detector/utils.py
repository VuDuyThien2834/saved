import colorsys

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont

from src.utils import cv2_img_2_pil, padding_img

font = ImageFont.truetype("./models/idcard_models/fonts/TrixiProRegular.ttf", 18)


def write_bbox(label, image, bboxes):
    # if label == 'qrcode' or label == 'national_emblem' or label == 'avatar' or label == 'stamp' or label == 'mrz':
    #     return image
    # if not os.path.exists(f'./ocr/{label}'):
    #     os.mkdir(f'./ocr/{label}')

    if bboxes is None or len(bboxes) == 0:
        return image
    for bbox in bboxes:
        # img = get_image_box(image, bbox[0], is_padding=False, is_gray=False if label == 'id' else True)
        # name = uuid.uuid4()
        #
        # content, score = ENGINE_MODELS.all_chars_ocr.predict(padding_img(img), return_prob=True) if label != 'id' else ENGINE_MODELS.id_ocr.predict(img, return_prob=True)
        # print(label, content, score)
        # # if score < 0.85:
        # #     continue
        # cv2.imwrite(f'./ocr/{label}/{name}.jpg', img)
        # with open(f'./ocr/{label}/{name}.txt', 'w') as f:
        #     f.write(content)


        x_min, y_min, x_max, y_max = bbox[0]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                      (0, 255, 255), 2)
        image = write((x_min, y_min), font, '{}: {}'.format(label, bbox[1]), image)
    return image

def choose_color(label):
    map_color = {'id': 0,
                 's_name': 1,
                 'g_name': 2,
                 'dob': 3,
                 'sex': 4,
                 'doi': 5,
                 'doe': 6,
                 'addr': 7,
                 'iloc': 8,
                 'avatar': 9,
                 'name': 1,
                 'ward': 3,
                 'district': 4,
                 'province': 5,
                 'ethnicity': 6,
                 'nationality': 7
                 }
    try:
        return len(map_color), map_color[label] + 1
    except:
        return len(map_color), 1


def write_bbox2(label, image, bboxes):
    if bboxes is None or len(bboxes) == 0:
        return image
    image_h, image_w, _ = image.shape
    num_classes, color_choose = choose_color(label)
    hsv_tuples = [(1.0 * x / (num_classes + 1), 1., 1.) for x in range((num_classes + 1))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for idx, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox[0]
        # if not os.path.exists(f'./data_ocr_cccd/{label}'):
        #     os.makedirs(f'./data_ocr_cccd/{label}')
        # cv2.imwrite(f'./data_ocr_cccd/{label}/{uuid.uuid4()}.jpg', image[y_min:y_max, x_min:x_max])
        # return image
        fontScale = 0.4
        bbox_color = colors[color_choose]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (x_min, y_min), (x_max, y_max)
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        bbox_mess = '%s' % label
        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(image, c3, c1, bbox_color, -1)  # filled

        cv2.putText(image, bbox_mess, c1, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


def write(point, _font, text, img):
    img = cv2_img_2_pil(img)
    draw = ImageDraw.Draw(img)
    draw.text(point, text, font=_font, fill=(255, 0, 0, 0))
    img = np.array(img)
    return img


def rgb_to_gbr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def update_map(map1, map2):
    result = {**map1, **map2}
    return result


def get_image_box(image, box, is_gray=True, is_padding=True, is_resize=False, is_resize_y=False):
    """

    Args:
        image: image
        box: [xmin, ymin, x_max, ymax]
        is_gray:

    Returns:

    """
    if len(box) == 0:
        return []
    ih, iw, _ = image.shape
    x_min, y_min, x_max, y_max = box
    # print(box)
    if is_resize:
        h = (x_max - x_min) * 0.05
        w = (y_max - y_min)  * 0.05
        x_min = int(x_min - h) if int(x_min - h) > 0 else 0
        y_min = int(y_min - w) if int(y_min - w) > 0 else 0
        x_max = int(x_max + h) if int(x_max + h) < iw else iw
        y_max = int(y_max + w) if int(y_max + w) < ih else ih
    if is_resize_y:
        # x_min = int(x_min - x_min * 0.01)
        y_min = int(y_min - y_min * 0.005)
        # x_max = int(x_max + x_max * 0.01)
        y_max = int(y_max + y_max * 0.005)

    if is_padding:
        result = padding_img(image[y_min:y_max, x_min:x_max])
    else:
        result = (image[y_min:y_max, x_min:x_max])

    if is_gray:
        return cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # cv2.imwrite(f'./{uuid.uuid4()}.jpg', result)
    return result
