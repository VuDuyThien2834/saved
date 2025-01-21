import traceback

from idcard.spoofing_check.checking import checking
from src import ENGINE_MODELS
from src.utils import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def recognize(img, check, device_type='0', full_image=(), get_avatar=False, get_qrcode=False):
    """
    Returns:
    """
    save_image(f'./debug/preprocessing/idcard/1_raw_image.jpg', img)

    i_type, score, polygon = ENGINE_MODELS.cropper.get_type_polygon(img)
    print(' * Type of card: ', i_type)
    if i_type is None:
        result = {'errorCode': 6, 'errorMessage': 'Không chứa cmt/cccd, vui lòng thực hiện lại', 'data': []}
        return json.dumps(result, indent=2).encode('utf8')

    if i_type != 5 and i_type != 6 and i_type != 8 and i_type != 9:
        status, data = recognition_card(1, None, img, check)

    else:
        if i_type == 8 or i_type == 9:
            save_image(f'./debug/{uuid.uuid4()}_{i_type}.jpg', img)
        pos_img, (tl, br) = ENGINE_MODELS.cropper.crop_idcard(img, device_type=device_type)
        if pos_img is None:
            result = {'errorCode': 5, 'errorMessage': 'Không phát hiện 4 góc cccd, vui lòng thực hiện lại', 'data': []}
            return json.dumps(result, indent=2).encode('utf8')

        print(f' * Crop object... - Type: {i_type}, Score: {score}')
        save_image(f'./debug/preprocessing/idcard/2_cropped.jpg', pos_img)

        status, data = recognition_card(i_type, cv2.resize(pos_img, (1000, 600), interpolation=cv2.INTER_AREA), img,
                                        check)

        try:
            if not get_avatar and 'avatar' in data:
                data.pop('avatar')
                data.pop('avatar_prob')
            if not get_qrcode and 'qrcode' in data:
                data.pop('qrcode')
                data.pop('qrcode_prob')
            if i_type == 5 or i_type == 6 or i_type == 8 or i_type == 9:
                data['checking_result'] = checking(pos_img, img, data, check, (tl, br), i_type, device_type, full_image)
        except Exception as e:
            print(e, flush=True)

    result = {'errorCode': status[0], 'errorMessage': status[1], 'data': [data]}
    print('============================================================')
    clear_cache()
    return json.dumps(result, indent=2).encode('utf8')


def save_request(img):
    pil_raw = f'./debug/preprocessing/idcard/temp/{uuid.uuid4()}.jpg'
    cv2.imwrite(pil_raw, img)
    return pil_raw


def recognize_text_and_crop_front(card_type, post_image):
    """
    Args:
        card_type:
        post_image:
    Returns:
    """
    data = ENGINE_MODELS.text_box_det.recognition_front(post_image, card_type, ENGINE_MODELS.id_ocr,
                                                        ENGINE_MODELS.all_chars_ocr)

    if data['id'] is None or len(data.get('id')) != 12:
        status = (1, 'Sai số cmt/cccd')
    else:
        status = (0, '')
    return status, data, None


def recognition_card(i_type, pos_img, raw_image, check):
    if i_type == 5 or i_type == 8:
        status, data, _ = recognize_text_and_crop_front(i_type, pos_img)
    elif i_type == 6 or i_type == 9:
        status, data = recognize_text_and_crop_end(i_type, pos_img)
    else:
        temp_path = f'./debug/preprocessing/idcard/temp/error_{uuid.uuid4()}.jpg'
        try:
            cv2.imwrite(temp_path, raw_image)
            response = call_1234_api(temp_path, check)
            status = (0, '')
            data = response['data'][0]
        except Exception as e:
            status = (2, 'Không phải cmt/cccd Việt Nam')
            data = {}
        finally:
            os.remove(temp_path)

    return status, data


def call_1234_api(image_path, check):
    import requests

    url = "http://103.191.147.17:8888/api/recognition"

    payload = {
        'check': check}
    files = [
        ('file', ('445374161_419194391106019_7101138333232030185_n.jpg',
                  open(image_path, 'rb'),
                  'image/jpeg'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files, timeout=20)

    return json.loads(response.text)


def recognize_text_and_crop_end(i_type, origin_image):
    """

    Args:
        i_type:
        origin_image:

    Returns:

    """
    data = ENGINE_MODELS.text_box_det.recognition_back(
            origin_image, i_type, ENGINE_MODELS.all_chars_ocr, ENGINE_MODELS.mrz_ocr)


    return (0, ''), data


# def convert_img_2_base64(image):
#     """
#
#     Args:
#         image:
#
#     Returns:
#
#     """
#     try:
#         retval, buffer_img = cv2.imencode('.jpg', image)
#         return str(base64.b64encode(buffer_img))[2:].replace('\'', '')
#     except:
#         traceback.print_exc()
#         return None
