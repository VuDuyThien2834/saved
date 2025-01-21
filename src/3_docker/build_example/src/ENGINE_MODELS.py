from readmrz import MrzDetector, MrzReader

from engine._core.address_parser.address_parser import ADDRESS_PARSER
from engine.idcard.cache.cache_map import CACHE
from engine.idcard.spoofing_check.glare_det import GLARE_DET
from engine._core.image_ocr.ocr_machine import OCR_ENGINE
from src.infor_detector.infor_box_all import INFO_BOX_DET

# rotate textbox
from src.utils import clear_cache
from src.align_card import AlignCard
from src.card_cropper import CardCropper
from src.card_rotator import CardRotator
import warnings
from src.spoofing_check.DeviceDetection import DeviceDetection
from src.spoofing_check.avatarReplacementDetection import avatarReplacementDetection
from src.spoofing_check.obs_det.IDCard_Obstructed_Detection import IDCard_Obstructed_Detection
from src.spoofing_check.recapture.recapture_check import RECAPTURE_CHECK

warnings.filterwarnings("ignore", category=UserWarning)

address_parser = ADDRESS_PARSER()
cache = CACHE()

id_ocr = OCR_ENGINE(config_name='./models/idcard_models/ocr/configs/seq_id_210322.yml',
                    weight='./models/idcard_models/ocr/weights/seq_id_210322.pth')

all_chars_ocr = OCR_ENGINE(config_name='./models/idcard_models/ocr/configs/seq_allchars.yml',
                           weight='./models/idcard_models/ocr/weights/seq_allchars.pth')

name_ocr = OCR_ENGINE(config_name='./models/idcard_models/ocr/configs/seq_name_010422.yml',
                      weight='./models/idcard_models/ocr/weights/seq_name_010422.pth')

# comm_0cr = OCR_ENGINE(weight='vgg_seq2seq')

mrz_ocr = OCR_ENGINE(config_name='./models/idcard_models/ocr/configs/mrz_26072024.yml',
                     weight='./models/idcard_models/ocr/weights/mrz_26072024.pth')

align_card = AlignCard('./models/idcard_models/obstructed_card/front_corners_detection_model.pt',
                       './models/idcard_models/obstructed_card/back_corners_detection_model.pt')

cropper = CardCropper(n_ngon_model_file='./models/idcard_models/cropper/type_ngon.pt',
                      corner_model_file='./models/idcard_models/cropper/corners_new.pt',
                      rotator_model_file='./models/idcard_models/rotator/best.pt')

# rotator = CardRotator(model_file='./models/idcard_models/rotator/best.pt')

text_box_det = INFO_BOX_DET(weight_path='./models/idcard_models/textbox/text_box_cccd_all.pt')

# text_box_det_2024 = INFO_BOX_DET_2024(weight_path='./models/idcard_models/textbox/textbox_cc2024.pt')


recapture_new = RECAPTURE_CHECK(model_1='./models/idcard_models/spoofing_check/recapture/photo_recap_best.pt')

glare_check = GLARE_DET(weight_path='./models/idcard_models/spoofing_check/photo_recap/glare_det_100_epochs.pth')

# replace_field = REPLACEMENT_DET(weight_path='./models/idcard_models/spoofing_check/photo_recap/replacements_100e.pth')

obs_detect = IDCard_Obstructed_Detection('./models/idcard_models/obstructed_card/card_obstruction_detect_model.pt')
# obs_detect = IDCard_Obstructed_Detection('./models/idcard_models/obstructed_card/best_train11.pt')

avatar_replacement_detect = avatarReplacementDetection('./models/idcard_models/obstructed_card/avatarReplacementDetect.pt')
recap_device_detect = DeviceDetection()

clear_cache()
