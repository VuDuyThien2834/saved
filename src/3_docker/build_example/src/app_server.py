import json
import numpy as np
import cv2
from flask import Flask, request
import traceback
from flask_cors import cross_origin
from engine._core.address_parser.address_parser import ADDRESS_PARSER
from src import ENGINE_MODELS
from src.new_engine import recognize

app = Flask(__name__)
app.config["DEBUG"] = True
address_parser = ADDRESS_PARSER()


@app.route("/")
def index():
    return "Nice to meet you!"


@app.route('/api/recognition', methods=['POST'])
@cross_origin(origin='*')
def recognition():
    try:
        file = request.files['file'].read()
        try:
            check = str(request.form['check'])
        except:
            check = ''

        try:
            get_avatar = str(request.form['get_avatar'])
            get_avatar = True if get_avatar == '1' else False
        except:
            get_avatar = False

        try:
            get_qrcode = str(request.form['get_qrcode'])
            get_qrcode = True if get_qrcode == '1' else False
        except:
            get_qrcode = False

        try:
            device_type = str(request.form['device_type']) # 0: mobile app; 1: card scanner
        except:
            device_type = '0'

        try:
            full_image_file = request.files['full_image'].read()
            full_image = (cv2.imdecode(np.frombuffer(full_image_file, np.uint8), cv2.IMREAD_COLOR))
        except:
            full_image = None

        image = (cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR))
        h, w, _ = image.shape
        # image = resize(image, w, h)
        result = api_predict(image, check, device_type=device_type, full_image=full_image, get_avatar=get_avatar, get_qrcode=get_qrcode)
        print(result)
        return result, 200
    except Exception as e:
        print(e)
        traceback.print_exc()


def api_predict(image, check, device_type='0', full_image=None, get_avatar=False, get_qrcode=False):
    if image is None:
        response = app.response_class(
            response=json.dumps({
                'result': 'Image is none'
            }, ensure_ascii=False),
            mimetype='application/json'
        )
        return response

    try:
        result = recognize(image, check,device_type=device_type, full_image=full_image, get_avatar=get_avatar, get_qrcode=get_qrcode)
        # write_log(image, result)
        if result is not None:
            response = app.response_class(
                response=result,
                mimetype='application/json'
            )
            return response
    except Exception as e:
        print(e)
        traceback.print_exc()

        response = app.response_class(
            response=json.dumps({
                'result': 'failed to recognize'
            }, ensure_ascii=False),
            mimetype='application/json'
        )

        return response


def api_predict_address(raw_addr):
    try:
        result = json.dumps(address_parser.matching(raw_addr))
        if result != None:
            response = app.response_class(
                response=result,
                mimetype='application/json'
            )
            print('result = ', response)
            return response
        return app.response_class(
            response=json.dumps({
                'result': 'failed to recognize'
            }, ensure_ascii=False),
            mimetype='application/json'
        )
    except Exception as e:

        traceback.print_exc()

        response = app.response_class(
            response=json.dumps({
                'result': 'failed to recognize'
            }, ensure_ascii=False),
            mimetype='application/json'
        )

        return response


@app.route('/api/cache/add', methods=['POST'])
@cross_origin(origin='*')
def cache_add():
    try:
        id = str(request.form['id'])
        name = str(request.form['name'])
        print(id)
        print(name)
        result = ENGINE_MODELS.cache.add_update(id, name)
        response = {'code': 0, 'result': result}
        print(response)
        return response, 200
    except Exception as e:
        print(e)
        traceback.print_exc()
        response = {'code': 1, 'result': None}
        print(response)
        return response, 400


@app.route('/api/cache/remove', methods=['POST'])
@cross_origin(origin='*')
def cache_remove():
    try:
        id = str(request.form['id'])
        result = ENGINE_MODELS.cache.remove(id)
        response = {'code': 0, 'result': result}
        print(response)
        return response, 200
    except Exception as e:
        print(e)
        traceback.print_exc()
        response = {'code': 1, 'result': None}
        return response, 400


@app.route('/api/cache/list', methods=['GET'])
@cross_origin(origin='*')
def cache_list():
    try:
        result = ENGINE_MODELS.cache.map_cache
        print(result)
        response = {'code': 0, 'total': len(result.keys()), 'result': result}
        return response, 200
    except Exception as e:
        print(e)
        traceback.print_exc()
        response = {'code': 1, 'result': None}
        return response, 400


@app.route('/api/address/recognition', methods=['POST'])
@cross_origin(origin='*')
def address_recognition():
    try:
        addr = str(request.form['address'])
        result = api_predict_address(addr)
        print(result)
        return result, 200
    except Exception as e:
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8686, debug=False, threaded=True)
