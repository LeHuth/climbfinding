from flask import Flask, request, make_response
from doctr.io import DocumentFile
from werkzeug.utils import secure_filename
import os
from doctr.models import ocr_predictor
from flask_cors import CORS, cross_origin


import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/ocr')
def ocr():
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    file_path = 'card.png'
    image = DocumentFile.from_images(file_path)
    result = model(image)
    json_output = result.export()
    return {"message": json_output}


@app.route('/post_ocr', methods=['POST'])
def post_ocr():
    if 'file' not in request.files:
        return {"error": "No file part in the request."}, 400
    file = request.files['file']
    if file.filename == '':
        return {"error": "No file selected for uploading."}, 400
    if file:
        filename = secure_filename(file.filename)
        file.save(filename)
        model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        image = DocumentFile.from_images(filename)
        result = model(image)
        json_output = result.export()
        os.remove(filename)
        return {"message": json_output}


@app.route('/baseocr', methods=['POST'])
@cross_origin()
def base_ocr():
    data = request.get_json()
    if 'image' not in data:
        return {"error": "No image data in the request."}, 400
    image_data = data['image']
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(BytesIO(image_data))
    image.save("temp.png")
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    result = model(DocumentFile.from_images("temp.png"))
    json_output = result.export()
    os.remove("temp.png")
    return {"message": json_output}

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    app.run()
