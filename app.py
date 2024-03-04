import re

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
app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'


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
    os.remove("temp.png")

    # Concatenate all 'value' fields into one string
    full_text = ''.join(
        block['value'] for page in result.pages for block in page.blocks for line in block.lines for word in line.words)

    # Remove all non-alphanumeric characters
    cleaned_text = re.sub(r'\W+', '', full_text)

    # Search for an IBAN using a regular expression
    iban_match = re.search(r'[A-Z]{2}[0-9]{2}[A-Z0-9]{12,30}', cleaned_text)
    if iban_match:
        iban = iban_match.group()
        # Validate the IBAN (simple validation example)
        if validate_iban(iban):
            return {"message": f"Valid IBAN found: {iban}"}
        else:
            return {"message": "Invalid IBAN found."}
    else:
        return {"message": "No IBAN found."}


def validate_iban(iban):
    # IBAN validation (simplified version)
    # Convert letters to numbers (A=10, B=11, ..., Z=35) and move the first four characters to the end
    iban_rearranged = iban[4:] + iban[:4]
    iban_numeric = ''.join(str(int(char, 36)) for char in iban_rearranged)  # Convert each character to int base 36

    # Perform mod-97 operation and check if the result is 1
    return int(iban_numeric) % 97 == 1


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
