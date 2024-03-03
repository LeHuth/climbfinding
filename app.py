from flask import Flask
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

app = Flask(__name__)


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


if __name__ == '__main__':
    app.run()
