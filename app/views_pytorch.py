#!/usr/bin/python3
# coding:utf-8
from flask import Flask, render_template,json,jsonify,request
import base64
import numpy as np
import pickle
from PIL import Image, ImageFont, ImageDraw
import io
import os

# PyTorch
from torchvision import models
import torch


global_times = 0
n_classes=3755
best_model_path = './train_model/resnet50-transfer-4-bestmodel.pth'
chinese_dictionary_file = './train_model/code_word.pkl'


test_image_file = './image/test.png'
pred1_image_file = './image/pred1.png'
pred2_image_file = './image/pred2.png'
pred3_image_file = './image/pred3.png'



def load_checkpoint(checkpoint_fpath):

    # Load in checkpoint
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))

    model = models.resnet50(pretrained=True)
    model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)
    # Resize
    img = image.resize((224, 224), Image.ANTIALIAS)
    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img)[:,:,:3].transpose((2, 0, 1)) / 256
    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor



def predict(image_path, model, topk=3):

    # Convert to pytorch tensor
    img_tensor = process_image(image_path)
    # Resize
    img_tensor = img_tensor.view(1, 3, 224, 224)
    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.nn.functional.softmax(out, dim = 1)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)
        top_p = topk.cpu().numpy()[0]

        return top_p, topclass


def createImage(predword,imagepath):
    im = Image.new("RGB", (100, 100), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    fonts = ImageFont.truetype("./static/fonts/msyh.ttc",60,encoding='utf-8')
    dr.text((20, 10), predword,font=fonts,fill="#000000")
    im.save(imagepath)

app = Flask(__name__, static_url_path='/static')
@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",title='Home')

@app.route('/chineseRecognize',methods=['POST'])
def chineseRecognize():

    data = json.loads(request.form.get('data'))
    imagedata = data["test_image"]
    imagedata = imagedata[22:]
    img = base64.b64decode(imagedata)
    file = open(test_image_file , 'wb')
    file.write(img)
    file.close()

    global global_times
    global model
    if (global_times == 0):
        model = load_checkpoint(checkpoint_fpath=best_model_path)
        global_times = 1

    predict_val, predict_index = predict(test_image_file , model = model, topk=3)
    with open(chinese_dictionary_file, 'rb') as f2:
        word_dict = pickle.load(f2)
    createImage(word_dict[int(predict_index[0][0])], pred1_image_file)
    createImage(word_dict[int(predict_index[0][1])], pred2_image_file)
    createImage(word_dict[int(predict_index[0][2])], pred3_image_file)


    with open(pred1_image_file, 'rb') as fin:
        image1_data = fin.read()
        pred1_image = base64.b64encode(image1_data)
    with open(pred2_image_file, 'rb') as fin:
        image2_data = fin.read()
        pred2_image = base64.b64encode(image2_data)
    with open(pred3_image_file, 'rb') as fin:
        image3_data = fin.read()
        pred3_image = base64.b64encode(image3_data)
    info = dict()
    info['pred1_image'] = "data:image/jpg;base64," + pred1_image.decode()
    info['pred1_accuracy'] = str('{:.2%}'.format(predict_val[0]))
    info['pred2_image'] = "data:image/jpg;base64," + pred2_image.decode()
    info['pred2_accuracy'] = str('{:.2%}'.format(predict_val[1]))
    info['pred3_image'] = "data:image/jpg;base64," + pred3_image.decode()
    info['pred3_accuracy'] = str('{:.2%}'.format(predict_val[2]))
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))
