# Chinese Handwritten Recognition

`Flask` web app recognizes 3,755 Chinese handwritten characters.

[![](https://img.shields.io/badge/Heroku-Open_Web_App-blue?logo=Heroku)](https://chinese-handwritten.herokuapp.com/)

## Description
Train convolutional neural network using pretrained ResNet50 on Imagenet 1000 dataset using `Pytorch`:

- **Fixed feature extractor**: The weights for all of the network will be freezed except that of the final fully connected layer. This step will be trained for 2 -3 epochs to avoid overfitting

- **Finetuning**: The entire weights of the network will be trained with discriminative learning rates. The layers closer to the input layer will be set low learning rate because they may learn more general features, such as lines and edges . On the other hand, we increase the learning rate for later layers as they learn the detail features.

|         | Top 1           | Top 5  |
| ------------- |:-------------:| -----:|
| **Accuracy**     | 95.41% | 99.27%|


## Screenshots
![demo](./Image/demo.gif)

## Prerequisites
- Python <= 3.7.9
- Anaconda (optional, is used to install environment, you can use python `venv` instead)
- HSK 3 at least (Just kidding >.<)

## Installation
1. Clone repository:
```bash
$ git clone https://github.com/yoogun143/Chinese-Handwritten-Recognition.git
$ cd app
```

2. Install dependencies using Anaconda and pip
```bash
$ conda create -n chinese-handwritten-app python=3.7.9  #Create new environment
$ conda activate chinese-handwritten-app #Activate environment
$ conda install pip #install pip inside the environment
$ pip install -r requirements.txt #Install required dependencies
```

3. Download `resnet50-transfer-4-bestmodel.pth` file weights from [here](https://drive.google.com/file/d/1Hh7R6QcnZ5mw9Xgj7cnnTjkAPSPlREht/view?usp=sharing) and place in *train_model* folder
```
train_model
├─code_word.pkl
└─resnet50-transfer-4-bestmodel.pth
```

4. Run the app
```bash
$ python views_pytorch.py
 * Running on http://127.0.0.1:5000/
```

Voila! the app is now run locally. Now head over to http://127.0.0.1:5000/, and you should see your app roaring.

## Training instruction
You need to download train and test HWDB1.1 dataset below

http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip

http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip

Put 2 zip files into data folder like below and unzip. Data folders tree:

```
data
├─test
│   └─HWDB1.1tst_gnt.zip
└─train
    └─HWDB1.1trn_gnt.zip
```

Convert gnt to png

```bash
$ python gnt2png.py
```
Start training

```bash
$ python train.py
```

## Roadmap
- [ ] Reduce web app latency, loading function
- [ ] Handwritten keyboard
- [ ] Train with more words


## Credits
[cnn_handwritten_chinese_recognition](https://github.com/taosir/cnn_handwritten_chinese_recognition)

[tf28: 手写汉字识别](https://cloud.tencent.com/developer/article/1016464)

[drawingboard.js](https://github.com/Leimi/drawingboard.js)
## License
MIT License

Copyright (c) [2021] [Thanh Hoang]
