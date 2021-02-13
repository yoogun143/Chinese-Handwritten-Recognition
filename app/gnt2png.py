import os
import numpy as np
import struct
from PIL import Image
# data文件夹存放转换后的.png文件
data_dir = './data'
# 路径为存放数据集解压后的.gnt文件
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')


def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tagcode = header[5] + (header[4] << 8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                break
            #global image
            image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width)) #将file以二进制一位数组读取到image里面
            yield image, tagcode  #tagcode是汉字的16进制码

    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode


char_set = set()
for _, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312','ignore')  #tagcode_unicode是汉字(decode后是汉字，encode后是16进制)
    char_set.add(tagcode_unicode) #所有汉字的集合
for _, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312','ignore')
    char_set.add(tagcode_unicode)
char_list = list(char_set)
char_dict = dict(zip(sorted(char_list), range(len(char_list))))
#print(len(char_dict))
#print("char_dict=", char_dict)

import pickle

with open('char_dict', 'wb') as f:
 pickle.dump(char_dict, f)

train_counter = 0
test_counter = 0

for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312','ignore')
    im = Image.fromarray(image)
# 路径为data文件夹下的子文件夹，train为存放训练集.png的文件夹
    dir_name = './data/train/' + '%0.5d' % char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    im.convert('RGB').save(dir_name + '/' + str(train_counter) + '.png')
    #print("train_counter=", train_counter)
    train_counter += 1
for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312','ignore')
    im = Image.fromarray(image)
# 路径为data文件夹下的子文件夹，test为存放测试集.png的文件夹
    dir_name = './data/test/' + '%0.5d' % char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    im.convert('RGB').save(dir_name + '/' + str(test_counter) + '.png')
    #print("test_counter=", test_counter)
    test_counter += 1
# 样本数
#print(train_counter, test_counter)
