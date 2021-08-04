import torch
from torch.utils.data import DataLoader

import cv2
import numpy as np
from torchvision import transforms
from YOLO_V1_DataSet_small import YoloV1DataSet

import matplotlib.pyplot as plt

from nms import generate_q_sigmoid, sigmoid_lut, post_process, NMS_max, torch_post_process, torch_NMS_max
from sigmoid import generate_q_sigmoid, sigmoid_lut, q17p14, q_mul, q_div

import importlib
mod = importlib.import_module("yolov1_bn_model_noaffine")

import sys
sys.path.append("../../../") # go to the directory of ai8x
import ai8x
# from batchnormfuser import bn_fuser
# import distiller.apputils as apputils

ai8x.set_device(85, simulate=True, round_avg=False, verbose=True)


# from YOLO_V1_DataSet_small import YoloV1DataSet
# dataSet = YoloV1DataSet(imgs_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
#                         annotations_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/Annotations",
#                         ClassesFile="../../VOC_remain_class.data",
#                         data_path='../../../../../YOLO_V1_GPU/VOC2007/Train/ImageSets/Main')

from YOLO_V1_DataSet_V2 import YoloV1DataSet
dataSet = YoloV1DataSet(imgs_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages",
                        annotations_dir="../../../../../YOLO_V1_GPU/VOC2007/Train/Annotations",
                        ClassesFile="../../VOC_remain_class.data")

Yolo = mod.Yolov1_net(num_classes=dataSet.Classes, bias=True)

qat_policy = {'start_epoch':150,
              'weight_bits':8,
              'bias_bits':8,
              'shift_quantile': 0.99}

ai8x.fuse_bn_layers(Yolo)
ai8x.initiate_qat(Yolo, qat_policy)

checkpoint = torch.load('../../Yolov1_checkpoint-q.pth.tar')
Yolo.load_state_dict(checkpoint['state_dict'])


# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000012.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000138.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000047.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000060.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000083.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000282.jpg"

test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000012.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000138.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000050.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000089.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000860.jpg"
# test_dir = "../../../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000133.jpg"


img_data = cv2.imread(test_dir)

transfrom = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # hui zi dong bian huan tong dao
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.act_mode_8bit = True
normalizer = ai8x.normalize(args)
train_data = transfrom(img_data) # .float()
train_data = normalizer(train_data)
train_data = torch.unsqueeze(train_data, 0)
print(train_data.max(), train_data.min())

bounding_boxes, fl_y = Yolo(train_data)

feature_map = fl_y.permute(0, 2, 3, 1).detach().reshape(-1).numpy().astype(np.int)

# print("Final Layer output:", fl_y.permute(0, 2, 3, 1)[0,:,:,[4,9,10,11,12,13,14]].detach().numpy())

print(feature_map.shape)

q_sigmoid, l, h, resolution = generate_q_sigmoid()
x = sigmoid_lut(feature_map, q_sigmoid, l, h, resolution)
# print(x)

torch_softmax = torch_post_process(x)
# print(torch_softmax.numpy().tolist()[0][0][0])
boxes1, pred_boxes1 = torch_NMS_max(torch_softmax, img_size=224, classes=5, confidence_threshold=0.1)
print('torch', boxes1)  # [0, 20, 24, 54, 0.82427978515625, 0.0579104907810688, 19]

softmax_x = post_process(x)
# print(np.array(softmax_x).reshape(7,7,15)[:,:,[4,9,10,11,12,13,14]])

boxes2, pred_boxes2 = NMS_max(softmax_x, img_size=224, classes=5, confidence_threshold=q17p14(0.), topk=1)
print('approx', boxes2)


img_data_copy = cv2.imread(test_dir)
# img_data_copy = torch.from_numpy(img_data_copy).permute(2, 1, 0).numpy()
# img_data = cv2.resize(img_data,(448,448),interpolation=cv2.INTER_AREA)
img_data_resize = cv2.resize(img_data_copy, (224,224), interpolation=cv2.INTER_AREA)

for box in boxes2:
    print("HERE", box[0],box[1],box[2],box[3],box[4],box[6])
    confidence = box[5]
    class_index = box[6]
    box = np.array(box[0:4]).astype(np.int)
    # img_data = cv2.rectangle(img_data, (box[0] * 2,box[1] * 2),(box[2] * 2,box[3] * 2),(0,255,0),1)
    # img_data = cv2.putText(img_data, "class:{} confidence:{}".format(dataSet.IntToClassName[box[5]],box[4]),(box[0] * 2,box[1] * 2),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
    img_data_resize = cv2.rectangle(img_data_resize, (box[0],box[1]), (box[2],box[3]), (0,255,0), 1)
    img_data_resize = cv2.putText(img_data_resize, "{}".format(dataSet.IntToClassName[class_index]),(box[0],box[1]+10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

print(img_data_resize.shape)
# img_data = np.transpose(img_data, (2, 1, 0))
plt.figure()
plt.imshow(img_data_resize)
plt.show()

