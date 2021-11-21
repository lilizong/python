# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:55:18 2021

@author: Administrator
"""

import cv2
import numpy as np
# ===========初始化、推理===========
net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
classes =  open('coco.names', 'rt').read().strip().split("\n")
img = cv2.imread("a.jpg")
blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outInfo = net.getUnconnectedOutLayersNames() 
outs = net.forward(outInfo)
# ===========获取置信度较高的边框===========
classIDs = [] # 所有分类ID
boxes = [] # 所有边框
confidences = [] # 所有置信度
(H, W) = img.shape[:2]   #图像的宽、高，辅助确定图像内边框位置
for out in outs:  # 各个输出层
    for alternative in out:  # 各个框框
        #detection内存储的是边框位置(前四个)的百分比、置信度(从第5个开始所有)
        #例如，0.5表示高度（或宽度）的50%
        scores = alternative[5:]  # 所有分类的置信度
        classID = np.argmax(scores)  # 根据最高置信度的id确定分类id
        confidence = scores[classID]  # 置信度
        # 确定所有可能的边框
        # 仅考虑置信度大于0.5的，太小的忽略
        if confidence > 0.3:
            box = alternative[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
            (centerX, centerY, width, height) = box.astype("int")
            #centerX，centerY是矩形框的中心点，要通过他们计算出左上角坐标x,y
            x = int(centerX - (width / 2))   #x方向中心点-框宽度/2
            y = int(centerY - (height / 2))  #y方向中心点-框高度/2
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
# 非极大值抑制，将重合的边框去重处理
indices = cv2.dnn.NMSBoxes(boxes, confidences,0.6,0.5)
print(boxes)
print(confidences)
print(indices)
for i in indices:
    print(confidences[int(i)])