# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 23:17:27 2021

@author: Administrator
"""
# 文件名：2useModel.py
import cv2
# ===========模型初始化、推理===========
net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
img = cv2.imread("a.jpg")
blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outInfo = net.getUnconnectedOutLayersNames() 
print(outInfo)
outs = net.forward(outInfo)
print("len(outs):",len(outs))
print("len(outs[0])：",len(outs[0]))
print("len(outs[1])：",len(outs[1]))
print("len(outs[2])：",len(outs[2]))
for i  in range(506):
    if(outs[0][i][4]>0.5):
        print("len(outs[0][i])：",i,outs[0][i][4])
print("outs[0][237]：\n",outs[0][237])
      

