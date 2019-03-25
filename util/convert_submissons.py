# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:06:00 2019

@author: guoxiaofeng
"""

import pickle
import numpy as np
import cv2
import os
import csv

bbox_result = []
img_lists = []
img_file = "./data/voc07_test.pkl"
result_path = "test_result"
img_prefix = "data/coco/test2017"

img2box = []

with open(img_file, 'rb') as f:
    imgs = pickle.load(f, encoding="utf-8")
    
    for img_name in imgs:
        img_name = os.path.join(img_prefix, img_name["filename"])
        img_lists.append(img_name)
        
with open("result_test.pkl", 'rb') as f:
    bbox_result = pickle.load(f, encoding="utf-8")
    for i, bbox in enumerate(bbox_result):
        bbox = bbox[0]
        # img = cv2.imread(img_lists[i])
        predict_box = bbox[:, :4]
        predict_box = np.round(predict_box)
        basename = os.path.basename(img_lists[i])
        for box in predict_box:
            # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            box = [str(int(i)) for i in box]
            box = " ".join(box)
            img2box.append([basename, box])
            
        # cv2.imwrite(os.path.join(result_path, basename), img)

with open("submissons.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for row in img2box:
        writer.writerow(row)