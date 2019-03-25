
import os
import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import pandas as pd
import numpy as np

train_file=open('../data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt','w')
test_file=open('../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','w')
train_files = []
test_files = []


for _, _, train_files in os.walk('../data/data_foundation/train_dataset'):
    continue
for _,_,test_files in os.walk('../data/data_foundation/test_dataset'):
    continue
for file in train_files:
    train_file.write(file.split('.')[0]+'\n')
 
for file in test_files:
    test_file.write(file.split('.')[0]+'\n')

train_file.close()
test_file.close()

def save_xml(image_name, bbox, save_dir, width=2666, height=2000, channel=3):
 
    node_root = Element('annotation')
 
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
 
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name
 
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width
 
    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height
 
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel
 
    for x, y, x1, y1 in bbox:
        left, top, right, bottom = x, y, x1, y1
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'person'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom
 
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
 
    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)
 
    return
 
 
def change2xml(label_dict={}):
    for image in label_dict.keys():
        image_name = os.path.split(image)[-1]
        bbox = label_dict.get(image, [])
        save_xml(image_name, bbox)
    return
 

data = pd.read_table("../data/data_foundation/train_labels.csv", sep=",")
name_file = open("../data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt", 'r')

file_lists = name_file.readlines()

for name in file_lists:
    image_path = os.path.join('../data/VOCdevkit/VOC2007/JPEGImages/', name[:-1]+'.jpg')
    print(image_path)
    img=cv2.imread(image_path)
    height,width  = img.shape[:2]
    name=name[:-1]+'.jpg'
    xx = np.array(data[data['ID'] == name][' Detection'])
    bbox=[]
    for i in range(xx.shape[0]):
        bbox.append(xx[i].split(' '))
    save_xml(image_name=name, bbox=bbox, save_dir='../data/VOCdevkit/VOC2007/Annotations', width=width, height=height, channel=3)
