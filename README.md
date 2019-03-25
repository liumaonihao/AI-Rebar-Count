# AI Rebar Count
[钢筋数量识别比赛](https://www.datafountain.cn/competitions/332/details/rule)


## Requirements
- mmdetection
- Python3
- Pytorch 0.4

## How to run it?
### convert voc data to custom data
`cd tools/convert_datasets`

 `python pascal_voc.py ../../data/VOCdevkit -o ../../data/coco`

### train model
`python train.py  configs/pascal_voc/cascade_rcnn_dconv_c3-c5_r101_fpn_1x.py`

### eval model use voc
`python voc_eval.py work_dirs/cascade_rcnn_dconv_c3-c5_r101_fpn_1x/result.pkl ./configs/pascal_voc/cascade_rcnn_dconv_c3-c5_r101_fpn_1x.py`


### test model（no ground truth）
`python test_no_ground_truth.py ./configs/pascal_voc/cascade_rcnn_dconv_c3-c5_r101_fpn_1x.py ./work_dirs/cascade_rcnn_dconv_c3-c5_r101_fpn_1x/latest.pth --out work_dirs/result.pkl`

### convert submissons file
`python ./util/convert_submissons.py` 
