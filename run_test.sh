#!/bin/bash
# train model and eval in validation set, and eval in coco.
echo "configs: $1"
#echo "result name: $2"

config_name=`echo $1 | cut -d / -f 2 | cut -d . -f 1`
echo "config name: $config_name"

#python train.py $1 --gpus 1 --work_dir ./work_dirs/$config_name
python test_no_ground_truth.py $1  ./work_dirs/$config_name/latest.pth  --out work_dirs/$config_name/result_test.pkl
#python voc_eval.py  --result work_dirs/$config_name/result.pkl --config $1
