#!/bin/bash
# train model and eval in validation set, and eval in coco.
echo "configs: $1"
#echo "result name: $2"

config_name=`echo $1 | cut -d / -f 2 | cut -d . -f 1`
echo "config name: $config_name"

rm -fr ./work_dirs/$config_name/tf_logs
echo "remove history train logs!"

python train.py $1 --gpus $2 --work_dir ./work_dirs/$config_name
python test.py $1  ./work_dirs/$config_name/latest.pth  --out work_dirs/$config_name/result_eval.pkl
python voc_eval.py  --result work_dirs/$config_name/result_eval.pkl --config $1
