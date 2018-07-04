#!/bin/bash
# 后台进程训练情感模型
set -x
c_dir=`dirname $0`
c_dir=`cd $c_dir;pwd`
train_file="../data/rdc-catalog-train.tsv"
model_folder="../data/"
tag="dcnn-9"
logs="${c_dir}/${tag}.log"

echo "停止$tag";
ps -ef|grep python |grep "${tag}" | awk '{print "kill -9 " $2}' |sh
echo "删除输出日志文件";
\rm -vf ${logs};
echo "删除已经训练好的模型文件 ${model_path}/${tag}"
\rm -rvf "${model_path}/${tag}"

nohup python3 -u ./pt_dcnn_train.py  --tag=${tag} --train_file=${train_file} --model_folder=${model_folder} \
 --train_add_rate=0  --drop_out=0.5  --maxlen=28  --epoch_num=10000  --lr=0.0001 --lr_halve_interval=30  --batch_size=200 --gpu=True  --label_pre=False > ${logs}  2>&1  &
ps -ef|grep "python3";
tail -fn 100 ${logs}
