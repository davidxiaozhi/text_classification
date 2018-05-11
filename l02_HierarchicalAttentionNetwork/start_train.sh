#!/bin/bash
# 后台进程训练情感模型
set -x
c_dir=`dirname $0`
c_dir=`cd $c_dir;pwd`
tag="attention-01"
logs="${c_dir}/${tag}.log"
corpus="../data/corpus_0.txt"
model_path="../data/"
echo "删除输出日志文件";

#data_lines=`cat ${data_file}|wc -l`
echo "训练数据总共${data_lines}"
echo "开始进行分类模型训练";
\rm -vf ${logs};
echo "删除已经训练好的模型文件 ${model_path}/${tag}"
\rm -rvf "${model_path}/${tag}"
#nohup \
nohup python3 -u  ./pt_attention_train.py --tag=${tag} --corpus=${corpus} --model_path=${model_path} > ${logs}  2>&1 & 
#>> ${logs}  2>&1  &
ps -ef|grep "python3";
tail -fn 100 ${logs}
