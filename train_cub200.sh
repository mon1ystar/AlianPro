#!/bin/bash

# 设置环境变量

export CUDA_VISIBLE_DEVICES="0"


# 运行 Python 脚本
python "train.py" \
  -project "negative_prompt" \
  -dataset "cub200" \
  -base_mode "ft_dot" \
  -new_mode "avg_cos" \
  -gamma "0.05" \
  -lr_base "0.1" \
  -lr_new "0.1" \
  -decay "0.0005" \
  -epochs_base "10" \
  -epochs_new "5" \
  -schedule "Cosine" \
  -milestones "20" "30" "45" \
  -gpu "0" \
  -temperature "16" \
  -start_session "0" \
  -batch_size_base "64" \
  -test_batch_size "64" \
  -seed "1" \
  -vit \
  -out "PriViLege" \
  -no5shot \
  -dataroot "/amax/2020/qyl/CVPR22-Fact-main/data" \
  
  
  
  
