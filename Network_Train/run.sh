#!/bin/sh
CONFIG=$1
GPUNUM=$2
PORT=${PORT:-29500}
python -m torch.distributed.launch --nproc_per_node=$GPUNUM --master_port=$PORT train.py --config $CONFIG --distp --nprocs $GPUNUM
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config $CONFIG --eval_only --show_class_acc --nprocs $GPUNUM

