#!/bin/sh
#python -m torch.distributed.launch train_search.py -t M
CONFIG=$1
GPUNUM=$2
PORT=${PORT:-29500}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$GPUNUM --master_port=$PORT train_search.py --config $CONFIG --nprocs $GPUNUM --SYNC_BN
