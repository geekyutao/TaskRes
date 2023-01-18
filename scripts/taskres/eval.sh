#!/bin/bash

# custom config
DATA=/data0/dataset
TRAINER=TaskRes
SHOTS=16
NCTX=16
CSC=False
CTP=end
SCALE=0.5

DATASET=$1
CFG=$2
OUTDIR=$3
MODELDIR=$4

for SEED in 1 2 3
do
    CUDA_VISIBLE_DEVICES=3 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${OUTDIR}/seed${SEED} \
    --model-dir ${MODELDIR}/seed${SEED} \
    --load-epoch 200 \
    --eval-only \
    TRAINER.TaskRes.RESIDUAL_SCALE ${SCALE}
done