#!/bin/bash

lr=0.01
steps=10
max_norm=0.01
sigma_0=0.08
init_s=0.08
alpha=1.0
data=cifar10
root=./
model=vgg
model_out=./checkpoints/cifar10_vgg_BNN.pth

python ./train_BNN.py \
        --lr ${lr} \
        --step ${steps} \
        --max_norm ${max_norm} \
        --sigma_0 ${sigma_0} \
        --init_s ${init_s} \
        --data cifar10 \
        --model vgg \
        --root ./ \
        --model_out ${model_out} \
        #--resume 