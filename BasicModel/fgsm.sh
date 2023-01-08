#!/bin/sh
eps=0.01
model_path=./checkpoints/cifar10_BNN.pth 
alpha=1.0
python3 FGSMAttack.py\
        --epsilon ${eps}
        --path_model ${model_path}
