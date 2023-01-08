#!/bin/sh
eps=0.01
model_path=./checkpoints/cifar10_BNN.pth 
alpha=1.0
python3 LinfPGDAttack.py\
        --epsilon ${eps}
        --alpha ${alpha}
        --path_model ${model_path}
