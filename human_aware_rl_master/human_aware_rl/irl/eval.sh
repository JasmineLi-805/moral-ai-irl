#!/bin/bash

for fname in coop_experiment_1 coop_experiment_2 coop_experiment_3 coop_experiment_4
do
    for i in 1 2 3 4 5 6 7 8 9 10
    do
        PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type ltn --epoch 20 --layout $fname
        echo "complete run $i for ltn on $fname" >> progress.txt
        PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type wte --epoch 30 --layout $fname
        echo "complete run $i for wte on $fname" >> progress.txt
    done
done
