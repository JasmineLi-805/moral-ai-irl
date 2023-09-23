#!/bin/bash

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type ltn --epoch 20 --layout coop_experiment_1

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type ltn --epoch 20 --layout coop_experiment_2

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type ltn --epoch 20 --layout coop_experiment_3

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type ltn --epoch 20 --layout coop_experiment_4

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type wte --epoch 30 --layout coop_experiment_1

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type wte --epoch 30 --layout coop_experiment_2

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type wte --epoch 30 --layout coop_experiment_3

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type wte --epoch 30 --layout coop_experiment_4
