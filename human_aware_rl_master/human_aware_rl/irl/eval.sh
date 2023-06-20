#!/bin/bash

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type ltn --epoch 20 --layout coop_experiment_5

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type wte --epoch 30 --layout coop_experiment_5

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type ltn --epoch 20 --layout coop_experiment_6

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type wte --epoch 30 --layout coop_experiment_6

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type ltn --epoch 20 --layout coop_experiment_7

PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=0,1 python train_modified_kitchen.py --trial 2 --type wte --epoch 30 --layout coop_experiment_7