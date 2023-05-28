#!/bin/bash
for i in 25 50 75
do
   PYTHONPATH=../../:../../../ CUDA_VISIBLE_DEVICES=3 python evaluate_maxent.py --trial 23 --epoch $i --type noncoop
done