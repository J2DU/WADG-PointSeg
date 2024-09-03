#!/bin/bash

module load libfabric/1.10.1 
module load openmpi/4.0.3
module load ucx/1.8.0
module load cuda/11.4
module load python/3.8.2
source ~/ENV/bin/activate
python train.py configs/kitti.yaml 
# python train.py configs/synlidar.yaml 

