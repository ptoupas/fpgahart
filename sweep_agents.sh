#!/bin/bash

PYTHON_ENV_NAME=$1
NUM_AGENTS=$2
MODEL_NAME=$3
OPTIMIZATION_TYPE=$4
OPTIMIZATION_TARGET=$5

source ~/anaconda3/etc/profile.d/conda.sh
conda activate $PYTHON_ENV_NAME

for (( c=0; c<=$NUM_AGENTS; c++ ))
do
   echo "Starting wandb agent on exp$c screen..."
   screen -S "exp$c" -d -m python main.py $MODEL_NAME $OPTIMIZATION_TYPE $OPTIMIZATION_TARGET
   sleep 5
done