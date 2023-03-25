#!/bin/bash

PYTHON_ENV_NAME=$1
NUM_AGENTS=$2
MODEL_NAME=$3
OPTIMIZATION_TYPE=$4
OPTIMIZATION_TARGET=$5

source ~/anaconda3/etc/profile.d/conda.sh
conda activate $PYTHON_ENV_NAME

PLATFORMS=("zc706" "vus440" "vc709" "zcu104-106" "zcu102")

exp_count=0
for platform_name in ${PLATFORMS[@]}; do
    for (( c=0; c<$NUM_AGENTS; c++ )) do
        exp_num=$((exp_count+$c))
        echo "Starting wandb agent on exp$exp_num screen..."
        screen -S "${MODEL_NAME}-exp$exp_num" -d -m python main.py $MODEL_NAME $platform_name $OPTIMIZATION_TYPE $OPTIMIZATION_TARGET --enable_wandb
        sleep 5
    done
    exp_count=$((exp_count+$NUM_AGENTS))
done