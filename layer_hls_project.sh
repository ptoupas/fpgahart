#!/bin/bash

FPGAHART_PATH=$PWD
EXECUTION_TYPE="layer"
TARGET="throughput"
MODEL_NAME="custom_layers"
LAYER_NAME="custom_conv"
CONFIG_FILE=""
HLS_PARENT_DIR="/data/HLS_projects/fpga-hart-hls/$MODEL_NAME/layers"

python main.py $MODEL_NAME $EXECUTION_TYPE $TARGET

python fpga_hart/backend/generate_cpp/generate_layer.py $MODEL_NAME $HLS_PARENT_DIR --config_file $CONFIG_FILE

for layer in generated_files/$MODEL_NAME/*;
do
    CURRENT_LAYER=$(basename $layer)
    echo "Generating HLS project for layer $CURRENT_LAYER"
    mkdir -p $HLS_PARENT_DIR/$CURRENT_LAYER
    cp -r generated_files/$MODEL_NAME/$CURRENT_LAYER/* $HLS_PARENT_DIR/$CURRENT_LAYER

    cp fpga_hart/backend/run_hls.tcl $HLS_PARENT_DIR/$CURRENT_LAYER

    cd $HLS_PARENT_DIR/$CURRENT_LAYER/src
    ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/swish_3d.hpp swish_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/squeeze_3d.hpp squeeze_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/split_3d.hpp split_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/sliding_window_3d.hpp sliding_window_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/sigmoid_3d.hpp sigmoid_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/relu_3d.hpp relu_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/glue_3d.hpp glue_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/gap_3d.hpp gap_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/fork_3d.hpp fork_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/elemwise_3d.hpp elemwise_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/conv_3d.hpp conv_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/pool_3d.hpp pool_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/common.hpp common_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/common_tb.hpp common_tb_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/accum_3d.hpp accum_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/gemm.hpp gemm_.hpp
    cd - > /dev/null
done