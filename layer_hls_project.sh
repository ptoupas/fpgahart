#!/bin/bash

EXECUTION_TYPE="layer"
TARGET="throughput"
MODEL_NAME="x3d_m"
LAYER_NAME="custom_conv"
CONFIG_FILE="fpga_modeling_reports/x3d_m_layers.csv"
HLS_PARENT_DIR="/home/ptoupas/Development/hls_single_layers"

python main.py $MODEL_NAME $EXECUTION_TYPE $TARGET --disable_wandb

python fpga_hart/backend/generate_cpp/generate_layer.py --model_name $MODEL_NAME --prefix $LAYER_NAME --config_file $CONFIG_FILE --hls_project_path $HLS_PARENT_DIR

for file in generated_data/$LAYER_NAME/*;
do
    CURRENT_LAYER=$(basename $file)
    echo "Copying data files to HLS project folder: $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/data/"
    mkdir -p $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/data
    cp -r generated_data/$LAYER_NAME/$CURRENT_LAYER/* $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/data/
    echo "Data files copied."
    echo "Copying source files to HLS project folder: $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/src/"
    mkdir -p $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/src
    cp -r generated_files/$LAYER_NAME/$CURRENT_LAYER/* $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/src/
    rm -rf $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/src/tb
    echo "Source files copied."
    echo "Copying testbench files to HLS project folder: $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/tb/"
    mkdir -p $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/tb
    cp -r generated_files/$LAYER_NAME/$CURRENT_LAYER/tb/* $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/tb/
    echo "Testbench files copied."

    cp fpga_hart/backend/run_hls.tcl $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/

    FPGAHART_PATH=$PWD
    cd $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/src
    ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/swish_3d.hpp swish_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/squeeze_3d.hpp squeeze_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/split_3d.hpp split_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/sliding_window_3d.hpp sliding_window_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/sigmoid_3d.hpp sigmoid_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/relu_3d.hpp relu_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/glue_3d.hpp glue_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/gap_3d.hpp gap_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/fork_3d.hpp fork_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/elemwise_3d.hpp elemwise_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/conv_3d.hpp conv_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/common.hpp common_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/common_tb.hpp common_tb_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/accum_3d.hpp accum_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/gemm.hpp gemm_.hpp
    cd - > /dev/null
done