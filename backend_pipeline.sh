#!/bin/bash

EXECUTION_TYPE="partition"
TARGET="throughput"
MODEL_NAME="x3d_m"
CONFIG_FILE="fpga_modeling_reports/x3d_m_custom_partitions.csv"
PARTITION_NAME="custom_partitions"
HLS_PARENT_DIR="/data/HLS_projects/fpga-hart-hls"

python main.py $MODEL_NAME $EXECUTION_TYPE $TARGET

python fpga_hart/backend/python_prototyping/generate_data.py --op_type 3d_part --prefix $PARTITION_NAME --config_file $CONFIG_FILE

python fpga_hart/backend/generate_cpp/generate_partition.py --model_name $MODEL_NAME --prefix $PARTITION_NAME --config_file $CONFIG_FILE --hls_project_path $HLS_PARENT_DIR

for file in generated_data/$PARTITION_NAME/*;
do
    CURRENT_PARTITION=$(basename $file)
    echo "Copying data files to HLS project folder: $HLS_PARENT_DIR/$PARTITION_NAME/$CURRENT_PARTITION/data/"
    mkdir -p $HLS_PARENT_DIR/$PARTITION_NAME/$CURRENT_PARTITION/data
    cp -r generated_data/$PARTITION_NAME/$CURRENT_PARTITION/* $HLS_PARENT_DIR/$PARTITION_NAME/$CURRENT_PARTITION/data/
    echo "Data files copied."
    echo "Copying source files to HLS project folder: $HLS_PARENT_DIR/$PARTITION_NAME/$CURRENT_PARTITION/src/"
    mkdir -p $HLS_PARENT_DIR/$PARTITION_NAME/$CURRENT_PARTITION/src
    cp -r generated_files/$PARTITION_NAME/$CURRENT_PARTITION/* $HLS_PARENT_DIR/$PARTITION_NAME/$CURRENT_PARTITION/src/
    rm -rf $HLS_PARENT_DIR/$LAYER_NAME/$CURRENT_LAYER/src/tb
    echo "Source files copied."
    echo "Copying testbench files to HLS project folder: $HLS_PARENT_DIR/$PARTITION_NAME/$CURRENT_PARTITION/tb/"
    mkdir -p $HLS_PARENT_DIR/$PARTITION_NAME/$CURRENT_PARTITION/tb
    cp -r generated_files/$PARTITION_NAME/$CURRENT_PARTITION/tb/* $HLS_PARENT_DIR/$PARTITION_NAME/$CURRENT_PARTITION/tb/
    echo "Testbench files copied."

    cp fpga_hart/backend/run_hls.tcl $HLS_PARENT_DIR/$PARTITION_NAME/$CURRENT_PARTITION/

    FPGAHART_PATH=$PWD
    cd $HLS_PARENT_DIR/$PARTITION_NAME/$CURRENT_PARTITION/src
    ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/swish_3d.hpp swish_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/squeeze_3d.hpp squeeze_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/split_3d.hpp split_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/sliding_window_3d.hpp sliding_window_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/sigmoid_3d.hpp sigmoid_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/relu_3d.hpp relu_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/glue_3d.hpp glue_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/gap_3d.hpp gap_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/fork_3d.hpp fork_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/elemwise_3d.hpp elemwise_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/conv_3d.hpp conv_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/pool_3d.hpp pool_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/common.hpp common_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/common_tb.hpp common_tb_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/accum_3d.hpp accum_3d_.hpp && ln -nsf $FPGAHART_PATH/fpga_hart/backend/include_files/gemm.hpp gemm_.hpp
    cd - > /dev/null
done