#!/bin/bash 

model_partitions="custom_partitions"
hls_parent_dir="/data/HLS_projects/fpgaHART_hls"

python main.py x3d_m

python fpgaHART/backend/python_prototyping/generate_data_3d.py --op_type 3d_part --prefix $model_partitions --config_file fpga_modeling_reports/x3d_m_custom_partitions.csv

python fpgaHART/backend/generate_cpp/generate_partition.py --model_name x3d_m --prefix $model_partitions --config_file fpga_modeling_reports/x3d_m_custom_partitions.csv

for file in generated_data/$model_partitions/*;
do
    current_partition=$(basename $file)
    mkdir -p $hls_parent_dir/$model_partitions/$current_partition/data
    cp -r generated_data/$model_partitions/$current_partition/* $hls_parent_dir/$model_partitions/$current_partition/data/
    mkdir -p $hls_parent_dir/$model_partitions/$current_partition/src
    cp -r generated_files/$model_partitions/$current_partition/* $hls_parent_dir/$model_partitions/$current_partition/src/
    mkdir -p $hls_parent_dir/$model_partitions/$current_partition/tb
    cp -r generated_files/$model_partitions/$current_partition/tb/* $hls_parent_dir/$model_partitions/$current_partition/tb/

    cp fpgaHART/backend/run_hls.tcl $hls_parent_dir/$model_partitions/$current_partition/
    
    cd $hls_parent_dir/$model_partitions/$current_partition/src
    ln -nsf $hls_parent_dir/include/swish_3d.hpp swish_3d_.hpp && ln -nsf $hls_parent_dir/include/squeeze_3d.hpp squeeze_3d_.hpp && ln -nsf $hls_parent_dir/include/split_3d.hpp split_3d_.hpp && ln -nsf $hls_parent_dir/include/sliding_window_3d.hpp sliding_window_3d_.hpp && ln -nsf $hls_parent_dir/include/sigmoid_3d.hpp sigmoid_3d_.hpp && ln -nsf $hls_parent_dir/include/relu_3d.hpp relu_3d_.hpp && ln -nsf $hls_parent_dir/include/glue_3d.hpp glue_3d_.hpp && ln -nsf $hls_parent_dir/include/gap_3d.hpp gap_3d_.hpp && ln -nsf $hls_parent_dir/include/fork_3d.hpp fork_3d_.hpp && ln -nsf $hls_parent_dir/include/elemwise_3d.hpp elemwise_3d_.hpp && ln -nsf $hls_parent_dir/include/conv_3d.hpp conv_3d_.hpp && ln -nsf $hls_parent_dir/include/common.hpp common_.hpp && ln -nsf $hls_parent_dir/include/accum_3d.hpp accum_3d_.hpp
    cd -
done