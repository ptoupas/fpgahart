import argparse
import configparser
import json
import os
from copy import deepcopy
from typing import Tuple

import pandas as pd
from generate_tb import generate_tb_files_partition
from generate_top_level import generate_top_level_files
from layers.generate_conv import generate_conv_files
from layers.generate_elemwise import generate_elemwise_files
from layers.generate_gap import generate_gap_files
from layers.generate_gemm import generate_gemm_files
from layers.generate_pool import generate_pool_files
from layers.generate_relu import generate_relu_files
from layers.generate_sigmoid import generate_sigmoid_files
from layers.generate_split import generate_split_files
from layers.generate_squeeze import generate_squeeze_files
from layers.generate_swish import generate_swish_files

from fpga_hart.backend.python_prototyping.generate_data import partition_3d
from fpga_hart.parser.onnx_parser import OnnxModelParser
from fpga_hart.utils import utils
from fpga_hart.utils.graph_manipulation import visualize_graph


def parse_args():
    parser = argparse.ArgumentParser(description="fpga_hart toolflow parser")
    parser.add_argument(
        "model_name",
        choices=["x3d_m", "slowonly", "r2plus1d", "c3d"],
        type=str,
        help="name of the HAR model",
    )
    parser.add_argument(
        "hls_project_path",
        type=str,
        help="path of the HLS project to be generated",
    )
    parser.add_argument(
        "--config_file", help="name of the model's configuration file"
    )

    args = parser.parse_args()
    return args


def get_partitions_configurations(config_file):
    result = {}

    configuration = pd.read_json(config_file)
    partitions = configuration.columns.to_list()
    for p in partitions:
        if "metrics" in p:
            continue
        partition_layers_config = configuration[p]["config"]
        partition_structure = configuration[p]["structure"]
        partition_branch_depth = configuration[p]["branch_depth"]
        result[p] = {
            "layers": partition_layers_config,
            "structure": partition_structure,
            "branch_depth": partition_branch_depth,
        }

    return result


def generate_partition_code(
    layers_config, partition_structure, branch_depth, partition_name, model_name, onnx_parser, hls_project_path):
    # Generate layers files
    for layer_name, layer_config in layers_config.items():
        if "Swish" in layer_name:
            generate_swish_files(layer_name, layer_config, model_name, partition_name=partition_name)
        elif "Relu" in layer_name:
            generate_relu_files(layer_name, layer_config, model_name, partition_name=partition_name)
        elif "Sigmoid" in layer_name:
            generate_sigmoid_files(layer_name, layer_config, model_name, partition_name=partition_name)
        elif "Add" in layer_name or "Mul" in layer_name:
            generate_elemwise_files(layer_name, layer_config, model_name, partition_name=partition_name)
        elif "Conv" in layer_name:
            generate_conv_files(layer_name, layer_config, model_name, hls_project_path, partition_name=partition_name)
        elif "MaxPool" in layer_name or "AveragePool" in layer_name and not "GlobalAveragePool" in layer_name:
            generate_pool_files(layer_name, layer_config, model_name, partition_name=partition_name)
        elif "GlobalAveragePool" in layer_name:
            layer_name = "Gap_" + layer_name.split("_")[1]
            generate_gap_files(layer_name, layer_config, model_name, partition_name=partition_name)
        elif "Gemm" in layer_name:
            generate_gemm_files(layer_name, layer_config, model_name, hls_project_path, partition_name=partition_name)
        else:
            raise Exception(f"Layer {layer_name} not supported")

    # Update layers config with split and squeeze layers
    layer_config = utils.generate_supportive_layer_config(partition_structure['layers'], layers_config)

    # Generate extra supporting files (split, squeeze)
    split_points = [n for n, sp in partition_structure['layers'].items() if sp['type'] == 'Split']
    for sp in split_points:
        generate_split_files(sp, layers_config[sp], model_name, partition_name=partition_name)

    squeeze_points = [n for n, sp in partition_structure['layers'].items() if sp['type'] == 'Squeeze']
    for sp in squeeze_points:
        generate_squeeze_files(
            sp,
            layers_config[sp],
            model_name,
            partition_name=partition_name,
        )

    # Generate top level partition file
    generate_top_level_files(partition_name, model_name, branch_depth, partition_structure, layers_config)

    # Generate testbench file
    generate_tb_files_partition(partition_name, model_name, hls_project_path, branch_depth, partition_structure, layers_config)

    # Generate data files
    store_path = os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "data")
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    if model_name == "custom_partitions":
        partition_3d(partition_name, partition_structure, layers_config, None, file_format="bin", store_path=store_path)
    else:
        partition_3d(partition_name, partition_structure, layers_config, onnx_parser, file_format="bin", store_path=store_path)


def identify_streams_mismatches(layers_config, connections):
    squeeze_layers = []
    for con in connections:
        in_node = con[0]
        if "coarse_factor" in layers_config[in_node].keys():
            in_node_streams = layers_config[in_node]["coarse_factor"]
        else:
            depthwise = layers_config[in_node]["depthwise"]
            in_node_streams = layers_config[in_node]["coarse_out_factor"]
        out_node = con[1]
        if "coarse_factor" in layers_config[out_node].keys():
            out_node_streams = layers_config[out_node]["coarse_factor"]
        else:
            out_node_streams = layers_config[out_node]["coarse_in_factor"]
        if in_node_streams != out_node_streams:
            squeeze_layers.append([in_node, out_node])
    return squeeze_layers

def get_fpga_specs() -> Tuple[str, int, int, int, float]:
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "fpga_hart", "config", "config_fpga.ini"))

    word_length = int(config.get("FPGA Specifications", "word_length"))
    clock_freq = int(config.get("FPGA Specifications", "clock_freq"))
    bram = int(config.get("FPGA Specifications", "bram"))
    bram_Kbytes = int(config.get("FPGA Specifications", "bram_type")) / 8
    dsp = int(config.get("FPGA Specifications", "dsp"))
    mem_bw = float(config.get("FPGA Specifications", "mem_bw"))
    fpga_device = config.get("FPGA Specifications", "fpga_device")

    return fpga_device, clock_freq, dsp, bram, mem_bw

if __name__ == "__main__":
    args = parse_args()

    onnx_parser = OnnxModelParser(args.model_name)

    if args.config_file:
        partition_configuration = get_partitions_configurations(args.config_file)
    else:
        partition_configuration = get_partitions_configurations(os.path.join(os.getcwd(), "fpga_modeling_reports", args.model_name, f"{args.model_name}_partitions.json"))

    for k, v in partition_configuration.items():
        print(f"Generating partition {k}")
        if args.config_file:
            generate_partition_code(v['layers'], v['structure'], v['branch_depth'], k, "custom_partitions", deepcopy(onnx_parser), args.hls_project_path)
        else:
            generate_partition_code(v['layers'], v['structure'], v['branch_depth'], k, args.model_name, deepcopy(onnx_parser), args.hls_project_path)