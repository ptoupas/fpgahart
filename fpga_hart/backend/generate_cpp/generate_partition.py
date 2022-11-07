import argparse
import configparser
import json
import os
from typing import Tuple

import pandas as pd
import yaml
from dotmap import DotMap
from generate_tb import generate_tb_files
from generate_top_level import generate_top_level_files
from layers.generate_conv import generate_conv_files
from layers.generate_elemwise import generate_elemwise_files
from layers.generate_gap import generate_gap_files
from layers.generate_gemm import generate_gemm_files
from layers.generate_relu import generate_relu_files
from layers.generate_sigmoid import generate_sigmoid_files
from layers.generate_split import generate_split_files
from layers.generate_squeeze import generate_squeeze_files
from layers.generate_swish import generate_swish_files

from fpga_hart.backend.python_prototyping.generate_data import partition_3d
from fpga_hart.partitions.partition_parser import PartitionParser
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
        partition_branch_depth = configuration[p]["branch_depth"]
        result[p] = {
            "layers": partition_layers_config,
            "branch_depth": partition_branch_depth,
        }

    return result


def generate_partition_code(
    layers_config, branch_depth, partition_name, model_name, parser, hls_project_path):
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
        elif "MaxPool" in layer_name or "AveragePool" in layer_name:
            # TODO: Implement pooling
            # generate_pool_files(layer_name, layer_config, model_name, partition_name=partition_name)
            print("MaxPool and AveragePool not implemented yet")
            pass
        elif "GlobalAveragePool" in layer_name:
            layer_name = "Gap_" + layer_name.split("_")[1]
            generate_gap_files(layer_name, layer_config, model_name, partition_name=partition_name)
        elif "Gemm" in layer_name:
            generate_gemm_files(layer_name, layer_config, model_name, partition_name=partition_name)
        else:
            raise Exception(f"Layer {layer_name} not supported")
    # Create the graph of the partition
    graph = parser.create_graph([*layers_config])

    # Generate extra supporting files (split, squeeze)
    split_points = utils.get_split_points(graph)
    for sf in split_points:
        generate_split_files(sf, layers_config[sf], model_name, partition_name=partition_name)

    squeeze_layers = identify_streams_mismatches(layers_config, graph.edges)
    for sl in squeeze_layers:
        generate_squeeze_files(
            sl[0],
            layers_config[sl[0]],
            sl[1],
            layers_config[sl[1]],
            model_name,
            partition_name=partition_name,
        )

    # Update the graph with the supporting layers
    graph = utils.update_graph(
        graph, split_points=split_points, squeeze_layers=squeeze_layers
    )
    os.path.join(os.getcwd(), "fpga_modeling_reports", model_name, "partition_graphs")
    if not os.path.exists(os.path.join(os.getcwd(), "fpga_modeling_reports", model_name, "partition_graphs")):
        os.makedirs(os.path.join(os.getcwd(), "fpga_modeling_reports", model_name, "partition_graphs"))
    visualize_graph(graph, os.path.join(os.getcwd(), "fpga_modeling_reports", model_name, "partition_graphs", partition_name + "_final"), False, partition_name)

    # Generate data files
    store_path = os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "data")
    partition_3d(partition_name, layers_config, graph, file_format="bin", store_path=store_path)
    exit()

    # Generate top level partition file
    generate_top_level_files(graph, branch_depth, layers_config, partition_name, prefix)

    # Generate testbench file
    generate_tb_files(partition_name, prefix, hls_project_path, is_layer=False)


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

    with open("fpga_hart/config/config_optimizer.yaml", "r") as yaml_file:
        config_dictionary = yaml.load(yaml_file, Loader=yaml.FullLoader)
        fpga_device, clock_freq, dsp, bram, mem_bw = get_fpga_specs()
        config_dictionary['device'] = fpga_device
        config_dictionary['clock_frequency'] = clock_freq
        config_dictionary['total_dsps'] = dsp
        config_dictionary['total_brams'] = bram
        config_dictionary['total_mem_bw'] = mem_bw

    config = DotMap(config_dictionary)

    parser = PartitionParser(args.model_name, False, False, False, False, config, False)

    if args.config_file:
        partition_configuration = get_partitions_configurations(args.config_file)
    else:
        partition_configuration = get_partitions_configurations(os.path.join(os.getcwd(), "fpga_modeling_reports", args.model_name, f"{args.model_name}_partitions.json"))

    for k, v in partition_configuration.items():
        if args.config_file:
            generate_partition_code(v['layers'], v['branch_depth'], k, "custom_partitions", parser, args.hls_project_path)
        else:
            generate_partition_code(v['layers'], v['branch_depth'], k, args.model_name, parser, args.hls_project_path)
