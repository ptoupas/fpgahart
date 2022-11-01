import argparse
import configparser
import json
import os
from typing import Tuple

import pandas as pd
import yaml
from dotmap import DotMap
from fpga_hart.partitions.partition_parser import PartitionParser
from fpga_hart.utils import utils

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


def parse_args():
    parser = argparse.ArgumentParser(description="fpga_hart toolflow parser")
    parser.add_argument("--model_name", help="name of the HAR model", required=True)
    parser.add_argument(
        "--hls_project_path",
        help="path of the HLS project to be generated",
        required=True,
    )
    parser.add_argument(
        "--prefix",
        help="the parent folder in which to store the model's partitions",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config_file", help="name of the model's configuration file", required=True
    )

    args = parser.parse_args()
    return args


def get_partitions_configurations(config_file):
    result = {}

    configuration = pd.read_csv(config_file)
    partitions = configuration["Part"].to_list()
    for p in partitions:
        partition_layers_config = configuration[configuration["Part"] == p][
            "config"
        ].to_dict()
        partition_layers_config = partition_layers_config[[*partition_layers_config][0]]
        partition_layers_config = partition_layers_config.replace("'", '"')
        partition_layers_config = json.loads(partition_layers_config)
        partition_branch_depth = configuration[configuration["Part"] == p][
            "Branch Depth"
        ].values[0]
        result[p] = {
            "layers": partition_layers_config,
            "branch_depth": partition_branch_depth,
        }

    return result


def generate_partition_code(
    layers_config, branch_depth, partition_name, parser, prefix, hls_project_path
):
    # Generate layers files
    for l in [*layers_config]:
        if "Swish" in l:
            generate_swish_files(l, layers_config[l], f"{prefix}/{partition_name}")
        elif "Relu" in l:
            generate_relu_files(l, layers_config[l], f"{prefix}/{partition_name}")
        elif "Sigmoid" in l:
            generate_sigmoid_files(l, layers_config[l], f"{prefix}/{partition_name}")
        elif "Add" in l or "Mul" in l:
            generate_elemwise_files(l, layers_config[l], f"{prefix}/{partition_name}")
        elif "Conv" in l:
            generate_conv_files(
                l, layers_config[l], f"{prefix}/{partition_name}", hls_project_path
            )
        elif "GlobalAveragePool" in l:
            shorted_name = "Gap_" + l.split("_")[1]
            generate_gap_files(
                shorted_name, layers_config[l], f"{prefix}/{partition_name}"
            )
        elif "Gemm" in l:
            generate_gemm_files(l, layers_config[l], f"{prefix}/{partition_name}")
        else:
            raise Exception(f"Layer {l} not supported")

    # Create the graph of the partition
    graph = parser.create_graph([*layers_config])

    # Generate extra supporting files (split, squeeze)
    split_points = utils.get_split_points(graph)
    for sf in split_points:
        generate_split_files(sf, layers_config[sf], f"{prefix}/{partition_name}")

    squeeze_layers = identify_streams_mismatches(layers_config, graph.edges)
    for sl in squeeze_layers:
        generate_squeeze_files(
            sl[0],
            layers_config[sl[0]],
            sl[1],
            layers_config[sl[1]],
            f"{prefix}/{partition_name}",
        )

    # Update the graph with the supporting layers
    graph = utils.update_graph(
        graph, split_points=split_points, squeeze_layers=squeeze_layers
    )
    if not os.path.exists(f"generated_files/{prefix}/graphs/"):
        os.makedirs(f"generated_files/{prefix}/graphs/")
    parser.visualize_graph(graph, f"generated_files/{prefix}/graphs/{partition_name}")

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

    parser = PartitionParser(args.model_name, False, False, False, config, False)

    partition_configuration = get_partitions_configurations(args.config_file)

    for p in [*partition_configuration]:
        generate_partition_code(
            partition_configuration[p]["layers"],
            partition_configuration[p]["branch_depth"],
            p,
            parser,
            args.prefix,
            args.hls_project_path,
        )
