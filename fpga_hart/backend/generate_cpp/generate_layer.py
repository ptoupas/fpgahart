import argparse
import json
import os

import pandas as pd
from fpga_hart.backend.python_prototyping.generate_data import (
    conv_3d,
    elemwise_3d,
    gap_3d,
    gemm,
    relu_3d,
    shish_3d,
    sigmoid_3d,
)
from fpga_hart.layers.layer_parser import LayerParser
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


def get_layers_configurations(config_file):
    result = {}

    configuration = pd.read_csv(config_file)
    layers = configuration["Layer"].to_list()
    for l in layers:
        layers_config = configuration[configuration["Layer"] == l]["config"].to_dict()
        layers_config = layers_config[[*layers_config][0]]
        layers_config = layers_config.replace("'", '"')
        layers_config = json.loads(layers_config)
        partition_branch_depth = configuration[configuration["Layer"] == l][
            "Branch Depth"
        ].values[0]
        result[l] = {
            "config": layers_config,
            "type": configuration[configuration["Layer"] == l]["Type"].values[0],
        }

    return result


def generate_layer_code(
    layers_config, layers_type, layer_name, parser, prefix, hls_project_path
):
    # Generate layers files
    if "Swish" in layers_type:
        generate_swish_files(layer_name, layers_config, f"{prefix}/{layer_name}")
    elif "Relu" in layers_type:
        generate_relu_files(layer_name, layers_config, f"{prefix}/{layer_name}")
    elif "Sigmoid" in layers_type:
        generate_sigmoid_files(layer_name, layers_config, f"{prefix}/{layer_name}")
    elif "Add" in layers_type or "Mul" in layers_type:
        generate_elemwise_files(layer_name, layers_config, f"{prefix}/{layer_name}")
    elif "Conv" in layers_type:
        generate_conv_files(
            layer_name, layers_config, f"{prefix}/{layer_name}", hls_project_path
        )
    elif "GlobalAveragePool" in layers_type:
        generate_gap_files(layer_name, layers_config, f"{prefix}/{layer_name}")
    elif "Gemm" in layers_type:
        generate_gemm_files(layer_name, layers_config, f"{prefix}/{layer_name}")
    else:
        raise Exception(f"Layer {layers_type} not supported")

    # Generate top level partition file
    # generate_top_level_files(graph, branch_depth, layers_config, layer_name, prefix)

    # Generate testbench file
    generate_tb_files(layer_name, prefix, hls_project_path, is_layer=True)

    # Generate testbench data
    if "Swish" in layers_type:
        # to be implemented
        shish_3d()
    elif "Relu" in layers_type:
        # to be implemented
        relu_3d()
    elif "Sigmoid" in layers_type:
        # to be implemented
        sigmoid_3d()
    elif "Add" in layers_type or "Mul" in layers_type:
        # to be implemented
        elemwise_3d()
    elif "Conv" in layers_type:
        shape_in = layers_config["shape_in"]
        shape_out = layers_config["shape_out"]
        shape_kernel = layers_config["shape_kernel"]
        padding = layers_config["padding"]
        stride = layers_config["stride"]
        groups = layers_config["groups"]
        depthwise = layers_config["depthwise"]
        coarse_in_factor = layers_config["coarse_in_factor"]
        coarse_out_factor = layers_config["coarse_out_factor"]

        conv_3d(
            input_shape=shape_in,
            kernel_shape=shape_kernel,
            filters=shape_out[1],
            padding=padding,
            stride=stride,
            groups=groups,
            depthwise=depthwise,
            coarse_in=coarse_in_factor,
            coarse_out=coarse_out_factor,
            file_format="bin",
            prefix="generated_data/" + prefix,
            layer_name=layer_name,
        )
    elif "GlobalAveragePool" in layers_type:
        # to be implemented
        gap_3d()
    elif "Gemm" in layers_type:
        shape_in = layers_config["shape_in"]
        shape_out = layers_config["shape_out"]
        shape_bias = layers_config["shape_bias"]
        gemm(
            in_features=shape_in[1],
            out_features=shape_out[1],
            bias=True if shape_bias else False,
            prefix="generated_data",
            file_format="bin",
        )
    else:
        raise Exception(f"Layer {layers_type} not supported")


if __name__ == "__main__":
    args = parse_args()

    parser = LayerParser(
        model_name=args.model_name,
        se_block=False,
        singlethreaded=False,
        per_layer_plot=False,
        wandb_config=None,
    )

    layer_configuration = get_layers_configurations(args.config_file)

    for k, v in layer_configuration.items():
        generate_layer_code(
            v["config"], v["type"], k, parser, args.prefix, args.hls_project_path
        )
