import argparse
import json
import os

import pandas as pd
from fpga_hart.backend.python_prototyping.generate_data import (conv_3d,
                                                                elemwise_3d,
                                                                gap_3d, gemm,
                                                                pool_3d,
                                                                relu_3d,
                                                                shish_3d,
                                                                sigmoid_3d)
from fpga_hart.layers.layer_parser import LayerParser

from generate_tb import generate_tb_files
from layers.generate_conv import generate_conv_files
from layers.generate_elemwise import generate_elemwise_files
from layers.generate_gap import generate_gap_files
from layers.generate_gemm import generate_gemm_files
from layers.generate_relu import generate_relu_files
from layers.generate_sigmoid import generate_sigmoid_files
from layers.generate_swish import generate_swish_files


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
        "--config_file", type=str, help="path of the custom configuration file"
    )

    args = parser.parse_args()
    return args


def get_layers_configurations(config_file):
    result = {}

    configuration = pd.read_json(config_file)
    layers = configuration.columns.to_list()
    for l in layers:
        layers_config = configuration[l]["config"]
        if "Swish" in l:
            layers_type = "Swish"
        elif "Relu" in l:
            layers_type = "Relu"
        elif "Sigmoid" in l:
            layers_type = "Sigmoid"
        elif "Add" in l:
            layers_type = "Add"
        elif "Mul" in l:
            layers_type = "Mul"
        elif "Conv" in l:
            layers_type = "Conv"
        elif "Pooling" in l:
            layers_type = "Pooling"
        elif "GlobalAveragePool" in l:
            layers_type = "GlobalAveragePool"
        elif "Gemm" in l:
            layers_type = "Gemm"
        elif "Activation" in l:
            layers_type = "Activation"
        elif "ElementWise" in l:
            layers_type = "ElementWise"
        else:
            raise Exception(f"Layer {l} not supported")
        result[l] = {
            "config": layers_config,
            "type": layers_type,
        }

    return result


def generate_layer_code(
    layers_config, layers_type, layer_name, model_name, hls_project_path
):
    # Generate layers files
    if "Swish" in layers_type:
        generate_swish_files(layer_name, layers_config, model_name)
    elif "Relu" in layers_type:
        generate_relu_files(layer_name, layers_config, model_name)
    elif "Sigmoid" in layers_type:
        generate_sigmoid_files(layer_name, layers_config, model_name)
    elif "Add" in layers_type or "Mul" in layers_type:
        generate_elemwise_files(layer_name, layers_config, model_name)
    elif "Conv" in layers_type:
        generate_conv_files(
            layer_name, layers_config, model_name, hls_project_path
        )
    elif "Pooling" in layers_type:
        # TODO: Implement pooling
        pass
    elif "GlobalAveragePool" in layers_type:
        generate_gap_files(layer_name, layers_config, model_name)
    elif "Gemm" in layers_type:
        generate_gemm_files(layer_name, layers_config, model_name)
    elif "Activation" in layers_type:
        # TODO: Implement DYNAMIC reconfigurable activation
        pass
    elif "ElementWise" in layers_type:
        # TODO: Implement DYNAMIC reconfigurable elementWise
        pass
    else:
        raise Exception(f"Layer {layers_type} not supported")

    # Generate top level layer file
    # TODO: Create a script to provide support for all the types of layers
    # generate_top_level_files(graph, branch_depth, layers_config, layer_name, prefix)

    # Generate testbench file
    generate_tb_files(layer_name, model_name, hls_project_path, is_layer=True)

    # Generate testbench data
    if "Swish" in layers_type:
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        coarse_factor = layers_config["coarse_factor"]
        shish_3d(input_shape=shape_in,
                 coarse_in=coarse_factor,
                 file_format="bin",
                 store_path=os.path.join(os.getcwd(), "generated_files", model_name, layer_name, "data"))
    elif "Relu" in layers_type:
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        coarse_factor = layers_config["coarse_factor"]
        relu_3d(input_shape=shape_in,
                 coarse_in=coarse_factor,
                 file_format="bin",
                 store_path=os.path.join(os.getcwd(), "generated_files", model_name, layer_name, "data"))
    elif "Sigmoid" in layers_type:
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        coarse_factor = layers_config["coarse_factor"]
        sigmoid_3d(input_shape=shape_in,
                    coarse_in=coarse_factor,
                    file_format="bin",
                    store_path=os.path.join(os.getcwd(), "generated_files", model_name, layer_name, "data"))
    elif "Add" in layers_type or "Mul" in layers_type:
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        coarse_factor = layers_config["coarse_factor"]
        op_type = layers_config["op_type"]
        broadcasting = layers_config["broadcasting"]
        shape_in_2 = shape_in
        if broadcasting:
            shape_in_2[2], shape_in_2[3], shape_in_2[4] = 1, 1, 1
        elemwise_3d(input_shape=shape_in,
                    input_shape_2=shape_in_2,
                    coarse_in=coarse_factor,
                    elemwise_op_type=op_type,
                    file_format="bin",
                    store_path=os.path.join(os.getcwd(), "generated_files", model_name, layer_name, "data"))
    elif "Conv" in layers_type:
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        filters = layers_config["channels_out"]
        shape_kernel = [layers_config["kernel_depth"],
                        layers_config["kernel_height"],
                        layers_config["kernel_width"]]
        padding = [layers_config["pad_depth"],
                   layers_config["pad_height"],
                     layers_config["pad_width"]]
        stride = [layers_config["stride_depth"],
                  layers_config["stride_height"],
                    layers_config["stride_width"]]
        groups = layers_config["groups"]
        depthwise = layers_config["depthwise"]
        coarse_in_factor = layers_config["coarse_in_factor"]
        coarse_out_factor = layers_config["coarse_out_factor"]

        conv_3d(
            input_shape=shape_in,
            kernel_shape=shape_kernel,
            bias=False,
            filters=filters,
            padding=padding,
            stride=stride,
            groups=groups,
            depthwise=depthwise,
            coarse_in=coarse_in_factor,
            coarse_out=coarse_out_factor,
            file_format="bin",
            store_path=os.path.join(os.getcwd(), "generated_files", model_name, layer_name, "data"),
            layer_name = layer_name.lower(),
        )
    elif "Pooling" in layers_type:
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        shape_kernel = [layers_config["kernel_depth"],
                        layers_config["kernel_height"],
                        layers_config["kernel_width"]]
        padding = [layers_config["pad_depth"],
                   layers_config["pad_height"],
                     layers_config["pad_width"]]
        stride = [layers_config["stride_depth"],
                  layers_config["stride_height"],
                    layers_config["stride_width"]]
        op_type = layers_config["op_type"]
        coarse_factor = layers_config["coarse_factor"]

        pool_3d(
            input_shape=shape_in,
            kernel_shape=shape_kernel,
            padding=padding,
            stride=stride,
            coarse_in=coarse_factor,
            pool_op_type=op_type,
            file_format="bin",
            store_path=os.path.join(os.getcwd(), "generated_files", model_name, layer_name, "data"),
        )
    elif "GlobalAveragePool" in layers_type:
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        coarse_factor = layers_config["coarse_factor"]
        gap_3d(input_shape=shape_in,
               coarse_in=coarse_factor,
               file_format="bin",
               store_path=os.path.join(os.getcwd(), "generated_files", model_name, layer_name, "data"))
    elif "Gemm" in layers_type:
        shape_in = [layers_config["batch_size"],
                    layers_config["features_in"]]
        shape_out = [layers_config["batch_size"],
                     layers_config["features_out"]]
        coarse_in_factor = layers_config["coarse_in_factor"]
        coarse_out_factor = layers_config["coarse_out_factor"]
        gemm(
            input_shape=shape_in,
            output_shape=shape_out,
            coarse_in=coarse_in_factor,
            coarse_out=coarse_out_factor,
            bias=False,
            file_format="bin",
            store_path=os.path.join(os.getcwd(), "generated_files", model_name, layer_name, "data"),
            layer_name = layer_name.lower(),
        )
    elif "Activation" in layers_type:
        # TODO: Implement DYNAMIC reconfigurable activation
        pass
    elif "ElementWise" in layers_type:
        # TODO: Implement DYNAMIC reconfigurable elementWise
        pass
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

    if args.config_file:
        layer_configuration = get_layers_configurations(args.config_file)
    else:
        layer_configuration = get_layers_configurations(os.path.join(os.getcwd(), "fpga_modeling_reports", args.model_name, f"{args.model_name}_layers.json"))

    for k, v in layer_configuration.items():
        print(f"Generating data for layer {k} of type {v['type']}")
        if args.config_file:
            generate_layer_code(
                v["config"], v["type"], k, "custom_layers", args.hls_project_path
            )
        else:
            generate_layer_code(
                v["config"], v["type"], k, args.model_name, args.hls_project_path
            )
