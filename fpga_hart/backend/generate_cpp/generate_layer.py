import argparse
import json
import os

import pandas as pd
from generate_tb import generate_tb_files
from generate_top_level_layer import generate_top_level_layer_files
from layers.generate_conv import generate_conv_files
from layers.generate_elemwise import generate_elemwise_files
from layers.generate_gap import generate_gap_files
from layers.generate_gemm import generate_gemm_files
from layers.generate_pool import generate_pool_files
from layers.generate_relu import generate_relu_files
from layers.generate_sigmoid import generate_sigmoid_files
from layers.generate_swish import generate_swish_files

from fpga_hart.backend.python_prototyping.generate_data import (conv_3d,
                                                                elemwise_3d,
                                                                gap_3d, gemm,
                                                                pool_3d,
                                                                relu_3d,
                                                                shish_3d,
                                                                sigmoid_3d)


def parse_args():
    parser = argparse.ArgumentParser(description="fpga_hart toolflow parser")
    parser.add_argument(
        "model_name",
        choices=["x3d_m", "slowonly", "r2plus1d", "c3d"],
        type=str,
        help="name of the HAR model",
    )
    parser.add_argument(
        "target",
        choices=["throughput", "latency"],
        type=str,
        help="target of the optimization",
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
        if "Type" in configuration[l].index.to_list():
            layers_type = configuration[l]["Type"]
        else:
            layers_type = "DR_"
            if "Activation" in l:
                layers_type += "Activation"
            elif "GlobalAveragePool" in l:
                layers_type += "GlobalAveragePool"
            elif "Gemm" in l:
                layers_type += "Gemm"
            elif "Conv" in l:
                layers_type += "Conv"
            elif "ElementWise" in l:
                layers_type += "ElementWise"
            elif "Pooling" in l:
                layers_type += "Pooling"
            else:
                raise ValueError(f"Layer type {l} not recognized")
        layers_config = configuration[l]["config"]
        result[l] = {
            "config": layers_config,
            "type": layers_type,
        }

    return result


def generate_layer_code(
    layers_config, layers_type, layer_name, model_name, hls_project_path
):
    dynamic_reconfig = True if "DR_" in layers_type else False
    elem_bc = layers_config["broadcasting"] if "broadcasting" in layers_config else False
    # Generate layers files
    if layers_type == "Swish":
        generate_swish_files(layer_name, layers_config, model_name)
    elif layers_type == "Relu":
        generate_relu_files(layer_name, layers_config, model_name)
    elif layers_type == "Sigmoid":
        generate_sigmoid_files(layer_name, layers_config, model_name)
    elif layers_type == "Add" or layers_type == "Mul":
        generate_elemwise_files(layer_name, layers_config, model_name)
    elif layers_type == "Conv":
        generate_conv_files(
            layer_name, layers_config, model_name, hls_project_path
        )
    elif layers_type == "MaxPool" or layers_type == "AveragePool":
        generate_pool_files(layer_name, layers_config, model_name)
    elif layers_type == "GlobalAveragePool":
        layer_name = "Gap_" + layer_name.split("_")[1]
        generate_gap_files(layer_name, layers_config, model_name)
    elif layers_type == "Gemm":
        generate_gemm_files(layer_name, layers_config, model_name)
    elif layers_type == "DR_Activation":
        # TODO: Implement DYNAMIC reconfigurable activation
        # generate_dr_activation_files(layer_name, layers_config, model_name)
        print("DYNAMIC reconfigurable activation not implemented yet")
        pass
    elif layers_type == "DR_Conv":
        # TODO: Implement DYNAMIC reconfigurable conv
        # generate_dr_conv_files(layer_name, layers_config, model_name, hls_project_path)
        print("DYNAMIC reconfigurable conv not implemented yet")
        pass
    elif layers_type == "DR_ElementWise":
        # TODO: Implement DYNAMIC reconfigurable elementwise
        # generate_dr_elemwise_files(layer_name, layers_config, model_name)
        print("DYNAMIC reconfigurable elementwise not implemented yet")
        pass
    elif layers_type == "DR_Pooling":
        # TODO: Implement DYNAMIC reconfigurable pool
        # generate_dr_pool_files(layer_name, layers_config, model_name)
        print("DYNAMIC reconfigurable pool not implemented yet")
        pass
    elif layers_type == "DR_GlobalAveragePool":
        layer_name = "Gap"
        generate_gap_files(layer_name, layers_config, model_name, dynamic_reconfig=dynamic_reconfig)
    elif layers_type == "DR_Gemm":
        generate_gemm_files(layer_name, layers_config, model_name, dynamic_reconfig=dynamic_reconfig)
    else:
        raise Exception(f"Layer {layers_type} not supported")

    # Generate top level layer file
    # TODO: Create a script to provide support for all the types of layers
    generate_top_level_layer_files(layer_name, model_name, dynamic_reconfig=dynamic_reconfig)

    # Generate testbench file
    generate_tb_files(layer_name, model_name, hls_project_path, is_layer=True, dynamic_reconfig=dynamic_reconfig, elem_bc=elem_bc)

    # Generate testbench data
    if dynamic_reconfig:
        store_path = os.path.join(os.getcwd(), "generated_files", model_name, "latency_driven", layer_name, "data")
    else:
        store_path = os.path.join(os.getcwd(), "generated_files", model_name, layer_name, "data")
    if layers_type == "Swish":
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        coarse_factor = layers_config["coarse_factor"]
        shish_3d(input_shape=shape_in,
                 coarse_in=coarse_factor,
                 file_format="bin",
                 store_path=store_path)
    elif layers_type == "Relu":
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        coarse_factor = layers_config["coarse_factor"]
        relu_3d(input_shape=shape_in,
                 coarse_in=coarse_factor,
                 file_format="bin",
                 store_path=store_path)
    elif layers_type == "Sigmoid":
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        coarse_factor = layers_config["coarse_factor"]
        sigmoid_3d(input_shape=shape_in,
                    coarse_in=coarse_factor,
                    file_format="bin",
                    store_path=store_path)
    elif layers_type == "Add" or layers_type == "Mul":
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        coarse_factor = layers_config["coarse_factor"]
        op_type = layers_config["op_type"]
        broadcasting = layers_config["broadcasting"]
        shape_in_2 = shape_in.copy()
        if broadcasting:
            shape_in_2[2], shape_in_2[3], shape_in_2[4] = 1, 1, 1
        elemwise_3d(input_shape=shape_in,
                    input_shape_2=shape_in_2,
                    coarse_in=coarse_factor,
                    elemwise_op_type=op_type,
                    file_format="bin",
                    store_path=store_path)
    elif layers_type == "Conv":
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        filters = layers_config["channels_out"]
        shape_kernel = [layers_config["kernel_depth"],
                        layers_config["kernel_height"],
                        layers_config["kernel_width"]]
        shape_bias = layers_config["shape_bias"]
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
            bias=False if shape_bias == 0 else True,
            filters=filters,
            padding=padding,
            stride=stride,
            groups=groups,
            depthwise=depthwise,
            coarse_in=coarse_in_factor,
            coarse_out=coarse_out_factor,
            file_format="bin",
            store_path=store_path,
            layer_name = layer_name.lower(),
        )
    elif layers_type == "MaxPool" or layers_type == "AvgPool":
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
            store_path=store_path,
        )
    elif layers_type == "GlobalAveragePool":
        shape_in = [layers_config["batch_size"],
                    layers_config["channels_in"],
                    layers_config["depth_in"],
                    layers_config["height_in"],
                    layers_config["width_in"]]
        coarse_factor = layers_config["coarse_factor"]
        gap_3d(input_shape=shape_in,
               coarse_in=coarse_factor,
               file_format="bin",
               store_path=store_path)
    elif layers_type == "Gemm":
        shape_in = [layers_config["batch_size"],
                    layers_config["features_in"]]
        shape_out = [layers_config["batch_size"],
                     layers_config["features_out"]]
        shape_bias = layers_config["shape_bias"]
        coarse_in_factor = layers_config["coarse_in_factor"]
        coarse_out_factor = layers_config["coarse_out_factor"]
        gemm(
            input_shape=shape_in,
            output_shape=shape_out,
            coarse_in=coarse_in_factor,
            coarse_out=coarse_out_factor,
            bias=False if shape_bias == 0 else True,
            file_format="bin",
            store_path=store_path,
            layer_name = layer_name.lower(),
        )
    elif layers_type == "DR_Activation":
        # TODO: Implement DYNAMIC reconfigurable activation
        pass
    elif layers_type == "DR_Conv":
        # TODO: Implement DYNAMIC reconfigurable conv
        pass
    elif layers_type == "DR_ElementWise":
        # TODO: Implement DYNAMIC reconfigurable elementwise
        pass
    elif layers_type == "DR_Pooling":
        # TODO: Implement DYNAMIC reconfigurable pool
        pass
    elif layers_type == "DR_GlobalAveragePool":
        # TODO: Implement DYNAMIC reconfigurable GlobalAveragePool
        pass
    elif layers_type == "DR_Gemm":
        # TODO: Implement DYNAMIC reconfigurable gemm
        pass
    else:
        raise Exception(f"Layer {layers_type} not supported")

if __name__ == "__main__":
    args = parse_args()

    if args.config_file:
        layer_configuration = get_layers_configurations(args.config_file)
    else:
        if args.target == "throughput":
            layer_configuration = get_layers_configurations(os.path.join(os.getcwd(), "fpga_modeling_reports", args.model_name, f"{args.model_name}_layers.json"))
        elif args.target == "latency":
            layer_configuration = get_layers_configurations(os.path.join(os.getcwd(), "fpga_modeling_reports", args.model_name, "latency_driven", "config.json"))

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
