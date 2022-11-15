import os

import numpy as np

from .codegen import *


def generate_squeeze_cpp(name: str, config: dict, model_name: str, partition_name: str):

    batch_size = config["batch_size"]
    channels = config["channels_out"]
    depth = config["depth_out"]
    height = config["height_out"]
    width = config["width_out"]

    node_in_coarse_factor = config['coarse_in_factor']
    node_out_coarse_factor = config['coarse_out_factor']

    squeeze_buffer = np.lcm(node_in_coarse_factor, node_out_coarse_factor)

    layer_name_lower = name.replace("GlobalAveragePool", "GAP").lower()
    layer_name_upper = name.replace("GlobalAveragePool", "GAP").upper()

    if partition_name != '':
        cpp = CppFile(
            os.path.join(
                os.getcwd(),
                "generated_files",
                model_name, partition_name, "src",
                f"{layer_name_lower}.cpp",
            )
        )
    else:
        cpp = CppFile(
            os.path.join(
                os.getcwd(),
                "generated_files",
                model_name, partition_name, name, "src",
                f"{layer_name_lower}.cpp",
            )
        )

    cpp(
        f'#include "{layer_name_lower}.hpp"',
        newlines=2,
    )

    with cpp.block(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE_IN],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE_OUT])"
    ):

        cpp("#pragma HLS INLINE OFF")

        cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

        cpp("#pragma HLS DATAFLOW", newlines=2)

        cpp(
            f"squeeze_3d<\n\
            {layer_name_upper}_SQUEEZE_BATCH_SIZE,\n\
            {layer_name_upper}_SQUEEZE_CHANNELS,\n\
            {layer_name_upper}_SQUEEZE_HEIGHT,\n\
            {layer_name_upper}_SQUEEZE_WIDTH,\n\
            {layer_name_upper}_SQUEEZE_DEPTH,\n\
            {layer_name_upper}_SQUEEZE_COARSE_IN,\n\
            {layer_name_upper}_SQUEEZE_COARSE_OUT,\n\
            {layer_name_upper}_SQUEEZE_BUFFER,\n\
            {layer_name_lower}_data_t\n\
        >(in,out);",
            newlines=2,
        )

    cpp.close()


def generate_squeeze_hpp(name: str, config: dict, model_name: str, partition_name: str):

    batch_size = config["batch_size"]
    channels = config["channels_out"]
    depth = config["depth_out"]
    height = config["height_out"]
    width = config["width_out"]

    node_in_coarse_factor = config['coarse_in_factor']
    node_out_coarse_factor = config['coarse_out_factor']

    squeeze_buffer = np.lcm(node_in_coarse_factor, node_out_coarse_factor)

    layer_name_lower = name.replace("GlobalAveragePool", "GAP").lower()
    layer_name_upper = name.replace("GlobalAveragePool", "GAP").upper()

    if partition_name != '':
        hpp = CppFile(
            os.path.join(
                os.getcwd(),
                "generated_files",
                model_name, partition_name, "src",
                f"{layer_name_lower}.hpp",
            )
        )
    else:
        hpp = CppFile(
            os.path.join(
                os.getcwd(),
                "generated_files",
                model_name, partition_name, name, "src",
                f"{layer_name_lower}.hpp",
            )
        )

    hpp("#pragma once", newlines=2)
    hpp('#include "common_.hpp"')
    hpp('#include "squeeze_3d_.hpp"', newlines=2)

    hpp(
        f"#define {layer_name_upper}_BATCH_SIZE {batch_size}"
    )
    hpp(
        f"#define {layer_name_upper}_CHANNELS {channels}"
    )
    hpp(f"#define {layer_name_upper}_DEPTH {depth}")
    hpp(f"#define {layer_name_upper}_HEIGHT {height}")
    hpp(
        f"#define {layer_name_upper}_WIDTH {width}",
        newlines=2,
    )

    hpp(
        f"#define {layer_name_upper}_COARSE_IN {node_in_coarse_factor}"
    )
    hpp(
        f"#define {layer_name_upper}_COARSE_OUT {node_out_coarse_factor}",
        newlines=2,
    )

    hpp(
        f"#define {layer_name_upper}_SQUEEZE_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE"
    )
    hpp(
        f"#define {layer_name_upper}_SQUEEZE_CHANNELS \t{layer_name_upper}_CHANNELS"
    )
    hpp(
        f"#define {layer_name_upper}_SQUEEZE_DEPTH \t{layer_name_upper}_DEPTH"
    )
    hpp(
        f"#define {layer_name_upper}_SQUEEZE_HEIGHT \t{layer_name_upper}_HEIGHT"
    )
    hpp(
        f"#define {layer_name_upper}_SQUEEZE_WIDTH \t{layer_name_upper}_WIDTH"
    )
    hpp(
        f"#define {layer_name_upper}_SQUEEZE_COARSE_IN \t{layer_name_upper}_COARSE_IN"
    )
    hpp(
        f"#define {layer_name_upper}_SQUEEZE_COARSE_OUT \t{layer_name_upper}_COARSE_OUT"
    )
    hpp(
        f"#define {layer_name_upper}_SQUEEZE_BUFFER \t{squeeze_buffer}",
        newlines=2,
    )

    hpp(
        f"typedef ap_fixed<16,8,AP_RND, AP_SAT> \t{layer_name_lower}_data_t;",
        newlines=3,
    )

    hpp(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE_IN],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE_OUT]);"
    )

    hpp.close()


def generate_squeeze_files(name: str, config: dict, model_name: str, partition_name: str = ''):
    if partition_name != '':
        if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "src")):
            os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "src"))
    else:
        if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, name, "src")):
            os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, name, "src"))

    generate_squeeze_hpp(name, config, model_name, partition_name)
    generate_squeeze_cpp(name, config, model_name, partition_name)
