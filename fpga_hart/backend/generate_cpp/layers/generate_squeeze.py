import os

import numpy as np

from .codegen import *


def get_node_coarse_factor(config, mode="in"):
    if "coarse_factor" in config.keys():
        return config["coarse_factor"]
    if mode == "in":
        return config["coarse_in_factor"]
    elif mode == "out":
        depthwise = config["depthwise"]
        return config["coarse_out_factor"]


def generate_squeeze_cpp(in_name, in_config, out_name, out_config, partition_name):
    if "broadcasting" in in_config.keys() or "broadcasting" in out_config.keys():
        assert (
            in_config["channels_out"] == out_config["channels_in"]
        ), "Output shape of the first node must be the same as input shape of the second node"
    else:
        assert (
            [in_config["channels_out"], in_config["depth_out"], in_config["height_out"], in_config["width_out"]] == [out_config["channels_in"], out_config["depth_in"], out_config["height_in"], out_config["width_in"]]
        ), "Output shape of the first node must be the same as input shape of the second node"
    batch_size = in_config["batch_size"]
    channels = in_config["channels_out"]
    depth = in_config["depth_out"]
    height = in_config["height_out"]
    width = in_config["width_out"]

    node_in_coarse_factor = get_node_coarse_factor(in_config, mode="out")
    node_out_coarse_factor = get_node_coarse_factor(out_config, mode="in")

    layer_in_name_lower = in_name.replace("GlobalAveragePool", "GAP").lower()
    layer_in_name_upper = in_name.replace("GlobalAveragePool", "GAP").upper()
    layer_out_name_lower = out_name.replace("GlobalAveragePool", "GAP").lower()
    layer_out_name_upper = out_name.replace("GlobalAveragePool", "GAP").upper()

    cpp = CppFile(
        os.path.join(
            os.getcwd(),
            "generated_files",
            partition_name,
            f"squeeze_{layer_in_name_lower}_{layer_out_name_lower}.cpp",
        )
    )

    cpp(
        f'#include "squeeze_{layer_in_name_lower}_{layer_out_name_lower}.hpp"',
        newlines=2,
    )

    with cpp.block(
        f"void squeeze_{layer_in_name_lower}_{layer_out_name_lower}_layer(\n\
        stream_t(squeeze_{layer_in_name_lower}_{layer_out_name_lower}_data_t) in[SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_COARSE_IN],\n\
        stream_t(squeeze_{layer_in_name_lower}_{layer_out_name_lower}_data_t) out[SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_COARSE_OUT])"
    ):

        cpp("#pragma HLS INLINE OFF")
        cpp("#pragma HLS DATAFLOW", newlines=2)

        cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

        cpp(
            f"squeeze_3d<\n\
            SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_BATCH_SIZE,\n\
            SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_CHANNELS,\n\
            SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_HEIGHT,\n\
            SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_WIDTH,\n\
            SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_DEPTH,\n\
            SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_COARSE_IN,\n\
            SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_COARSE_OUT,\n\
            SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_BUFFER,\n\
            squeeze_{layer_in_name_lower}_{layer_out_name_lower}_data_t\n\
        >(in,out);",
            newlines=2,
        )

    cpp.close()


def generate_squeeze_hpp(in_name, in_config, out_name, out_config, partition_name):
    if "broadcasting" in in_config.keys() or "broadcasting" in out_config.keys():
        assert (
            in_config["channels_out"] == out_config["channels_in"]
        ), "Output shape of the first node must be the same as input shape of the second node"
    else:
        assert (
            [in_config["channels_out"], in_config["depth_out"], in_config["height_out"], in_config["width_out"]] == [out_config["channels_in"], out_config["depth_in"], out_config["height_in"], out_config["width_in"]]
        ), "Output shape of the first node must be the same as input shape of the second node"
    batch_size = in_config["batch_size"]
    channels = in_config["channels_out"]
    depth = in_config["depth_out"]
    height = in_config["height_out"]
    width = in_config["width_out"]

    node_in_coarse_factor = get_node_coarse_factor(in_config, mode="out")
    node_out_coarse_factor = get_node_coarse_factor(out_config, mode="in")

    squeeze_buffer = np.lcm(node_in_coarse_factor, node_out_coarse_factor)

    layer_in_name_lower = in_name.replace("GlobalAveragePool", "GAP").lower()
    layer_in_name_upper = in_name.replace("GlobalAveragePool", "GAP").upper()
    layer_out_name_lower = out_name.replace("GlobalAveragePool", "GAP").lower()
    layer_out_name_upper = out_name.replace("GlobalAveragePool", "GAP").upper()

    hpp = CppFile(
        os.path.join(
            os.getcwd(),
            "generated_files",
            partition_name,
            f"squeeze_{layer_in_name_lower}_{layer_out_name_lower}.hpp",
        )
    )

    hpp("#pragma once", newlines=2)
    hpp('#include "common_.hpp"')
    hpp('#include "squeeze_3d_.hpp"', newlines=2)

    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_BATCH_SIZE {batch_size}"
    )
    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_CHANNELS {channels}"
    )
    hpp(f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_DEPTH {depth}")
    hpp(f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_HEIGHT {height}")
    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_WIDTH {width}",
        newlines=2,
    )

    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_COARSE_IN {node_in_coarse_factor}"
    )
    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_COARSE_OUT {node_out_coarse_factor}",
        newlines=2,
    )

    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_BATCH_SIZE \tSQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_BATCH_SIZE"
    )
    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_CHANNELS \tSQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_CHANNELS"
    )
    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_DEPTH \tSQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_DEPTH"
    )
    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_HEIGHT \tSQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_HEIGHT"
    )
    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_WIDTH \tSQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_WIDTH"
    )
    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_COARSE_IN \tSQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_COARSE_IN"
    )
    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_COARSE_OUT \tSQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_COARSE_OUT"
    )
    hpp(
        f"#define SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_SQUEEZE_BUFFER \t{squeeze_buffer}",
        newlines=2,
    )

    hpp(
        f"typedef ap_fixed<16,8,AP_RND, AP_SAT> \tsqueeze_{layer_in_name_lower}_{layer_out_name_lower}_data_t;",
        newlines=3,
    )

    hpp(
        f"void squeeze_{layer_in_name_lower}_{layer_out_name_lower}_layer(\n\
        stream_t(squeeze_{layer_in_name_lower}_{layer_out_name_lower}_data_t) in[SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_COARSE_IN],\n\
        stream_t(squeeze_{layer_in_name_lower}_{layer_out_name_lower}_data_t) out[SQUEEZE_{layer_in_name_upper}_{layer_out_name_upper}_COARSE_OUT]);"
    )

    hpp.close()


def generate_squeeze_files(in_name, in_config, out_name, out_config, partition_name):
    if not os.path.exists(os.path.join(os.getcwd(), "generated_files", partition_name)):
        os.makedirs(os.path.join(os.getcwd(), "generated_files", partition_name))

    generate_squeeze_hpp(in_name, in_config, out_name, out_config, partition_name)
    generate_squeeze_cpp(in_name, in_config, out_name, out_config, partition_name)
