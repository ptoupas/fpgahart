import os

from .codegen import *


def generate_gap_cpp(name: str, config: dict, model_name: str, partition_name: str, dynamic_reconfig: bool):
    batch_size = config["batch_size"]
    channels = config["channels_in"]
    depth = config["depth_in"]
    height = config["height_in"]
    width = config["width_in"]
    coarse_factor = config["coarse_factor"]

    layer_name_lower = name.lower()
    layer_name_upper = name.upper()

    if dynamic_reconfig:
        cpp = CppFile(
            os.path.join(
                os.getcwd(), "generated_files", model_name, partition_name, "latency_driven", name, "src", f"{layer_name_lower}.cpp"
            )
        )
    else:
        if partition_name != '':
            cpp = CppFile(
                os.path.join(
                    os.getcwd(), "generated_files", model_name, partition_name, "src", f"{layer_name_lower}.cpp"
                )
            )
        else:
            cpp = CppFile(
                os.path.join(
                    os.getcwd(), "generated_files", model_name, partition_name, name, "src", f"{layer_name_lower}.cpp"
                )
            )

    cpp(f'#include "{layer_name_lower}.hpp"', newlines=2)

    with cpp.block(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE])"
    ):

        cpp("#pragma HLS INLINE OFF")

        cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

        cpp("#pragma HLS DATAFLOW", newlines=2)

        with cpp.block(
            f"for(int coarseIndex=0; coarseIndex<{layer_name_upper}_COARSE; coarseIndex++)"
        ):
            cpp("#pragma HLS unroll", newlines=2)

            cpp(
                f"gap_3d<\n\
                {layer_name_upper}_GAP_BATCH_SIZE,\n\
                {layer_name_upper}_GAP_CHANNELS,\n\
                {layer_name_upper}_GAP_HEIGHT,\n\
                {layer_name_upper}_GAP_WIDTH,\n\
                {layer_name_upper}_GAP_DEPTH,\n\
                {layer_name_lower}_data_t,\n\
                accum_data_t\n\
            >(in[coarseIndex],out[coarseIndex]);",
                newlines=2,
            )

    cpp.close()


def generate_gap_hpp(name: str, config: dict, model_name: str, partition_name: str, dynamic_reconfig: bool):
    batch_size = config["batch_size"]
    channels = config["channels_in"]
    depth = config["depth_in"]
    height = config["height_in"]
    width = config["width_in"]
    coarse_factor = config["coarse_factor"]

    layer_name_lower = name.lower()
    layer_name_upper = name.upper()

    if dynamic_reconfig:
        hpp = CppFile(
            os.path.join(
                os.getcwd(), "generated_files", model_name, partition_name, "latency_driven", name, "src", f"{layer_name_lower}.hpp"
            )
        )
    else:
        if partition_name != '':
            hpp = CppFile(
                os.path.join(
                    os.getcwd(), "generated_files", model_name, partition_name, "src", f"{layer_name_lower}.hpp"
                )
            )
        else:
            hpp = CppFile(
                os.path.join(
                    os.getcwd(), "generated_files", model_name, partition_name, name, "src", f"{layer_name_lower}.hpp"
                )
            )

    hpp("#pragma once", newlines=2)
    hpp('#include "common_.hpp"')
    hpp('#include "gap_3d_.hpp"', newlines=2)

    hpp(f"#define {layer_name_upper}_BATCH_SIZE {batch_size}")
    hpp(f"#define {layer_name_upper}_CHANNELS {channels}")
    hpp(f"#define {layer_name_upper}_DEPTH {depth}")
    hpp(f"#define {layer_name_upper}_HEIGHT {height}")
    hpp(f"#define {layer_name_upper}_WIDTH {width}", newlines=2)

    hpp(f"#define {layer_name_upper}_COARSE {coarse_factor}", newlines=2)

    hpp(f"#define {layer_name_upper}_GAP_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE")
    hpp(
        f"#define {layer_name_upper}_GAP_CHANNELS \tDIVIDE({layer_name_upper}_CHANNELS, {layer_name_upper}_COARSE)"
    )
    hpp(f"#define {layer_name_upper}_GAP_DEPTH \t{layer_name_upper}_DEPTH")
    hpp(f"#define {layer_name_upper}_GAP_HEIGHT \t{layer_name_upper}_HEIGHT")
    hpp(f"#define {layer_name_upper}_GAP_WIDTH \t{layer_name_upper}_WIDTH", newlines=2)

    hpp(
        f"typedef ap_fixed<16,8,AP_RND, AP_SAT> \t{layer_name_lower}_data_t;",
        newlines=3,
    )

    hpp(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE]);"
    )

    hpp.close()


def generate_gap_files(name: str, config: dict, model_name: str, partition_name: str = '', dynamic_reconfig: bool=False):
    if dynamic_reconfig:
        if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "latency_driven", name, "src")):
            os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "latency_driven", name, "src"))
    else:
        if partition_name != '':
            if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "src")):
                os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "src"))
        else:
            if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, name, "src")):
                os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, name, "src"))

    generate_gap_hpp(name, config, model_name, partition_name, dynamic_reconfig)
    generate_gap_cpp(name, config, model_name, partition_name, dynamic_reconfig)
