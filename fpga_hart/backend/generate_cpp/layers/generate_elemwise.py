import os

from .codegen import *


def generate_elemwise_cpp(name, config, partition_name):
    batch_size = config["batch_size"]
    channels = config["channels_in"]
    depth = config["depth_in"]
    height = config["height_in"]
    width = config["width_in"]
    coarse_factor = config["coarse_factor"]
    broadcasting = config["broadcasting"]
    op_type = config["op_type"].upper()

    layer_name_lower = name.lower()
    layer_name_upper = name.upper()

    cpp = CppFile(
        os.path.join(
            os.getcwd(), "generated_files", partition_name, f"{layer_name_lower}.cpp"
        )
    )

    cpp(f'#include "{layer_name_lower}.hpp"', newlines=2)

    with cpp.block(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in_1[{layer_name_upper}_COARSE],\n\
        stream_t({layer_name_lower}_data_t) in_2[{layer_name_upper}_COARSE],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE])"
    ):

        cpp("#pragma HLS INLINE OFF")
        cpp("#pragma HLS DATAFLOW", newlines=2)

        cpp("#pragma HLS ARRAY_PARTITION variable=in_1  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=in_2  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

        with cpp.block(
            f"for(int coarseIndex=0; coarseIndex<{layer_name_upper}_COARSE; coarseIndex++)"
        ):
            cpp("#pragma HLS unroll", newlines=2)

            if broadcasting == 0 and op_type == "ADD":
                cpp(
                    f"add_3d<\n\
                    {layer_name_upper}_{op_type}_BATCH_SIZE,\n\
                    {layer_name_upper}_{op_type}_CHANNELS,\n\
                    {layer_name_upper}_{op_type}_HEIGHT,\n\
                    {layer_name_upper}_{op_type}_WIDTH,\n\
                    {layer_name_upper}_{op_type}_DEPTH,\n\
                    {layer_name_lower}_data_t,\n\
                    accum_data_t\n\
                >(in_1[coarseIndex],in_2[coarseIndex],out[coarseIndex]);",
                    newlines=2,
                )
            elif broadcasting == 1 and op_type == "ADD":
                cpp(
                    f"add_bc_3d<\n\
                    {layer_name_upper}_{op_type}_BATCH_SIZE,\n\
                    {layer_name_upper}_{op_type}_CHANNELS,\n\
                    {layer_name_upper}_{op_type}_HEIGHT,\n\
                    {layer_name_upper}_{op_type}_WIDTH,\n\
                    {layer_name_upper}_{op_type}_DEPTH,\n\
                    {layer_name_lower}_data_t,\n\
                    accum_data_t\n\
                >(in_1[coarseIndex],in_2[coarseIndex],out[coarseIndex]);",
                    newlines=2,
                )
            elif broadcasting == 0 and op_type == "MUL":
                cpp(
                    f"mul_3d<\n\
                    {layer_name_upper}_{op_type}_BATCH_SIZE,\n\
                    {layer_name_upper}_{op_type}_CHANNELS,\n\
                    {layer_name_upper}_{op_type}_HEIGHT,\n\
                    {layer_name_upper}_{op_type}_WIDTH,\n\
                    {layer_name_upper}_{op_type}_DEPTH,\n\
                    {layer_name_lower}_data_t,\n\
                    accum_data_t\n\
                >(in_1[coarseIndex],in_2[coarseIndex],out[coarseIndex]);",
                    newlines=2,
                )
            elif broadcasting == 1 and op_type == "MUL":
                cpp(
                    f"mul_bc_3d<\n\
                    {layer_name_upper}_{op_type}_BATCH_SIZE,\n\
                    {layer_name_upper}_{op_type}_CHANNELS,\n\
                    {layer_name_upper}_{op_type}_HEIGHT,\n\
                    {layer_name_upper}_{op_type}_WIDTH,\n\
                    {layer_name_upper}_{op_type}_DEPTH,\n\
                    {layer_name_lower}_data_t,\n\
                    accum_data_t\n\
                >(in_1[coarseIndex],in_2[coarseIndex],out[coarseIndex]);",
                    newlines=2,
                )
            else:
                raise Exception("Invalid op_type")

    cpp.close()


def generate_elemwise_hpp(name, config, partition_name):
    batch_size = config["batch_size"]
    channels = config["channels_in"]
    depth = config["depth_in"]
    height = config["height_in"]
    width = config["width_in"]
    coarse_factor = config["coarse_factor"]
    broadcasting = config["broadcasting"]
    op_type = config["op_type"].upper()

    layer_name_lower = name.lower()
    layer_name_upper = name.upper()

    hpp = CppFile(
        os.path.join(
            os.getcwd(), "generated_files", partition_name, f"{layer_name_lower}.hpp"
        )
    )

    hpp("#pragma once", newlines=2)
    hpp('#include "common_.hpp"')
    hpp('#include "elemwise_3d_.hpp"', newlines=2)

    hpp(f"#define {layer_name_upper}_BATCH_SIZE {batch_size}")
    hpp(f"#define {layer_name_upper}_CHANNELS {channels}")
    hpp(f"#define {layer_name_upper}_DEPTH {depth}")
    hpp(f"#define {layer_name_upper}_HEIGHT {height}")
    hpp(f"#define {layer_name_upper}_WIDTH {width}", newlines=2)

    hpp(f"#define {layer_name_upper}_COARSE {coarse_factor}", newlines=2)

    hpp(
        f"#define {layer_name_upper}_{op_type}_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE"
    )
    hpp(
        f"#define {layer_name_upper}_{op_type}_CHANNELS \tDIVIDE({layer_name_upper}_CHANNELS, {layer_name_upper}_COARSE)"
    )
    hpp(f"#define {layer_name_upper}_{op_type}_DEPTH \t{layer_name_upper}_DEPTH")
    hpp(f"#define {layer_name_upper}_{op_type}_HEIGHT \t{layer_name_upper}_HEIGHT")
    hpp(
        f"#define {layer_name_upper}_{op_type}_WIDTH \t{layer_name_upper}_WIDTH",
        newlines=2,
    )

    hpp(
        f"typedef ap_fixed<16,8,AP_RND, AP_SAT> \t{layer_name_lower}_data_t;",
        newlines=3,
    )

    hpp(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in_1[{layer_name_upper}_COARSE],\n\
        stream_t({layer_name_lower}_data_t) in_2[{layer_name_upper}_COARSE],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE]);"
    )

    hpp.close()


def generate_elemwise_files(name, config, partition_name):
    if not os.path.exists(os.path.join(os.getcwd(), "generated_files", partition_name)):
        os.makedirs(os.path.join(os.getcwd(), "generated_files", partition_name))

    generate_elemwise_hpp(name, config, partition_name)
    generate_elemwise_cpp(name, config, partition_name)
