import os

from .codegen import *


def generate_pool_cpp(name: str, config: dict, model_name: str, partition_name: str):
    batch_size = config["batch_size"]
    channels = config["channels_in"]
    depth = config["depth_in"]
    height = config["height_in"]
    width = config["width_in"]
    filters = config["channels_out"]
    depth_out = config["depth_out"]
    height_out = config["height_out"]
    width_out = config["width_out"]
    kd = config["kernel_depth"]
    kh = config["kernel_height"]
    kw = config["kernel_width"]
    pad_d = config["pad_depth"]
    pad_h = config["pad_height"]
    pad_w = config["pad_width"]
    stride_d = config["stride_depth"]
    stride_h = config["stride_height"]
    stride_w = config["stride_width"]
    op_type = config["op_type"]

    fine_factor = config["fine_factor"]
    coarse_factor = config["coarse_factor"]

    spatial = True if kd == 1 and kh > 1 and kw > 1 else False
    temporal = True if kd > 1 and kh == 1 and kw == 1 else False
    if spatial:
        sw_func_call = "sliding_window_3d_spatial"
    elif temporal:
        sw_func_call = "sliding_window_3d_temporal"
    else:
        sw_func_call = "sliding_window_3d"

    layer_name_lower = name.lower()
    layer_name_upper = name.upper()

    cpp = CppFile(
        os.path.join(
            os.getcwd(), "generated_files", model_name, partition_name, name, "src", f"{layer_name_lower}.cpp"
        )
    )

    cpp(f'#include "{layer_name_lower}.hpp"', newlines=2)

    with cpp.block(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE_IN],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE_IN])"
    ):

        cpp("#pragma HLS INLINE OFF")
        cpp("#pragma HLS DATAFLOW", newlines=2)

        cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

        cpp(    f"stream_t({layer_name_lower}_data_t) sw_out[{layer_name_upper}_COARSE_IN][{layer_name_upper}_KERNEL_SIZE_HEIGHT][{layer_name_upper}_KERNEL_SIZE_WIDTH][{layer_name_upper}_KERNEL_SIZE_DEPTH];"
            )
        cpp("#pragma HLS STREAM variable=sw_out")
        cpp(
            "#pragma HLS ARRAY_PARTITION variable=sw_out complete dim=0", newlines=2
        )

        with cpp.block(f"for(int coarseIndex=0; coarseIndex<{layer_name_upper}_COARSE_IN; coarseIndex++)"):
            cpp("#pragma HLS unroll", newlines=2)

            cpp(
                f"{sw_func_call}<\n\
                {layer_name_upper}_SW_BATCH_SIZE,\n\
                {layer_name_upper}_SW_CHANNELS,\n\
                {layer_name_upper}_SW_HEIGHT,\n\
                {layer_name_upper}_SW_WIDTH,\n\
                {layer_name_upper}_SW_DEPTH,\n\
                {layer_name_upper}_SW_KERNEL_SIZE_HEIGHT,\n\
                {layer_name_upper}_SW_KERNEL_SIZE_WIDTH,\n\
                {layer_name_upper}_SW_KERNEL_SIZE_DEPTH,\n\
                {layer_name_upper}_SW_PAD_HEIGHT,\n\
                {layer_name_upper}_SW_PAD_WIDTH,\n\
                {layer_name_upper}_SW_PAD_DEPTH,\n\
                {layer_name_upper}_SW_STRIDE_HEIGHT,\n\
                {layer_name_upper}_SW_STRIDE_WIDTH,\n\
                {layer_name_upper}_SW_STRIDE_DEPTH,\n\
                {layer_name_upper}_SW_PAD_VALUE,\n\
                {layer_name_lower}_data_t\n\
            >(in[coarseIndex],sw_out[coarseIndex]);",
                newlines=2,
            )

            cpp(
                f"pool_3d<\n\
                {layer_name_upper}_POOL_BATCH_SIZE,\n\
                {layer_name_upper}_POOL_CHANNELS,\n\
                {layer_name_upper}_POOL_HEIGHT,\n\
                {layer_name_upper}_POOL_WIDTH,\n\
                {layer_name_upper}_POOL_DEPTH,\n\
                {layer_name_upper}_POOL_KERNEL_SIZE_HEIGHT,\n\
                {layer_name_upper}_POOL_KERNEL_SIZE_WIDTH,\n\
                {layer_name_upper}_POOL_KERNEL_SIZE_DEPTH,\n\
                {layer_name_lower}_data_t\n\
            >(sw_out[coarseIndex],out[coarseIndex]);",
                newlines=2,
            )

    cpp.close()


def generate_pool_hpp(name: str, config: dict, model_name: str, partition_name: str):
    batch_size = config["batch_size"]
    channels = config["channels_in"]
    depth = config["depth_in"]
    height = config["height_in"]
    width = config["width_in"]
    filters = config["channels_out"]
    depth_out = config["depth_out"]
    height_out = config["height_out"]
    width_out = config["width_out"]
    kd = config["kernel_depth"]
    kh = config["kernel_height"]
    kw = config["kernel_width"]
    pad_d = config["pad_depth"]
    pad_h = config["pad_height"]
    pad_w = config["pad_width"]
    stride_d = config["stride_depth"]
    stride_h = config["stride_height"]
    stride_w = config["stride_width"]
    op_type = config["op_type"]

    fine_factor = config["fine_factor"]
    coarse_factor = config["coarse_factor"]

    layer_name_lower = name.lower()
    layer_name_upper = name.upper()

    hpp = CppFile(
        os.path.join(
            os.getcwd(), "generated_files", model_name, partition_name, name, "src", f"{layer_name_lower}.hpp"
        )
    )

    hpp("#pragma once", newlines=2)
    hpp('#include "common_.hpp"')
    hpp('#include "sliding_window_3d_.hpp"')
    hpp('#include "pool_3d_.hpp"', newlines=2)

    hpp(f"#define {layer_name_upper}_BATCH_SIZE {batch_size}")
    hpp(f"#define {layer_name_upper}_CHANNELS_IN {channels}")
    hpp(f"#define {layer_name_upper}_DEPTH_IN {depth}")
    hpp(f"#define {layer_name_upper}_HEIGHT_IN {height}")
    hpp(f"#define {layer_name_upper}_WIDTH_IN {width}")
    hpp(f"#define {layer_name_upper}_KERNEL_SIZE_DEPTH {kd}")
    hpp(f"#define {layer_name_upper}_KERNEL_SIZE_HEIGHT {kh}")
    hpp(f"#define {layer_name_upper}_KERNEL_SIZE_WIDTH {kw}")
    hpp(f"#define {layer_name_upper}_PAD_DEPTH {pad_d}")
    hpp(f"#define {layer_name_upper}_PAD_HEIGHT {pad_h}")
    hpp(f"#define {layer_name_upper}_PAD_WIDTH {pad_w}")
    hpp(f"#define {layer_name_upper}_STRIDE_DEPTH {stride_d}")
    hpp(f"#define {layer_name_upper}_STRIDE_HEIGHT {stride_h}")
    hpp(f"#define {layer_name_upper}_STRIDE_WIDTH {stride_w}")

    hpp(f"#define {layer_name_upper}_FINE {fine_factor}")
    hpp(f"#define {layer_name_upper}_COARSE_IN {coarse_factor}")
    hpp(f"#define {layer_name_upper}_COARSE_OUT {coarse_factor}")

    hpp(f"#define {layer_name_upper}_CHANNELS_OUT {channels}")
    hpp(f"#define {layer_name_upper}_DEPTH_OUT {depth_out}")
    hpp(f"#define {layer_name_upper}_HEIGHT_OUT {height_out}")
    hpp(f"#define {layer_name_upper}_WIDTH_OUT {width_out}", newlines=2)

    if "max" in op_type:
        hpp(f"#define {layer_name_upper}_SW_PAD_VALUE -100000")
    else:
        hpp(f"#define {layer_name_upper}_SW_PAD_VALUE 0")
    hpp(f"#define {layer_name_upper}_SW_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE")
    hpp(
        f"#define {layer_name_upper}_SW_CHANNELS \tDIVIDE({layer_name_upper}_CHANNELS_IN, {layer_name_upper}_COARSE_IN)"
    )
    hpp(f"#define {layer_name_upper}_SW_DEPTH \t{layer_name_upper}_DEPTH_IN")
    hpp(f"#define {layer_name_upper}_SW_HEIGHT \t{layer_name_upper}_HEIGHT_IN")
    hpp(f"#define {layer_name_upper}_SW_WIDTH \t{layer_name_upper}_WIDTH_IN")
    hpp(
        f"#define {layer_name_upper}_SW_KERNEL_SIZE_DEPTH \t{layer_name_upper}_KERNEL_SIZE_DEPTH"
    )
    hpp(
        f"#define {layer_name_upper}_SW_KERNEL_SIZE_HEIGHT \t{layer_name_upper}_KERNEL_SIZE_HEIGHT"
    )
    hpp(
        f"#define {layer_name_upper}_SW_KERNEL_SIZE_WIDTH \t{layer_name_upper}_KERNEL_SIZE_WIDTH"
    )
    hpp(f"#define {layer_name_upper}_SW_PAD_DEPTH \t{layer_name_upper}_PAD_DEPTH")
    hpp(f"#define {layer_name_upper}_SW_PAD_HEIGHT \t{layer_name_upper}_PAD_HEIGHT")
    hpp(f"#define {layer_name_upper}_SW_PAD_WIDTH \t{layer_name_upper}_PAD_WIDTH")
    hpp(f"#define {layer_name_upper}_SW_STRIDE_DEPTH \t{layer_name_upper}_STRIDE_DEPTH")
    hpp(
        f"#define {layer_name_upper}_SW_STRIDE_HEIGHT \t{layer_name_upper}_STRIDE_HEIGHT"
    )
    hpp(
        f"#define {layer_name_upper}_SW_STRIDE_WIDTH \t{layer_name_upper}_STRIDE_WIDTH",
        newlines=2,
    )

    hpp(f"#define {layer_name_upper}_POOL_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE")
    hpp(f"#define {layer_name_upper}_POOL_CHANNELS \tDIVIDE({layer_name_upper}_CHANNELS_IN, {layer_name_upper}_COARSE_IN)")
    hpp(f"#define {layer_name_upper}_POOL_DEPTH \t{layer_name_upper}_DEPTH_OUT")
    hpp(f"#define {layer_name_upper}_POOL_HEIGHT \t{layer_name_upper}_HEIGHT_OUT")
    hpp(f"#define {layer_name_upper}_POOL_WIDTH \t{layer_name_upper}_WIDTH_OUT")
    hpp(f"#define {layer_name_upper}_POOL_KERNEL_SIZE_DEPTH \t{layer_name_upper}_KERNEL_SIZE_DEPTH")
    hpp(f"#define {layer_name_upper}_POOL_KERNEL_SIZE_HEIGHT \t{layer_name_upper}_KERNEL_SIZE_HEIGHT")
    hpp(f"#define {layer_name_upper}_POOL_KERNEL_SIZE_WIDTH \t{layer_name_upper}_KERNEL_SIZE_WIDTH", newlines=2)

    hpp(
        f"typedef ap_fixed<16,8,AP_RND, AP_SAT> \t{layer_name_lower}_data_t;",
        newlines=2,
    )

    hpp(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE_IN],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE_IN]);"
    )

    hpp.close()


def generate_pool_files(name: str, config: dict, model_name: str, partition_name: str = ''):

    if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, name, "src")):
        os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, name, "src"))

    generate_pool_hpp(name, config, model_name, partition_name)
    generate_pool_cpp(name, config, model_name, partition_name)
