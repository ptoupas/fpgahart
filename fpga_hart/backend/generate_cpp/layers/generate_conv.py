import os

from .codegen import *


def generate_conv_cpp(name, config, partition_name):
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
    groups = config["groups"]
    depthwise = config["depthwise"]
    pointwise = config["pointwise"]

    fine_factor = config["fine_factor"]
    coarse_in_factor = config["coarse_in_factor"]
    coarse_out_factor = config["coarse_out_factor"]

    layer_name_lower = name.lower()
    layer_name_upper = name.upper()

    inner_block_loop = (
        f"{layer_name_upper}_COARSE_OUT"
        if not depthwise
        else f"{layer_name_upper}_COARSE_OUT_INNER"
    )

    cpp = CppFile(
        os.path.join(
            os.getcwd(), "generated_files", partition_name, f"{layer_name_lower}.cpp"
        )
    )

    cpp(f'#include "{layer_name_lower}.hpp"', newlines=2)

    with cpp.block(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE_IN],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE_OUT])"
    ):

        cpp("#pragma HLS INLINE OFF")
        cpp("#pragma HLS DATAFLOW", newlines=2)

        cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

        cpp(
            f"#pragma HLS ARRAY_PARTITION variable=weights_{layer_name_lower} complete dim=1"
        )
        cpp(
            f"#pragma HLS ARRAY_PARTITION variable=weights_{layer_name_lower} complete dim=2"
        )
        # cpp(f"#pragma HLS BIND_STORAGE variable=weights_{layer_name_lower} type=ram_2p")
        cpp(f"#pragma HLS STABLE variable=weights_{layer_name_lower}", newlines=2)

        if not pointwise:
            cpp(
                f"stream_t({layer_name_lower}_data_t) sw_out[{layer_name_upper}_COARSE_IN][{layer_name_upper}_KERNEL_SIZE_HEIGHT][{layer_name_upper}_KERNEL_SIZE_WIDTH][{layer_name_upper}_KERNEL_SIZE_DEPTH];"
            )
            cpp("#pragma HLS STREAM variable=sw_out")
            cpp(
                "#pragma HLS ARRAY_PARTITION variable=sw_out complete dim=0", newlines=2
            )

            if not depthwise:
                cpp(
                    f"stream_t({layer_name_lower}_data_t) fork_out[{layer_name_upper}_COARSE_IN][{layer_name_upper}_COARSE_OUT][{layer_name_upper}_KERNEL_SIZE_HEIGHT][{layer_name_upper}_KERNEL_SIZE_WIDTH][{layer_name_upper}_KERNEL_SIZE_DEPTH];"
                )
                cpp("#pragma HLS STREAM variable=fork_out")
                cpp(
                    "#pragma HLS ARRAY_PARTITION variable=fork_out complete dim=0",
                    newlines=2,
                )
        else:
            if not depthwise:
                cpp(
                    f"stream_t({layer_name_lower}_data_t) fork_out[{layer_name_upper}_COARSE_IN][{layer_name_upper}_COARSE_OUT];"
                )
                cpp("#pragma HLS STREAM variable=fork_out")
                cpp(
                    "#pragma HLS ARRAY_PARTITION variable=fork_out complete dim=0",
                    newlines=2,
                )

        if depthwise:
            cpp(
                f"stream_t(accum_data_t) conv_out[{layer_name_upper}_COARSE_IN][{layer_name_upper}_COARSE_OUT_INNER];"
            )
            cpp("#pragma HLS STREAM variable=conv_out")
            cpp(
                "#pragma HLS ARRAY_PARTITION variable=conv_out complete dim=0",
                newlines=2,
            )
        else:
            cpp(
                f"stream_t(accum_data_t) conv_out[{layer_name_upper}_COARSE_IN][{layer_name_upper}_COARSE_OUT];"
            )
            cpp("#pragma HLS STREAM variable=conv_out")
            cpp(
                "#pragma HLS ARRAY_PARTITION variable=conv_out complete dim=0",
                newlines=2,
            )

        if not depthwise:
            cpp(
                f"stream_t(accum_data_t) accum_out[{layer_name_upper}_COARSE_IN][{layer_name_upper}_COARSE_OUT];"
            )
            cpp("#pragma HLS STREAM variable=accum_out")
            cpp(
                "#pragma HLS ARRAY_PARTITION variable=accum_out complete dim=0",
                newlines=2,
            )

        with cpp.block(f"for(int i=0; i<{layer_name_upper}_COARSE_IN; i++)"):
            cpp("#pragma HLS unroll", newlines=2)

            if not pointwise:
                if not depthwise:
                    cpp(
                        f"sliding_window_3d<\n\
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
                        {layer_name_lower}_data_t\n\
                    >(in[i],sw_out[i]);",
                        newlines=2,
                    )

                    cpp(
                        f"fork_3d<\n\
                        {layer_name_upper}_FORK_BATCH_SIZE,\n\
                        {layer_name_upper}_FORK_CHANNELS,\n\
                        {layer_name_upper}_FORK_HEIGHT,\n\
                        {layer_name_upper}_FORK_WIDTH,\n\
                        {layer_name_upper}_FORK_DEPTH,\n\
                        {layer_name_upper}_FORK_COARSE,\n\
                        {layer_name_upper}_FORK_KERNEL_SIZE_HEIGHT,\n\
                        {layer_name_upper}_FORK_KERNEL_SIZE_WIDTH,\n\
                        {layer_name_upper}_FORK_KERNEL_SIZE_DEPTH,\n\
                        {layer_name_lower}_data_t\n\
                    >(sw_out[i],fork_out[i]);",
                        newlines=2,
                    )
                else:
                    cpp(
                        f"sliding_window_3d<\n\
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
                        {layer_name_lower}_data_t\n\
                    >(in[i],sw_out[i]);",
                        newlines=2,
                    )
            else:
                cpp(
                    f"fork_3d<\n\
                    {layer_name_upper}_FORK_BATCH_SIZE,\n\
                    {layer_name_upper}_FORK_CHANNELS,\n\
                    {layer_name_upper}_FORK_HEIGHT,\n\
                    {layer_name_upper}_FORK_WIDTH,\n\
                    {layer_name_upper}_FORK_DEPTH,\n\
                    {layer_name_upper}_FORK_STRIDE_HEIGHT,\n\
                    {layer_name_upper}_FORK_STRIDE_WIDTH,\n\
                    {layer_name_upper}_FORK_STRIDE_DEPTH,\n\
                    {layer_name_upper}_FORK_COARSE,\n\
                    {layer_name_lower}_data_t\n\
                >(in[i],fork_out[i]);",
                    newlines=2,
                )

            if not depthwise:
                with cpp.block(f"for(int j=0; j<{inner_block_loop}; j++)"):

                    cpp("#pragma HLS unroll", newlines=2)

                    if pointwise and not depthwise:
                        cpp(
                            f"conv_3d<\n\
                        {layer_name_upper}_CONV_BATCH_SIZE,\n\
                        {layer_name_upper}_CONV_CHANNELS,\n\
                        {layer_name_upper}_CONV_FILTERS,\n\
                        {layer_name_upper}_CONV_HEIGHT,\n\
                        {layer_name_upper}_CONV_WIDTH,\n\
                        {layer_name_upper}_CONV_DEPTH,\n\
                        {layer_name_upper}_CONV_GROUPS,\n\
                        {layer_name_lower}_data_t,\n\
                        {layer_name_lower}_data_t,\n\
                        accum_data_t\n\
                    >(fork_out[i][j],weights_{layer_name_lower}[i][j],conv_out[i][j]);",
                            newlines=2,
                        )
                    else:
                        cpp(
                            f"conv_3d<\n\
                        {layer_name_upper}_CONV_BATCH_SIZE,\n\
                        {layer_name_upper}_CONV_CHANNELS,\n\
                        {layer_name_upper}_CONV_FILTERS,\n\
                        {layer_name_upper}_CONV_HEIGHT,\n\
                        {layer_name_upper}_CONV_WIDTH,\n\
                        {layer_name_upper}_CONV_DEPTH,\n\
                        {layer_name_upper}_CONV_KERNEL_SIZE_HEIGHT,\n\
                        {layer_name_upper}_CONV_KERNEL_SIZE_WIDTH,\n\
                        {layer_name_upper}_CONV_KERNEL_SIZE_DEPTH,\n\
                        {layer_name_upper}_CONV_FINE,\n\
                        {layer_name_upper}_CONV_GROUPS,\n\
                        {layer_name_lower}_data_t,\n\
                        {layer_name_lower}_data_t,\n\
                        accum_data_t\n\
                    >(fork_out[i][j],weights_{layer_name_lower}[i][j],conv_out[i][j]);",
                            newlines=2,
                        )

                    if not depthwise:
                        cpp(
                            f"accum_3d<\n\
                        {layer_name_upper}_ACCUM_BATCH_SIZE,\n\
                        {layer_name_upper}_ACCUM_CHANNELS,\n\
                        {layer_name_upper}_ACCUM_FILTERS,\n\
                        {layer_name_upper}_ACCUM_HEIGHT,\n\
                        {layer_name_upper}_ACCUM_WIDTH,\n\
                        {layer_name_upper}_ACCUM_DEPTH,\n\
                        {layer_name_upper}_ACCUM_GROUPS,\n\
                        accum_data_t\n\
                    >(conv_out[i][j],accum_out[i][j]);",
                            newlines=2,
                        )
            else:
                cpp(
                    f"conv_3d<\n\
                {layer_name_upper}_CONV_BATCH_SIZE,\n\
                {layer_name_upper}_CONV_CHANNELS,\n\
                {layer_name_upper}_CONV_FILTERS,\n\
                {layer_name_upper}_CONV_HEIGHT,\n\
                {layer_name_upper}_CONV_WIDTH,\n\
                {layer_name_upper}_CONV_DEPTH,\n\
                {layer_name_upper}_CONV_KERNEL_SIZE_HEIGHT,\n\
                {layer_name_upper}_CONV_KERNEL_SIZE_WIDTH,\n\
                {layer_name_upper}_CONV_KERNEL_SIZE_DEPTH,\n\
                {layer_name_upper}_CONV_FINE,\n\
                {layer_name_upper}_CONV_GROUPS,\n\
                {layer_name_lower}_data_t,\n\
                {layer_name_lower}_data_t,\n\
                accum_data_t\n\
            >(sw_out[i],weights_{layer_name_lower}[i][0],conv_out[i][0]);",
                    newlines=2,
                )
        if not depthwise:
            cpp(
                f"glue_3d<\n\
                {layer_name_upper}_GLUE_BATCH_SIZE,\n\
                {layer_name_upper}_GLUE_FILTERS,\n\
                {layer_name_upper}_GLUE_HEIGHT,\n\
                {layer_name_upper}_GLUE_WIDTH,\n\
                {layer_name_upper}_GLUE_DEPTH,\n\
                {layer_name_upper}_GLUE_COARSE_IN,\n\
                {layer_name_upper}_GLUE_COARSE_OUT,\n\
                accum_data_t,\n\
                {layer_name_lower}_data_t\n\
            >(accum_out,out);",
                newlines=2,
            )
        else:
            cpp(
                f"glue_dw_3d<\n\
                {layer_name_upper}_GLUE_BATCH_SIZE,\n\
                {layer_name_upper}_GLUE_CHANNELS,\n\
                {layer_name_upper}_GLUE_FILTERS,\n\
                {layer_name_upper}_GLUE_HEIGHT,\n\
                {layer_name_upper}_GLUE_WIDTH,\n\
                {layer_name_upper}_GLUE_DEPTH,\n\
                {layer_name_upper}_GLUE_GROUPS,\n\
                {layer_name_upper}_GLUE_COARSE_IN,\n\
                {layer_name_upper}_GLUE_COARSE_OUT,\n\
                accum_data_t,\n\
                {layer_name_lower}_data_t\n\
            >(conv_out,out);",
                newlines=2,
            )
    cpp.close()


def generate_conv_hpp(name, config, partition_name, hls_project_path):
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
    groups = config["groups"]
    depthwise = config["depthwise"]
    pointwise = config["pointwise"]

    fine_factor = config["fine_factor"]
    coarse_in_factor = config["coarse_in_factor"]
    coarse_out_factor = config["coarse_out_factor"]

    layer_name_lower = name.lower()
    layer_name_upper = name.upper()

    weights_file_path = f"{hls_project_path}/{partition_name}/data/weights_{layer_name_lower}_cin{coarse_in_factor}_cout{coarse_out_factor}.csv"

    hpp = CppFile(
        os.path.join(
            os.getcwd(), "generated_files", partition_name, f"{layer_name_lower}.hpp"
        )
    )

    hpp("#pragma once", newlines=2)
    hpp('#include "common_.hpp"')
    hpp('#include "sliding_window_3d_.hpp"')
    hpp('#include "fork_3d_.hpp"')
    hpp('#include "conv_3d_.hpp"')
    hpp('#include "accum_3d_.hpp"')
    hpp('#include "glue_3d_.hpp"', newlines=2)

    hpp(f"#define {layer_name_upper}_DEPTHWISE {depthwise}")
    hpp(f"#define {layer_name_upper}_POINTWISE {pointwise}", newlines=2)

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
    hpp(f"#define {layer_name_upper}_GROUPS {groups}", newlines=2)

    hpp(f"#define {layer_name_upper}_FINE {fine_factor}")
    hpp(f"#define {layer_name_upper}_COARSE_IN {coarse_in_factor}")
    if depthwise:
        hpp(f"#define {layer_name_upper}_COARSE_OUT {coarse_out_factor}")
        hpp(f"#define {layer_name_upper}_COARSE_OUT_INNER 1", newlines=2)
    else:
        hpp(f"#define {layer_name_upper}_COARSE_OUT {coarse_out_factor}", newlines=2)

    hpp(f"#define {layer_name_upper}_FILTERS {filters}")
    hpp(f"#define {layer_name_upper}_CHANNELS_OUT {filters}")
    hpp(f"#define {layer_name_upper}_DEPTH_OUT {depth_out}")
    hpp(f"#define {layer_name_upper}_HEIGHT_OUT {height_out}")
    hpp(f"#define {layer_name_upper}_WIDTH_OUT {width_out}", newlines=2)

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

    hpp(f"#define {layer_name_upper}_FORK_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE")
    hpp(
        f"#define {layer_name_upper}_FORK_CHANNELS \tDIVIDE({layer_name_upper}_CHANNELS_IN, {layer_name_upper}_COARSE_IN)"
    )
    if not pointwise:
        hpp(f"#define {layer_name_upper}_FORK_DEPTH \t{layer_name_upper}_DEPTH_OUT")
        hpp(f"#define {layer_name_upper}_FORK_HEIGHT \t{layer_name_upper}_HEIGHT_OUT")
        hpp(f"#define {layer_name_upper}_FORK_WIDTH \t{layer_name_upper}_WIDTH_OUT")
    else:
        hpp(f"#define {layer_name_upper}_FORK_DEPTH \t{layer_name_upper}_DEPTH_IN")
        hpp(f"#define {layer_name_upper}_FORK_HEIGHT \t{layer_name_upper}_HEIGHT_IN")
        hpp(f"#define {layer_name_upper}_FORK_WIDTH \t{layer_name_upper}_WIDTH_IN")
        hpp(
            f"#define {layer_name_upper}_FORK_STRIDE_DEPTH \t{layer_name_upper}_STRIDE_DEPTH"
        )
        hpp(
            f"#define {layer_name_upper}_FORK_STRIDE_HEIGHT \t{layer_name_upper}_STRIDE_HEIGHT"
        )
        hpp(
            f"#define {layer_name_upper}_FORK_STRIDE_WIDTH \t{layer_name_upper}_STRIDE_WIDTH"
        )
    hpp(
        f"#define {layer_name_upper}_FORK_KERNEL_SIZE_DEPTH \t{layer_name_upper}_KERNEL_SIZE_DEPTH"
    )
    hpp(
        f"#define {layer_name_upper}_FORK_KERNEL_SIZE_HEIGHT \t{layer_name_upper}_KERNEL_SIZE_HEIGHT"
    )
    hpp(
        f"#define {layer_name_upper}_FORK_KERNEL_SIZE_WIDTH \t{layer_name_upper}_KERNEL_SIZE_WIDTH"
    )
    if depthwise:
        hpp(
            f"#define {layer_name_upper}_FORK_COARSE \t{layer_name_upper}_COARSE_OUT_INNER",
            newlines=2,
        )
    else:
        hpp(
            f"#define {layer_name_upper}_FORK_COARSE \t{layer_name_upper}_COARSE_OUT",
            newlines=2,
        )

    hpp(f"#define {layer_name_upper}_CONV_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE")
    hpp(
        f"#define {layer_name_upper}_CONV_CHANNELS \tDIVIDE({layer_name_upper}_CHANNELS_IN, {layer_name_upper}_COARSE_IN)"
    )
    if depthwise:
        hpp(
            f"#define {layer_name_upper}_CONV_FILTERS \tDIVIDE({layer_name_upper}_FILTERS, {layer_name_upper}_COARSE_OUT_INNER)"
        )
    else:
        hpp(
            f"#define {layer_name_upper}_CONV_FILTERS \tDIVIDE({layer_name_upper}_FILTERS, {layer_name_upper}_COARSE_OUT)"
        )
    hpp(f"#define {layer_name_upper}_CONV_DEPTH \t{layer_name_upper}_DEPTH_OUT")
    hpp(f"#define {layer_name_upper}_CONV_HEIGHT \t{layer_name_upper}_HEIGHT_OUT")
    hpp(f"#define {layer_name_upper}_CONV_WIDTH \t{layer_name_upper}_WIDTH_OUT")
    hpp(
        f"#define {layer_name_upper}_CONV_KERNEL_SIZE_DEPTH \t{layer_name_upper}_KERNEL_SIZE_DEPTH"
    )
    hpp(
        f"#define {layer_name_upper}_CONV_KERNEL_SIZE_HEIGHT \t{layer_name_upper}_KERNEL_SIZE_HEIGHT"
    )
    hpp(
        f"#define {layer_name_upper}_CONV_KERNEL_SIZE_WIDTH \t{layer_name_upper}_KERNEL_SIZE_WIDTH"
    )
    hpp(f"#define {layer_name_upper}_CONV_GROUPS \t{layer_name_upper}_GROUPS")
    hpp(f"#define {layer_name_upper}_CONV_FINE \t{layer_name_upper}_FINE", newlines=2)

    hpp(f"#define {layer_name_upper}_ACCUM_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE")
    hpp(
        f"#define {layer_name_upper}_ACCUM_CHANNELS \tDIVIDE({layer_name_upper}_CHANNELS_IN, {layer_name_upper}_COARSE_IN)"
    )
    if depthwise:
        hpp(
            f"#define {layer_name_upper}_ACCUM_FILTERS \tDIVIDE({layer_name_upper}_FILTERS, {layer_name_upper}_COARSE_OUT_INNER)"
        )
    else:
        hpp(
            f"#define {layer_name_upper}_ACCUM_FILTERS \tDIVIDE({layer_name_upper}_FILTERS, {layer_name_upper}_COARSE_OUT)"
        )
    hpp(f"#define {layer_name_upper}_ACCUM_DEPTH \t{layer_name_upper}_DEPTH_OUT")
    hpp(f"#define {layer_name_upper}_ACCUM_HEIGHT \t{layer_name_upper}_HEIGHT_OUT")
    hpp(f"#define {layer_name_upper}_ACCUM_WIDTH \t{layer_name_upper}_WIDTH_OUT")
    hpp(
        f"#define {layer_name_upper}_ACCUM_GROUPS \t{layer_name_upper}_GROUPS",
        newlines=2,
    )

    hpp(f"#define {layer_name_upper}_GLUE_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE")
    hpp(
        f"#define {layer_name_upper}_GLUE_CHANNELS \tDIVIDE({layer_name_upper}_CHANNELS_IN, {layer_name_upper}_COARSE_IN)"
    )
    hpp(f"#define {layer_name_upper}_GLUE_FILTERS \t{layer_name_upper}_FILTERS")
    hpp(f"#define {layer_name_upper}_GLUE_DEPTH \t{layer_name_upper}_DEPTH_OUT")
    hpp(f"#define {layer_name_upper}_GLUE_HEIGHT \t{layer_name_upper}_HEIGHT_OUT")
    hpp(f"#define {layer_name_upper}_GLUE_WIDTH \t{layer_name_upper}_WIDTH_OUT")
    hpp(f"#define {layer_name_upper}_GLUE_GROUPS \t{layer_name_upper}_GROUPS")
    hpp(f"#define {layer_name_upper}_GLUE_COARSE_IN \t{layer_name_upper}_COARSE_IN")
    if depthwise:
        hpp(
            f"#define {layer_name_upper}_GLUE_COARSE_OUT \t{layer_name_upper}_COARSE_OUT_INNER",
            newlines=2,
        )
    else:
        hpp(
            f"#define {layer_name_upper}_GLUE_COARSE_OUT \t{layer_name_upper}_COARSE_OUT",
            newlines=2,
        )

    hpp(
        f"typedef ap_fixed<16,8,AP_RND, AP_SAT> \t{layer_name_lower}_data_t;",
        newlines=2,
    )

    if depthwise:
        hpp(
            f'static {layer_name_lower}_data_t weights_{layer_name_lower} [{layer_name_upper}_COARSE_IN]\n\
                                        [{layer_name_upper}_COARSE_OUT_INNER]\n\
                                        [DIVIDE({layer_name_upper}_CHANNELS_IN, {layer_name_upper}_COARSE_IN)]\n\
                                        [DIVIDE({layer_name_upper}_FILTERS, {layer_name_upper}_COARSE_OUT_INNER*{layer_name_upper}_GROUPS)]\n\
                                        [{layer_name_upper}_KERNEL_SIZE_HEIGHT]\n\
                                        [{layer_name_upper}_KERNEL_SIZE_WIDTH]\n\
                                        [{layer_name_upper}_KERNEL_SIZE_DEPTH] = {{\n\
                                        #include "{weights_file_path}"\n}};',
            newlines=3,
        )
    else:
        hpp(
            f'static {layer_name_lower}_data_t weights_{layer_name_lower} [{layer_name_upper}_COARSE_IN]\n\
                                            [{layer_name_upper}_COARSE_OUT]\n\
                                            [DIVIDE({layer_name_upper}_CHANNELS_IN, {layer_name_upper}_COARSE_IN)]\n\
                                            [DIVIDE({layer_name_upper}_FILTERS, {layer_name_upper}_COARSE_OUT*{layer_name_upper}_GROUPS)]\n\
                                            [{layer_name_upper}_KERNEL_SIZE_HEIGHT]\n\
                                            [{layer_name_upper}_KERNEL_SIZE_WIDTH]\n\
                                            [{layer_name_upper}_KERNEL_SIZE_DEPTH] = {{\n\
                                            #include "{weights_file_path}"\n}};',
            newlines=3,
        )

    hpp(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE_IN],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE_OUT]);"
    )

    hpp.close()


def generate_conv_files(name, config, partition_name, hls_project_path):
    if not os.path.exists(os.path.join(os.getcwd(), "generated_files", partition_name)):
        os.makedirs(os.path.join(os.getcwd(), "generated_files", partition_name))

    generate_conv_hpp(name, config, partition_name, hls_project_path)
    generate_conv_cpp(name, config, partition_name)
