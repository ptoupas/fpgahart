import os

from layers.codegen import *


def generate_top_level_layer_cpp(layer_name: str, model_name: str, partition_name: str, dynamic_reconfig: bool):
    layer_name_lower = layer_name.lower()
    layer_name_upper = layer_name.upper()

    if dynamic_reconfig:
        cpp = CppFile(
            os.path.join(
                os.getcwd(),
                "generated_files",
                model_name,
                partition_name,
                "latency_driven",
                layer_name,
                "src",
                f"{layer_name_lower}_top.cpp"
            )
        )
    else:
        cpp = CppFile(
            os.path.join(
                os.getcwd(),
                "generated_files",
                model_name,
                partition_name,
                layer_name,
                "src",
                f"{layer_name_lower}_top.cpp"
            )
        )

    cpp(f'#include "{layer_name_lower}_top.hpp"', newlines=2)

    with cpp.block(
        "template <int PIXEL_LOOP, int COARSE, typename T, typename T_AXIS>\n\
         void axis_to_stream(\n\
                stream_t(T_AXIS) in[COARSE],\n\
                stream_t(T) out[COARSE])"
    ):
        cpp("#pragma HLS INLINE OFF", newlines=2)

        cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

        with cpp.block("for(int pixelIndex=0; pixelIndex<PIXEL_LOOP; pixelIndex++)"):
            cpp("#pragma HLS PIPELINE II=1")
            with cpp.block("for(int coarseIndex=0; coarseIndex<COARSE; coarseIndex++)"):
                cpp("T tmp;")
                cpp("tmp.range() = in[coarseIndex].read().data;")
                cpp("out[coarseIndex].write(tmp);")
    if "Gemm" in layer_name:
        with cpp.block(
            "template <int PIXEL_LOOP, int COARSE_IN, int COARSE_OUT, typename T, typename T_AXIS>\n\
            void axis_to_stream(\n\
                    stream_t(T_AXIS) in[COARSE_IN][COARSE_OUT],\n\
                    stream_t(T) out[COARSE_IN][COARSE_OUT])"
        ):
            cpp("#pragma HLS INLINE OFF", newlines=2)

            cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

            with cpp.block("for(int coarseInIndex=0; coarseInIndex<COARSE_IN; coarseInIndex++)"):
                with cpp.block("for(int coarseOutIndex=0; coarseOutIndex<COARSE_OUT; coarseOutIndex++)"):
                    cpp("#pragma HLS PIPELINE II=1")
                    with cpp.block("for(int pixelIndex=0; pixelIndex<PIXEL_LOOP; pixelIndex++)"):
                        cpp("T tmp;")
                        cpp("tmp.range() = in[coarseInIndex][coarseOutIndex].read().data;")
                        cpp("out[coarseInIndex][coarseOutIndex].write(tmp);")
    with cpp.block(
        "template <int PIXEL_LOOP, int COARSE, typename T, typename T_AXIS>\n\
         void stream_to_axis(\n\
                stream_t(T) in[COARSE],\n\
                stream_t(T_AXIS) out[COARSE])"
    ):
        cpp("#pragma HLS INLINE OFF", newlines=2)

        cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

        with cpp.block("for(int pixelIndex=0; pixelIndex<PIXEL_LOOP; pixelIndex++)"):
            cpp("#pragma HLS PIPELINE II=1")
            with cpp.block("for(int coarseIndex=0; coarseIndex<COARSE; coarseIndex++)"):
                cpp("T_AXIS tmp;")
                cpp("tmp.data = in[coarseIndex].read().range();")
                cpp("tmp.keep = -1;")
                cpp("tmp.user = (pixelIndex == 0);")
                cpp("tmp.last = (pixelIndex == PIXEL_LOOP-1);")
                cpp("out[coarseIndex].write(tmp);")

    if "Sigmoid" in layer_name or "Swish" in layer_name or "Relu" in layer_name or "Gap" in layer_name or "Activation" in layer_name:
        with cpp.block(
            f"void {layer_name_lower}_top(\n\
            stream_t(axi_stream_t) in[{layer_name_upper}_COARSE],\n\
            stream_t(axi_stream_t) out[{layer_name_upper}_COARSE])"
        ):
            cpp("#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=in register")
            cpp("#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=out register")
            cpp("#pragma HLS INTERFACE mode=s_axilite bundle=control port=return", newlines=2)

            cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

            cpp("#pragma HLS DATAFLOW", newlines=2)

            cpp(f"stream_t({layer_name_lower}_data_t) in_stream[{layer_name_upper}_COARSE];")
            cpp(f"stream_t({layer_name_lower}_data_t) out_stream[{layer_name_upper}_COARSE];", newlines=2)

            cpp("#pragma HLS ARRAY_PARTITION variable=in_stream  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=out_stream complete dim=0", newlines=2)

            cpp(f"const int pixel_loop_in = {layer_name_upper}_BATCH_SIZE*DIVIDE({layer_name_upper}_CHANNELS, {layer_name_upper}_COARSE)*{layer_name_upper}_DEPTH*{layer_name_upper}_HEIGHT*{layer_name_upper}_WIDTH;")
            if not "Gap" in layer_name:
                cpp(f"const int pixel_loop_out = {layer_name_upper}_BATCH_SIZE*DIVIDE({layer_name_upper}_CHANNELS, {layer_name_upper}_COARSE)*{layer_name_upper}_DEPTH*{layer_name_upper}_HEIGHT*{layer_name_upper}_WIDTH;", newlines=2)
            else:
                cpp(f"const int pixel_loop_out = {layer_name_upper}_BATCH_SIZE*DIVIDE({layer_name_upper}_CHANNELS, {layer_name_upper}_COARSE);", newlines=2)

            cpp(f"axis_to_stream<pixel_loop_in, {layer_name_upper}_COARSE, {layer_name_lower}_data_t, axi_stream_t>(in, in_stream);")
            cpp(f"{layer_name_lower}_layer(in_stream, out_stream);")
            cpp(f"stream_to_axis<pixel_loop_out, {layer_name_upper}_COARSE, {layer_name_lower}_data_t, axi_stream_t>(out_stream, out);")
    elif "Mul" in layer_name or "Add" in layer_name or "ElementWise" in layer_name:
        with cpp.block(
            f"void {layer_name_lower}_top(\n\
            stream_t(axi_stream_t) in_1[{layer_name_upper}_COARSE],\n\
            stream_t(axi_stream_t) in_2[{layer_name_upper}_COARSE],\n\
            stream_t(axi_stream_t) out[{layer_name_upper}_COARSE])"
        ):
            cpp("#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=in_1 register")
            cpp("#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=in_2 register")
            cpp("#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=out register")
            cpp("#pragma HLS INTERFACE mode=s_axilite bundle=control port=return", newlines=2)

            cpp("#pragma HLS ARRAY_PARTITION variable=in_1  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=in_2  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

            cpp("#pragma HLS DATAFLOW", newlines=2)

            cpp(f"stream_t({layer_name_lower}_data_t) in_stream_1[{layer_name_upper}_COARSE];")
            cpp(f"stream_t({layer_name_lower}_data_t) in_stream_2[{layer_name_upper}_COARSE];")
            cpp(f"stream_t({layer_name_lower}_data_t) out_stream[{layer_name_upper}_COARSE];", newlines=2)

            cpp("#pragma HLS ARRAY_PARTITION variable=in_stream_1  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=in_stream_2  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=out_stream complete dim=0", newlines=2)

            cpp(f"const int pixel_loop_in_1 = {layer_name_upper}_BATCH_SIZE*DIVIDE({layer_name_upper}_CHANNELS, {layer_name_upper}_COARSE)*{layer_name_upper}_DEPTH*{layer_name_upper}_HEIGHT*{layer_name_upper}_WIDTH;")
            cpp(f"const int pixel_loop_in_2 = {layer_name_upper}_BATCH_SIZE*DIVIDE({layer_name_upper}_CHANNELS, {layer_name_upper}_COARSE)*{layer_name_upper}_DEPTH*{layer_name_upper}_HEIGHT*{layer_name_upper}_WIDTH;")
            cpp(f"const int pixel_loop_out = {layer_name_upper}_BATCH_SIZE*DIVIDE({layer_name_upper}_CHANNELS, {layer_name_upper}_COARSE)*{layer_name_upper}_DEPTH*{layer_name_upper}_HEIGHT*{layer_name_upper}_WIDTH;", newlines=2)

            cpp(f"axis_to_stream<pixel_loop_in_1, {layer_name_upper}_COARSE, {layer_name_lower}_data_t, axi_stream_t>(in_1, in_stream_1);")
            cpp(f"axis_to_stream<pixel_loop_in_2, {layer_name_upper}_COARSE, {layer_name_lower}_data_t, axi_stream_t>(in_2, in_stream_2);")
            cpp(f"{layer_name_lower}_layer(in_stream_1, in_stream_2, out_stream);")
            cpp(f"stream_to_axis<pixel_loop_out, {layer_name_upper}_COARSE, {layer_name_lower}_data_t, axi_stream_t>(out_stream, out);")
    elif "Gemm" in layer_name:
        with cpp.block(
            f"void {layer_name_lower}_top(\n\
            stream_t(axi_stream_t) in[{layer_name_upper}_COARSE_IN],\n\
            stream_t(axi_stream_t) weights[{layer_name_upper}_COARSE_IN][{layer_name_upper}_COARSE_OUT],\n\
            stream_t(axi_stream_t) out[{layer_name_upper}_COARSE_OUT])"
        ):
            cpp("#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=in register")
            cpp("#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=weights register")
            cpp("#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=out register")
            cpp("#pragma HLS INTERFACE mode=s_axilite bundle=control port=return", newlines=2)

            cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=weights  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

            cpp("#pragma HLS DATAFLOW", newlines=2)

            cpp(f"stream_t({layer_name_lower}_data_t) in_stream[{layer_name_upper}_COARSE_IN];")
            cpp(f"stream_t({layer_name_lower}_data_t) weights_stream[{layer_name_upper}_COARSE_IN][{layer_name_upper}_COARSE_OUT];")
            cpp(f"stream_t({layer_name_lower}_data_t) out_stream[{layer_name_upper}_COARSE_OUT];", newlines=2)

            cpp("#pragma HLS ARRAY_PARTITION variable=in_stream  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=weights_stream  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=out_stream complete dim=0", newlines=2)

            cpp(f"const int pixel_loop_in = {layer_name_upper}_BATCH_SIZE*DIVIDE({layer_name_upper}_IN_FEATURES, {layer_name_upper}_COARSE_IN);")
            cpp(f"const int pixel_loop_weights = DIVIDE({layer_name_upper}_IN_FEATURES, {layer_name_upper}_COARSE_IN)*DIVIDE({layer_name_upper}_OUT_FEATURES, {layer_name_upper}_COARSE_OUT);")
            cpp(f"const int pixel_loop_out = {layer_name_upper}_BATCH_SIZE*DIVIDE({layer_name_upper}_OUT_FEATURES, {layer_name_upper}_COARSE_OUT);", newlines=2)

            cpp(f"axis_to_stream<pixel_loop_in, {layer_name_upper}_COARSE_IN, {layer_name_lower}_data_t, axi_stream_t>(in, in_stream);")
            cpp(f"axis_to_stream<pixel_loop_weights, {layer_name_upper}_COARSE_IN, {layer_name_upper}_COARSE_OUT, {layer_name_lower}_data_t, axi_stream_t>(weights, weights_stream);")
            cpp(f"{layer_name_lower}_layer(in_stream, weights_stream, out_stream);")
            cpp(f"stream_to_axis<pixel_loop_out, {layer_name_upper}_COARSE_OUT, {layer_name_lower}_data_t, axi_stream_t>(out_stream, out);")
    else:
        with cpp.block(
            f"void {layer_name_lower}_top(\n\
            stream_t(axi_stream_t) in[{layer_name_upper}_COARSE_IN],\n\
            stream_t(axi_stream_t) out[{layer_name_upper}_COARSE_OUT])"
        ):
            cpp("#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=in register")
            cpp("#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=out register")
            cpp("#pragma HLS INTERFACE mode=s_axilite bundle=control port=return", newlines=2)

            cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

            cpp("#pragma HLS DATAFLOW", newlines=2)

            cpp(f"stream_t({layer_name_lower}_data_t) in_stream[{layer_name_upper}_COARSE_IN];")
            cpp(f"stream_t({layer_name_lower}_data_t) out_stream[{layer_name_upper}_COARSE_OUT];", newlines=2)

            cpp("#pragma HLS ARRAY_PARTITION variable=in_stream  complete dim=0")
            cpp("#pragma HLS ARRAY_PARTITION variable=out_stream complete dim=0", newlines=2)

            cpp(f"const int pixel_loop_in = {layer_name_upper}_BATCH_SIZE*DIVIDE({layer_name_upper}_CHANNELS_IN, {layer_name_upper}_COARSE_IN)*{layer_name_upper}_DEPTH_IN*{layer_name_upper}_HEIGHT_IN*{layer_name_upper}_WIDTH_IN;")
            cpp(f"const int pixel_loop_out = {layer_name_upper}_BATCH_SIZE*DIVIDE({layer_name_upper}_CHANNELS_OUT, {layer_name_upper}_COARSE_OUT)*{layer_name_upper}_DEPTH_OUT*{layer_name_upper}_HEIGHT_OUT*{layer_name_upper}_WIDTH_OUT;", newlines=2)

            cpp(f"axis_to_stream<pixel_loop_in, {layer_name_upper}_COARSE_IN, {layer_name_lower}_data_t, axi_stream_t>(in, in_stream);")
            cpp(f"{layer_name_lower}_layer(in_stream, out_stream);")
            cpp(f"stream_to_axis<pixel_loop_out, {layer_name_upper}_COARSE_OUT, {layer_name_lower}_data_t, axi_stream_t>(out_stream, out);")

    cpp.close()

def generate_top_level_layer_hpp(layer_name: str, model_name: str, partition_name: str, dynamic_reconfig: bool):

    layer_name_lower = layer_name.lower()
    layer_name_upper = layer_name.upper()

    if dynamic_reconfig:
        hpp = CppFile(
            os.path.join(
                os.getcwd(),
                "generated_files",
                model_name,
                partition_name,
                "latency_driven",
                layer_name,
                "src",
                f"{layer_name_lower}_top.hpp"
            )
        )
    else:
        hpp = CppFile(
            os.path.join(
                os.getcwd(),
                "generated_files",
                model_name,
                partition_name,
                layer_name,
                "src",
                f"{layer_name_lower}_top.hpp"
            )
        )

    hpp("#pragma once", newlines=2)

    hpp(f'#include "{layer_name_lower}.hpp"', newlines=2)

    if "Mul" in layer_name or "Add" in layer_name or "ElementWise" in layer_name:
        hpp(
            f"void {layer_name_lower}_top(\n\
            stream_t(axi_stream_t) in_1[{layer_name_upper}_COARSE],\n\
            stream_t(axi_stream_t) in_2[{layer_name_upper}_COARSE],\n\
            stream_t(axi_stream_t) out[{layer_name_upper}_COARSE]);"
        )
    elif "Sigmoid" in layer_name or "Swish" in layer_name or "Relu" in layer_name or "Gap" in layer_name or "Activation" in layer_name:
        hpp(
            f"void {layer_name_lower}_top(\n\
            stream_t(axi_stream_t) in[{layer_name_upper}_COARSE],\n\
            stream_t(axi_stream_t) out[{layer_name_upper}_COARSE]);"
        )
    elif "Gemm" in layer_name:
        hpp(
            f"void {layer_name_lower}_top(\n\
            stream_t(axi_stream_t) in[{layer_name_upper}_COARSE_IN],\n\
            stream_t(axi_stream_t) weights[{layer_name_upper}_COARSE_IN][{layer_name_upper}_COARSE_OUT],\n\
            stream_t(axi_stream_t) out[{layer_name_upper}_COARSE_OUT]);"
        )
    else:
        hpp(
            f"void {layer_name_lower}_top(\n\
            stream_t(axi_stream_t) in[{layer_name_upper}_COARSE_IN],\n\
            stream_t(axi_stream_t) out[{layer_name_upper}_COARSE_OUT]);"
        )

    hpp.close()

def generate_top_level_layer_files(layer_name: str, model_name: str, partition_name: str="", dynamic_reconfig: bool=False):
    if dynamic_reconfig:
        if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "latency_driven", layer_name, "src")):
            os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "latency_driven", layer_name, "src"))
    else:
        if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, layer_name, "src")):
            os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, layer_name, "src"))

    generate_top_level_layer_hpp(layer_name, model_name, partition_name, dynamic_reconfig)
    generate_top_level_layer_cpp(layer_name, model_name, partition_name, dynamic_reconfig)
