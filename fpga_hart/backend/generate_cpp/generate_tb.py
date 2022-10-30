import os

from layers.codegen import *


def generate_tb_cpp(layer_name: str, model_name: str, partition_name: str, hls_project_path: str, is_layer: bool, dynamic_reconfig: bool):
    if is_layer:
        partition_name_lower = layer_name.lower()
        partition_name_upper = layer_name.upper()
    else:
        partition_name_lower = partition_name.lower()
        partition_name_upper = partition_name.upper()

    if dynamic_reconfig:
        data_dir = os.path.join(hls_project_path, model_name, partition_name, "latency_driven", layer_name, "data")
        cpp = CppFile(
            os.path.join(
                os.getcwd(),
                "generated_files",
                model_name,
                partition_name,
                "latency_driven",
                layer_name,
                "tb",
                f"{partition_name_lower}_tb.cpp"
            )
        )
    else:
        data_dir = os.path.join(hls_project_path, model_name, partition_name, layer_name, "data")
        cpp = CppFile(
            os.path.join(
                os.getcwd(),
                "generated_files",
                model_name,
                partition_name,
                layer_name,
                "tb",
                f"{partition_name_lower}_tb.cpp"
            )
        )

    cpp(f'#include "common_.hpp"')
    cpp(f'#include "common_tb_.hpp"')
    cpp(f'#include "{partition_name_lower}.hpp"', newlines=2)

    cpp(f'#define DATA_DIR "{data_dir}"', newlines=2)

    if is_layer:
        # TODO: Create different interfaces for different types of layers
        data_type_postfix_in = "data_t"
        data_type_postfix_out = "data_t"
        input_streams = "COARSE_IN"
        output_streams = "COARSE_OUT"
    else:
        data_type_postfix_in = "input_t"
        data_type_postfix_out = "output_t"
        input_streams = "STREAMS_IN"
        output_streams = "STREAMS_OUT"
    with cpp.block("int main()"):
        cpp("int err = 0;", newlines=2)

        # TODO: Create different interfaces for different types of layers
        cpp('std::string input_path  = std::string(DATA_DIR)+"/input.bin";')
        cpp(
            'std::string output_path  = std::string(DATA_DIR)+"/output.bin";',
            newlines=2,
        )

        cpp(
            f"stream_t({partition_name_lower}_{data_type_postfix_in}) in[{partition_name_upper}_{input_streams}];"
        )
        cpp(
            f"stream_t({partition_name_lower}_{data_type_postfix_out}) out[{partition_name_upper}_{output_streams}];"
        )
        cpp(
            f"stream_t({partition_name_lower}_{data_type_postfix_out}) out_correct[{partition_name_upper}_{output_streams}];",
            newlines=2,
        )

        cpp("#ifdef MALLOC_USAGE")
        cpp(
            f"{partition_name_lower}_{data_type_postfix_in}* test_in = ({partition_name_lower}_{data_type_postfix_in}*)malloc({partition_name_upper}_BATCH_SIZE*{partition_name_upper}_CHANNELS_IN*{partition_name_upper}_HEIGHT_IN*{partition_name_upper}_WIDTH_IN*{partition_name_upper}_DEPTH_IN * sizeof({partition_name_lower}_{data_type_postfix_in}));"
        )
        cpp(
            f"{partition_name_lower}_{data_type_postfix_out}* test_out = ({partition_name_lower}_{data_type_postfix_in}*)malloc({partition_name_upper}_BATCH_SIZE*{partition_name_upper}_CHANNELS_OUT*{partition_name_upper}_HEIGHT_OUT*{partition_name_upper}_WIDTH_OUT*{partition_name_upper}_DEPTH_OUT * sizeof({partition_name_lower}_{data_type_postfix_in}));"
        )
        cpp("#else")
        cpp(
            f"static {partition_name_lower}_{data_type_postfix_in} test_in[{partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_IN,{partition_name_upper}_{input_streams})*{partition_name_upper}_HEIGHT_IN*{partition_name_upper}_WIDTH_IN*{partition_name_upper}_DEPTH_IN][{partition_name_upper}_{input_streams}];"
        )
        cpp(
            f"static {partition_name_lower}_{data_type_postfix_out} test_out[{partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_OUT,{partition_name_upper}_{output_streams})*{partition_name_upper}_HEIGHT_OUT*{partition_name_upper}_WIDTH_OUT*{partition_name_upper}_DEPTH_OUT][{partition_name_upper}_{output_streams}];"
        )
        cpp("#endif", newlines=2)

        cpp("// load input data")
        cpp(
            f"load_data<\n\
                {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_IN,{partition_name_upper}_{input_streams})*{partition_name_upper}_HEIGHT_IN*{partition_name_upper}_WIDTH_IN*{partition_name_upper}_DEPTH_IN,\n\
                {partition_name_upper}_{input_streams},\n\
                {partition_name_lower}_{data_type_postfix_in}\n\
              >(input_path, test_in);",
            newlines=2,
        )

        cpp("// load output data")
        cpp(
            f"load_data<\n\
                {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_OUT,{partition_name_upper}_{output_streams})*{partition_name_upper}_HEIGHT_OUT*{partition_name_upper}_WIDTH_OUT*{partition_name_upper}_DEPTH_OUT,\n\
                {partition_name_upper}_{output_streams},\n\
                {partition_name_lower}_{data_type_postfix_out}\n\
              >(output_path, test_out);",
            newlines=2,
        )

        cpp("// convert input to streams")
        cpp(
            f"to_stream<\n\
                {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_IN,{partition_name_upper}_{input_streams})*{partition_name_upper}_HEIGHT_IN*{partition_name_upper}_WIDTH_IN*{partition_name_upper}_DEPTH_IN,\n\
                {partition_name_upper}_{input_streams},\n\
                {partition_name_lower}_{data_type_postfix_in}\n\
              >(test_in, in);",
            newlines=2,
        )

        cpp("// convert output to streams")
        cpp(
            f"to_stream<\n\
                {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_OUT,{partition_name_upper}_{output_streams})*{partition_name_upper}_HEIGHT_OUT*{partition_name_upper}_WIDTH_OUT*{partition_name_upper}_DEPTH_OUT,\n\
                {partition_name_upper}_{output_streams},\n\
                {partition_name_lower}_{data_type_postfix_out}\n\
              >(test_out, out_correct);",
            newlines=2,
        )

        if is_layer:
            cpp(f"{partition_name_lower}_layer(in, out);", newlines=2)
        else:
            cpp(f"{partition_name_lower}_top(in, out);", newlines=2)

        with cpp.block(f"for(int i=0;i<{partition_name_upper}_{output_streams};i++)"):
            cpp('printf("TESTING OUTPUT %d: ",i);')
            cpp("err += checkStreamEqual(out[i], out_correct[i]);")
            cpp(
                'printf("stream:%d -> %s\\n",i, (err==0) ? "passed" : "failed");',
                newlines=2,
            )

        cpp("return err;")


def generate_tb_files(layer_name: str, model_name: str, hls_project_path: str, partition_name: str="", is_layer: bool=False, dynamic_reconfig: bool=False):
    if dynamic_reconfig:
        if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "latency_driven", layer_name, "tb")):
            os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "latency_driven", layer_name, "tb"))
    else:
        if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, layer_name, "tb")):
            os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, layer_name, "tb"))

    generate_tb_cpp(layer_name, model_name, partition_name, hls_project_path, is_layer, dynamic_reconfig)
