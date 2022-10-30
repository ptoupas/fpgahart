import os

from layers.codegen import *


def generate_tb_cpp(layer_name: str, model_name: str, partition_name: str, hls_project_path: str, is_layer: bool, dynamic_reconfig: bool, elem_bc: bool):
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
    cpp(f'#include "{partition_name_lower}_top.hpp"', newlines=2)

    cpp(f'#define DATA_DIR "{data_dir}"', newlines=2)

    if is_layer:
        # TODO: Create different interfaces for different types of layers
        data_type_postfix_in = "data_t"
        data_type_postfix_out = "data_t"
        if "Sigmoid" in layer_name or "Swish" in layer_name or "Relu" in layer_name or "Gap" in layer_name or "Mul" in layer_name or "Add" in layer_name:
            input_streams = "COARSE"
            output_streams = "COARSE"
            cin_dim = f"{partition_name_upper}_CHANNELS"
            hin_dim = f"{partition_name_upper}_HEIGHT"
            win_dim = f"{partition_name_upper}_WIDTH"
            din_dim = f"{partition_name_upper}_DEPTH"
            cout_dim = f"{partition_name_upper}_CHANNELS"
            hout_dim = f"{partition_name_upper}_HEIGHT"
            wout_dim = f"{partition_name_upper}_WIDTH"
            dout_dim = f"{partition_name_upper}_DEPTH"
        else:
            input_streams = "COARSE_IN"
            output_streams = "COARSE_OUT"
            cin_dim = f"{partition_name_upper}_CHANNELS_IN"
            hin_dim = f"{partition_name_upper}_HEIGHT_IN"
            win_dim = f"{partition_name_upper}_WIDTH_IN"
            din_dim = f"{partition_name_upper}_DEPTH_IN"
            cout_dim = f"{partition_name_upper}_CHANNELS_OUT"
            hout_dim = f"{partition_name_upper}_HEIGHT_OUT"
            wout_dim = f"{partition_name_upper}_WIDTH_OUT"
            dout_dim = f"{partition_name_upper}_DEPTH_OUT"
    else:
        data_type_postfix_in = "input_t"
        data_type_postfix_out = "output_t"
        input_streams = "STREAMS_IN"
        output_streams = "STREAMS_OUT"
        cin_dim = f"{partition_name_upper}_CHANNELS_IN"
        hin_dim = f"{partition_name_upper}_HEIGHT_IN"
        win_dim = f"{partition_name_upper}_WIDTH_IN"
        din_dim = f"{partition_name_upper}_DEPTH_IN"
        cout_dim = f"{partition_name_upper}_CHANNELS_OUT"
        hout_dim = f"{partition_name_upper}_HEIGHT_OUT"
        wout_dim = f"{partition_name_upper}_WIDTH_OUT"
        dout_dim = f"{partition_name_upper}_DEPTH_OUT"
    with cpp.block("int main()"):
        cpp("int err = 0;", newlines=2)

        if "Mul" in layer_name or "Add" in layer_name or "ElementWise" in layer_name:
            cpp('std::string input_path_1  = std::string(DATA_DIR)+"/input1.bin";')
            cpp('std::string input_path_2  = std::string(DATA_DIR)+"/input2.bin";')
        elif "Gemm" in layer_name:
            cpp('std::string input_path  = std::string(DATA_DIR)+"/input.bin";')
            cpp('std::string weights_path  = std::string(DATA_DIR)+"/weights.bin";')
        else:
            cpp('std::string input_path  = std::string(DATA_DIR)+"/input.bin";')
        cpp(
            'std::string output_path  = std::string(DATA_DIR)+"/output.bin";',
            newlines=2,
        )

        if "Mul" in layer_name or "Add" in layer_name or "ElementWise" in layer_name:
            cpp(
                f"stream_t(axi_stream_t) in_1[{partition_name_upper}_{input_streams}];"
            )
            cpp(
                f"stream_t(axi_stream_t) in_2[{partition_name_upper}_{input_streams}];"
            )
        elif "Gemm" in layer_name:
            cpp(
                f"stream_t(axi_stream_t) in[{partition_name_upper}_{input_streams}];"
            )
            cpp(
                f"stream_t(axi_stream_t) weights[{partition_name_upper}_{input_streams}][{partition_name_upper}_{output_streams}];"
            )
        else:
            cpp(
                f"stream_t(axi_stream_t) in[{partition_name_upper}_{input_streams}];"
            )
        cpp(
            f"stream_t(axi_stream_t) out[{partition_name_upper}_{output_streams}];"
        )
        cpp(
            f"stream_t(axi_stream_t) out_correct[{partition_name_upper}_{output_streams}];",
            newlines=2,
        )

        cpp("#ifdef MALLOC_USAGE")
        if "Mul" in layer_name or "Add" in layer_name or "ElementWise" in layer_name:
            cpp(
                f"{partition_name_lower}_{data_type_postfix_in}* test_in_1 = ({partition_name_lower}_{data_type_postfix_in}*)malloc({partition_name_upper}_BATCH_SIZE*{cin_dim}*{hin_dim}*{win_dim}*{din_dim} * sizeof({partition_name_lower}_{data_type_postfix_in}));"
            )
            if not elem_bc:
                cpp(
                    f"{partition_name_lower}_{data_type_postfix_in}* test_in_2 = ({partition_name_lower}_{data_type_postfix_in}*)malloc({partition_name_upper}_BATCH_SIZE*{cin_dim}*{hin_dim}*{win_dim}*{din_dim} * sizeof({partition_name_lower}_{data_type_postfix_in}));"
                )
            else:
                cpp(
                    f"{partition_name_lower}_{data_type_postfix_in}* test_in_2 = ({partition_name_lower}_{data_type_postfix_in}*)malloc({partition_name_upper}_BATCH_SIZE*{cin_dim} * sizeof({partition_name_lower}_{data_type_postfix_in}));"
                )
        elif "Gemm" in layer_name:
            cpp(
                f"{partition_name_lower}_{data_type_postfix_in}* test_in = ({partition_name_lower}_{data_type_postfix_in}*)malloc({partition_name_upper}_BATCH_SIZE*{partition_name_upper}_IN_FEATURES * sizeof({partition_name_lower}_{data_type_postfix_in}));"
            )
            cpp(
                f"{partition_name_lower}_{data_type_postfix_in}* test_weights = ({partition_name_lower}_{data_type_postfix_in}*)malloc({partition_name_upper}_IN_FEATURES*{partition_name_upper}_OUT_FEATURES * sizeof({partition_name_lower}_{data_type_postfix_in}));"
            )
        else:
            cpp(
                f"{partition_name_lower}_{data_type_postfix_in}* test_in = ({partition_name_lower}_{data_type_postfix_in}*)malloc({partition_name_upper}_BATCH_SIZE*{cin_dim}*{hin_dim}*{win_dim}*{din_dim} * sizeof({partition_name_lower}_{data_type_postfix_in}));"
            )
        if "Gemm" in layer_name:
            cpp(
                f"{partition_name_lower}_{data_type_postfix_out}* test_out = ({partition_name_lower}_{data_type_postfix_in}*)malloc({partition_name_upper}_BATCH_SIZE*{partition_name_upper}_OUT_FEATURES * sizeof({partition_name_lower}_{data_type_postfix_in}));"
            )
        else:
            cpp(
                f"{partition_name_lower}_{data_type_postfix_out}* test_out = ({partition_name_lower}_{data_type_postfix_in}*)malloc({partition_name_upper}_BATCH_SIZE*{cout_dim}*{hout_dim}*{wout_dim}*{dout_dim} * sizeof({partition_name_lower}_{data_type_postfix_in}));"
            )
        cpp("#else")
        if "Mul" in layer_name or "Add" in layer_name or "ElementWise" in layer_name:
            cpp(
                f"static {partition_name_lower}_{data_type_postfix_in} test_in_1[{partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams})*{hin_dim}*{win_dim}*{din_dim}][{partition_name_upper}_{input_streams}];"
            )
            if not elem_bc:
                cpp(
                    f"static {partition_name_lower}_{data_type_postfix_in} test_in_2[{partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams})*{hin_dim}*{win_dim}*{din_dim}][{partition_name_upper}_{input_streams}];"
                )
            else:
                cpp(
                    f"static {partition_name_lower}_{data_type_postfix_in} test_in_2[{partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams})][{partition_name_upper}_{input_streams}];"
                )
        elif "Gemm" in layer_name:
            cpp(
                f"static {partition_name_lower}_{data_type_postfix_in} test_in[{partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_IN_FEATURES,{partition_name_upper}_{input_streams})][{partition_name_upper}_{input_streams}];"
            )
            cpp(
                f"static {partition_name_lower}_{data_type_postfix_in} test_weights[DIVIDE({partition_name_upper}_IN_FEATURES,{partition_name_upper}_{input_streams})*DIVIDE({partition_name_upper}_OUT_FEATURES,{partition_name_upper}_{output_streams})][{partition_name_upper}_{input_streams}][{partition_name_upper}_{output_streams}];"
            )
        else:
            cpp(
                f"static {partition_name_lower}_{data_type_postfix_in} test_in[{partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams})*{hin_dim}*{win_dim}*{din_dim}][{partition_name_upper}_{input_streams}];"
            )

        if "Gemm" in layer_name:
            cpp(
                f"static {partition_name_lower}_{data_type_postfix_out} test_out[{partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_OUT_FEATURES,{partition_name_upper}_{output_streams})][{partition_name_upper}_{output_streams}];"
            )
        else:
            cpp(
                f"static {partition_name_lower}_{data_type_postfix_out} test_out[{partition_name_upper}_BATCH_SIZE*DIVIDE({cout_dim},{partition_name_upper}_{output_streams})*{hout_dim}*{wout_dim}*{dout_dim}][{partition_name_upper}_{output_streams}];"
            )
        cpp("#endif", newlines=2)

        cpp("// load input data")
        if "Mul" in layer_name or "Add" in layer_name or "ElementWise" in layer_name:
            cpp(
                f"load_data<\n\
                    {partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams})*{hin_dim}*{win_dim}*{din_dim},\n\
                    {partition_name_upper}_{input_streams},\n\
                    {partition_name_lower}_{data_type_postfix_in}\n\
                >(input_path_1, test_in_1);",
                newlines=2,
            )
            if not elem_bc:
                cpp(
                    f"load_data<\n\
                        {partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams})*{hin_dim}*{win_dim}*{din_dim},\n\
                        {partition_name_upper}_{input_streams},\n\
                        {partition_name_lower}_{data_type_postfix_in}\n\
                    >(input_path_2, test_in_2);",
                    newlines=2,
                )
            else:
                cpp(
                    f"load_data<\n\
                        {partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams}),\n\
                        {partition_name_upper}_{input_streams},\n\
                        {partition_name_lower}_{data_type_postfix_in}\n\
                    >(input_path_2, test_in_2);",
                    newlines=2,
                )
        elif "Gemm" in layer_name:
            cpp(
                f"load_data<\n\
                    {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_IN_FEATURES,{partition_name_upper}_{input_streams}),\n\
                    {partition_name_upper}_{input_streams},\n\
                    {partition_name_lower}_{data_type_postfix_in}\n\
                >(input_path, test_in);",
                newlines=2,
            )
            cpp(
                f"load_data<\n\
                    DIVIDE({partition_name_upper}_IN_FEATURES,{partition_name_upper}_{input_streams})*DIVIDE({partition_name_upper}_OUT_FEATURES,{partition_name_upper}_{output_streams}),\n\
                    {partition_name_upper}_{input_streams},\n\
                    {partition_name_upper}_{output_streams},\n\
                    {partition_name_lower}_{data_type_postfix_in}\n\
                >(weights_path, test_weights);",
                newlines=2,
            )
        else:
            cpp(
                f"load_data<\n\
                    {partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams})*{hin_dim}*{win_dim}*{din_dim},\n\
                    {partition_name_upper}_{input_streams},\n\
                    {partition_name_lower}_{data_type_postfix_in}\n\
                >(input_path, test_in);",
                newlines=2,
            )

        cpp("// load output data")

        if "Gemm" in layer_name:
            cpp(
                f"load_data<\n\
                    {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_OUT_FEATURES,{partition_name_upper}_{output_streams}),\n\
                    {partition_name_upper}_{output_streams},\n\
                    {partition_name_lower}_{data_type_postfix_out}\n\
                >(output_path, test_out);",
                newlines=2,
            )
        else:
            cpp(
                f"load_data<\n\
                    {partition_name_upper}_BATCH_SIZE*DIVIDE({cout_dim},{partition_name_upper}_{output_streams})*{hout_dim}*{wout_dim}*{dout_dim},\n\
                    {partition_name_upper}_{output_streams},\n\
                    {partition_name_lower}_{data_type_postfix_out}\n\
                >(output_path, test_out);",
                newlines=2,
            )

        cpp("// convert input to streams")
        if "Mul" in layer_name or "Add" in layer_name or "ElementWise" in layer_name:
            cpp(
                f"to_axis_stream<\n\
                    {partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams})*{hin_dim}*{win_dim}*{din_dim},\n\
                    {partition_name_upper}_{input_streams},\n\
                    {partition_name_lower}_{data_type_postfix_in},\n\
                    axi_stream_t\n\
                >(test_in_1, in_1);",
                newlines=2,
            )
            if not elem_bc:
                cpp(
                    f"to_axis_stream<\n\
                        {partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams})*{hin_dim}*{win_dim}*{din_dim},\n\
                        {partition_name_upper}_{input_streams},\n\
                        {partition_name_lower}_{data_type_postfix_in},\n\
                        axi_stream_t\n\
                    >(test_in_2, in_2);",
                    newlines=2,
                )
            else:
                cpp(
                    f"to_axis_stream<\n\
                        {partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams}),\n\
                        {partition_name_upper}_{input_streams},\n\
                        {partition_name_lower}_{data_type_postfix_in},\n\
                        axi_stream_t\n\
                    >(test_in_2, in_2);",
                    newlines=2,
                )
        elif "Gemm" in layer_name:
            cpp(
                f"to_axis_stream<\n\
                    {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_IN_FEATURES,{partition_name_upper}_{input_streams}),\n\
                    {partition_name_upper}_{input_streams},\n\
                    {partition_name_lower}_{data_type_postfix_in},\n\
                    axi_stream_t\n\
                >(test_in, in);",
                newlines=2,
            )
            cpp(
                f"to_axis_stream<\n\
                    DIVIDE({partition_name_upper}_IN_FEATURES,{partition_name_upper}_{input_streams})*DIVIDE({partition_name_upper}_OUT_FEATURES,{partition_name_upper}_{output_streams}),\n\
                    {partition_name_upper}_{input_streams},\n\
                    {partition_name_upper}_{output_streams},\n\
                    {partition_name_lower}_{data_type_postfix_in},\n\
                    axi_stream_t\n\
                >(test_weights, weights);",
                newlines=2,
            )
        else:
            cpp(
                f"to_axis_stream<\n\
                    {partition_name_upper}_BATCH_SIZE*DIVIDE({cin_dim},{partition_name_upper}_{input_streams})*{hin_dim}*{win_dim}*{din_dim},\n\
                    {partition_name_upper}_{input_streams},\n\
                    {partition_name_lower}_{data_type_postfix_in},\n\
                    axi_stream_t\n\
                >(test_in, in);",
                newlines=2,
            )

        cpp("// convert output to streams")
        if "Gemm" in layer_name:
            cpp(
                f"to_axis_stream<\n\
                    {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_OUT_FEATURES,{partition_name_upper}_{output_streams}),\n\
                    {partition_name_upper}_{output_streams},\n\
                    {partition_name_lower}_{data_type_postfix_out},\n\
                    axi_stream_t\n\
                >(test_out, out_correct);",
                newlines=2,
            )
        else:
            cpp(
                f"to_axis_stream<\n\
                    {partition_name_upper}_BATCH_SIZE*DIVIDE({cout_dim},{partition_name_upper}_{output_streams})*{hout_dim}*{wout_dim}*{dout_dim},\n\
                    {partition_name_upper}_{output_streams},\n\
                    {partition_name_lower}_{data_type_postfix_out},\n\
                    axi_stream_t\n\
                >(test_out, out_correct);",
                newlines=2,
            )

        if "Mul" in layer_name or "Add" in layer_name or "ElementWise" in layer_name:
            cpp(f"{partition_name_lower}_top(in_1, in_2, out);", newlines=2)
        elif "Gemm" in layer_name:
            cpp(f"{partition_name_lower}_top(in, weights, out);", newlines=2)
        else:
            cpp(f"{partition_name_lower}_top(in, out);", newlines=2)

        with cpp.block(f"for(int i=0;i<{partition_name_upper}_{output_streams};i++)"):
            cpp('printf("TESTING OUTPUT %d: ",i);')
            cpp(f"err += checkAxisStreamEqual<{partition_name_lower}_{data_type_postfix_out}, axi_stream_t>(out[i], out_correct[i]);")
            cpp(
                'printf("stream:%d -> %s\\n",i, (err==0) ? "passed" : "failed");',
                newlines=2,
            )

        cpp("return err;")


def generate_tb_files(layer_name: str, model_name: str, hls_project_path: str, partition_name: str="", is_layer: bool=False, dynamic_reconfig: bool=False, elem_bc: bool = False):
    if dynamic_reconfig:
        if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "latency_driven", layer_name, "tb")):
            os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "latency_driven", layer_name, "tb"))
    else:
        if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, layer_name, "tb")):
            os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, layer_name, "tb"))

    generate_tb_cpp(layer_name, model_name, partition_name, hls_project_path, is_layer, dynamic_reconfig, elem_bc)
