import os

from layers.codegen import *


def generate_tb_cpp(partition_name, prefix, hls_project_path):
    partition_name_lower = partition_name.lower()
    partition_name_upper = partition_name.upper()

    data_dir = f"{hls_project_path}/{prefix}/{partition_name}/data"

    cpp = CppFile(
        os.path.join(
            os.getcwd(),
            "generated_files",
            f"{prefix}/{partition_name}/tb/{partition_name_lower}_tb.cpp",
        )
    )

    cpp(f'#include "common.hpp"')
    cpp(f'#include "common_tb.hpp"')
    cpp(f'#include "{partition_name_lower}.hpp"', newlines=2)

    cpp(f'#define DATA_DIR "{data_dir}"', newlines=2)

    with cpp.block("int main()"):
        cpp("int err = 0;", newlines=2)

        cpp('std::string input_path  = std::string(DATA_DIR)+"/input.dat";')
        cpp(
            'std::string output_path  = std::string(DATA_DIR)+"/output.dat";',
            newlines=2,
        )

        cpp(
            f"stream_t({partition_name_lower}_input_t) in[{partition_name_upper}_STREAMS_IN];"
        )
        cpp(
            f"stream_t({partition_name_lower}_output_t) out[{partition_name_upper}_STREAMS_OUT];"
        )
        cpp(
            f"stream_t({partition_name_lower}_output_t) out_correct[{partition_name_upper}_STREAMS_OUT];",
            newlines=2,
        )

        cpp("#ifdef MALLOC_USAGE")
        cpp(
            f"{partition_name_lower}_input_t* test_in = ({partition_name_lower}_input_t*)malloc({partition_name_upper}_BATCH_SIZE*{partition_name_upper}_CHANNELS_IN*{partition_name_upper}_HEIGHT_IN*{partition_name_upper}_WIDTH_IN*{partition_name_upper}_DEPTH_IN * sizeof({partition_name_lower}_input_t));"
        )
        cpp(
            f"{partition_name_lower}_output_t* test_out = ({partition_name_lower}_input_t*)malloc({partition_name_upper}_BATCH_SIZE*{partition_name_upper}_CHANNELS_OUT*{partition_name_upper}_HEIGHT_OUT*{partition_name_upper}_WIDTH_OUT*{partition_name_upper}_DEPTH_OUT * sizeof({partition_name_lower}_input_t));"
        )
        cpp("#else")
        cpp(
            f"static {partition_name_lower}_input_t test_in[{partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_IN,{partition_name_upper}_STREAMS_IN)*{partition_name_upper}_HEIGHT_IN*{partition_name_upper}_WIDTH_IN*{partition_name_upper}_DEPTH_IN][{partition_name_upper}_STREAMS_IN];"
        )
        cpp(
            f"static {partition_name_lower}_output_t test_out[{partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_OUT,{partition_name_upper}_STREAMS_OUT)*{partition_name_upper}_HEIGHT_OUT*{partition_name_upper}_WIDTH_OUT*{partition_name_upper}_DEPTH_OUT][{partition_name_upper}_STREAMS_OUT];"
        )
        cpp("#endif", newlines=2)

        cpp("// load input data")
        cpp(
            f"load_data<\n\
                {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_IN,{partition_name_upper}_STREAMS_IN)*{partition_name_upper}_HEIGHT_IN*{partition_name_upper}_WIDTH_IN*{partition_name_upper}_DEPTH_IN,\n\
                {partition_name_upper}_STREAMS_IN,\n\
                {partition_name_lower}_input_t\n\
              >(input_path, test_in);",
            newlines=2,
        )

        cpp("// load output data")
        cpp(
            f"load_data<\n\
                {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_OUT,{partition_name_upper}_STREAMS_OUT)*{partition_name_upper}_HEIGHT_OUT*{partition_name_upper}_WIDTH_OUT*{partition_name_upper}_DEPTH_OUT,\n\
                {partition_name_upper}_STREAMS_OUT,\n\
                {partition_name_lower}_output_t\n\
              >(output_path, test_out);",
            newlines=2,
        )

        cpp("// convert input to streams")
        cpp(
            f"to_stream<\n\
                {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_IN,{partition_name_upper}_STREAMS_IN)*{partition_name_upper}_HEIGHT_IN*{partition_name_upper}_WIDTH_IN*{partition_name_upper}_DEPTH_IN,\n\
                {partition_name_upper}_STREAMS_IN,\n\
                {partition_name_lower}_input_t\n\
              >(test_in, in);",
            newlines=2,
        )

        cpp("// convert output to streams")
        cpp(
            f"to_stream<\n\
                {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_OUT,{partition_name_upper}_STREAMS_OUT)*{partition_name_upper}_HEIGHT_OUT*{partition_name_upper}_WIDTH_OUT*{partition_name_upper}_DEPTH_OUT,\n\
                {partition_name_upper}_STREAMS_OUT,\n\
                {partition_name_lower}_output_t\n\
              >(test_out, out_correct);",
            newlines=2,
        )

        cpp(f"{partition_name_lower}_top(in, out);", newlines=2)

        with cpp.block(f"for(int i=0;i<{partition_name_upper}_STREAMS_OUT;i++)"):
            cpp('printf("TESTING OUTPUT %d: ",i);')
            cpp("err += checkStreamEqual(out[i], out_correct[i]);")
            cpp(
                'printf("stream:%d -> %s\\n",i, (err==0) ? "passed" : "failed");',
                newlines=2,
            )

        cpp("return err;")


def generate_tb_files(partition_name, prefix, hls_project_path):
    if not os.path.exists(
        os.path.join(os.getcwd(), "generated_files", f"{prefix}/{partition_name}/tb")
    ):
        os.makedirs(
            os.path.join(
                os.getcwd(), "generated_files", f"{prefix}/{partition_name}/tb"
            )
        )

    generate_tb_cpp(partition_name, prefix, hls_project_path)
