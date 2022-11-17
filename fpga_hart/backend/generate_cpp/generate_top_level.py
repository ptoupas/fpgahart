import os
import re

import numpy as np
from layers.codegen import *

from fpga_hart.utils.utils import (generate_supportive_layer_config,
                                   get_branch_edges, get_input_node,
                                   get_output_node)


def find_fifo_depth(edge_out, partition_structure, branch_depth):
    for k, v in branch_depth.items():
        if v['end'] == edge_out[1]:
            conn_node = edge_out[0]
            if conn_node == v['conn']:
                return v['depth']
            if partition_structure[conn_node]['type'] == "Squeeze":
                ref_layer_in = partition_structure[conn_node]['ref_layer_in']
                if partition_structure[ref_layer_in]['type'] == "Split":
                    new_ref_layer = partition_structure[ref_layer_in]['ref_layer']
                    if new_ref_layer == v['conn']:
                        return v['depth']
                    else:
                        return 2
                else:
                    if ref_layer_in == v['conn']:
                        return v['depth']
                    else:
                        return 2
            if partition_structure[conn_node]['type'] == "Split":
                if partition_structure[conn_node]['ref_layer'] == v['conn']:
                    return v['depth']
                else:
                    return 2
            raise Exception("Error in finding fifo depth")
    return 2

def generate_partition_level_cpp(partition_name: str, model_name: str, layers_config: dict, partition_structure: dict, branch_depth: dict, input_nodes: list, output_nodes: list):

    coarse_factor_in = []
    for node in input_nodes:
        coarse_factor_in.append(layers_config[node]["coarse_factor"] if "coarse_factor" in layers_config[node] else layers_config[node]["coarse_in_factor"])

    coarse_factor_out = []
    for node in output_nodes:
        coarse_factor_out.append(layers_config[node]["coarse_factor"] if "coarse_factor" in layers_config[node] else layers_config[node]["coarse_out_factor"])

    partition_name_lower = partition_name.lower()
    partition_name_upper = partition_name.upper()

    cpp = CppFile(
        os.path.join(
            os.getcwd(),
            "generated_files",
            model_name,
            partition_name,
            "src", f"{partition_name_lower}.cpp",
        ))

    cpp(f'#include "{partition_name_lower}.hpp"', newlines=2)

    cpp(f"void {partition_name_lower}(")
    for i in range(len(coarse_factor_in)):
        cpp(f"\tstream_t({partition_name_lower}_input_t) in_{i}[{partition_name_upper}_STREAMS_IN_{i}],")
    for i in range(len(coarse_factor_out)):
        if i == len(coarse_factor_out) - 1:
            cpp(f"\tstream_t({partition_name_lower}_output_t) out_{i}[{partition_name_upper}_STREAMS_OUT_{i}]\n)")
        else:
            cpp(f"\tstream_t({partition_name_lower}_output_t) out_{i}[{partition_name_upper}_STREAMS_OUT_{i}],")
    with cpp.block(''):

        cpp("#pragma HLS INLINE OFF")

        for i in range(len(coarse_factor_in)):
            cpp(f"#pragma HLS ARRAY_PARTITION variable=in_{i} complete dim=0")
        for i in range(len(coarse_factor_out)):
            if i == len(coarse_factor_out) - 1:
                cpp(f"#pragma HLS ARRAY_PARTITION variable=out_{i} complete dim=0", newlines=2)
            else:
                cpp(f"#pragma HLS ARRAY_PARTITION variable=out_{i} complete dim=0")


        for l, v in partition_structure['layers'].items():
            if v['type'] == "mem_in" or v['type'] == "mem_out":
                continue
            node_data_type = l.lower().replace("globalaveragepool", "gap") + "_data_t"
            node_coarse_out = l.upper().replace("GLOBALAVERAGEPOOL", "GAP") + "_COARSE_OUT" if v['type'] in ["Conv", "Pooling", "Gemm", "Squeeze"] else l.upper().replace("GLOBALAVERAGEPOOL", "GAP") + "_COARSE"

            for n_out, e_out in zip(v['out_nodes'], v['out_edges']):
                if "Mem_out" in n_out:
                    continue
                fifo_name = e_out[0].lower() + "_" + e_out[1].lower()
                fifo_depth = find_fifo_depth(e_out, partition_structure['layers'], branch_depth)
                cpp(f"stream_t({node_data_type}) {fifo_name}[{node_coarse_out}];")
                cpp(f"#pragma HLS STREAM variable={fifo_name} depth={fifo_depth}")
                cpp(
                    f"#pragma HLS ARRAY_PARTITION variable={fifo_name} complete dim=0",
                    newlines=2,
                )

        cpp("#pragma HLS DATAFLOW", newlines=2)

        for l, v in partition_structure['layers'].items():
            if v['type'] == "mem_in" or v['type'] == "mem_out":
                continue
            node_name = l.lower() + "_layer"
            node_name = node_name.replace("globalaveragepool", "gap")
            in_streams = ""
            out_streams = ""
            for n_in, e_in in zip(v['in_nodes'], v['in_edges']):
                if "Mem_in" in n_in:
                    in_id = int(n_in.split("Mem_in")[-1]) - 1
                    in_streams += f"in_{in_id}, "
                else:
                    in_streams += f"{e_in[0]}_{e_in[1]}, "
            for i, (n_out, e_out) in enumerate(zip(v['out_nodes'], v['out_edges'])):
                if i == len(v['out_nodes']) - 1:
                    postfix = ""
                else:
                    postfix = ", "
                if "Mem_out" in n_out:
                    out_id = int(n_out.split("Mem_out")[-1]) - 1
                    out_streams += f"out_{out_id}{postfix}"
                else:
                    out_streams += f"{e_out[0]}_{e_out[1]}{postfix}"
            cpp(f"{node_name}({in_streams.lower()}{out_streams.lower()});", newlines=2)

    cpp.close()


def generate_partition_level_hpp(partition_name: str, model_name: str, layers_config: dict, header_files: list, input_nodes: list, output_nodes: list):
    batch_size = layers_config[input_nodes[0]]["batch_size"]

    channels_in = []
    depth_in = []
    height_in = []
    width_in = []
    coarse_factor_in = []

    for node in input_nodes:
        channels_in.append(layers_config[node]["channels_in"] if "channels_in" in layers_config[node] else layers_config[node]["features_in"])
        depth_in.append(layers_config[node]["depth_in"] if "depth_in" in layers_config[node] else 1)
        height_in.append(layers_config[node]["height_in"] if "height_in" in layers_config[node] else 1)
        width_in.append(layers_config[node]["width_in"] if "width_in" in layers_config[node] else 1)
        coarse_factor_in.append(layers_config[node]["coarse_factor"] if "coarse_factor" in layers_config[node] else layers_config[node]["coarse_in_factor"])

    channels_out = []
    depth_out = []
    height_out = []
    width_out = []
    coarse_factor_out = []

    for node in output_nodes:
        channels_out.append(layers_config[node]["channels_out"] if "channels_out" in layers_config[node] else layers_config[node]["features_out"])
        depth_out.append(layers_config[node]["depth_out"] if "depth_out" in layers_config[node] else 1)
        height_out.append(layers_config[node]["height_out"] if "height_out" in layers_config[node] else 1)
        width_out.append(layers_config[node]["width_out"] if "width_out" in layers_config[node] else 1)
        coarse_factor_out.append(layers_config[node]["coarse_factor"] if "coarse_factor" in layers_config[node] else layers_config[node]["coarse_out_factor"])

    partition_name_lower = partition_name.lower()
    partition_name_upper = partition_name.upper()

    hpp = CppFile(
        os.path.join(
            os.getcwd(),
            "generated_files",
            model_name,
            partition_name,
            "src", f"{partition_name_lower}.hpp",
        ))

    hpp("#pragma once", newlines=2)
    hpp('#include "common_.hpp"')
    for include_file in header_files:
        hpp(f'#include "{include_file}"')
    hpp("\n")

    hpp(f"#define {partition_name_upper}_BATCH_SIZE {batch_size}", newlines=2)

    for i, (cin, din, hin, win) in enumerate(zip(channels_in, depth_in, height_in, width_in)):
        hpp(f"#define {partition_name_upper}_CHANNELS_IN_{i} {cin}")
        hpp(f"#define {partition_name_upper}_DEPTH_IN_{i} {din}")
        hpp(f"#define {partition_name_upper}_HEIGHT_IN_{i} {hin}")
        hpp(f"#define {partition_name_upper}_WIDTH_IN_{i} {win}", newlines=2)

    for i, (cout, dout, hout, wout) in enumerate(zip(channels_out, depth_out, height_out, width_out)):
        hpp(f"#define {partition_name_upper}_CHANNELS_OUT_{i} {cout}")
        hpp(f"#define {partition_name_upper}_DEPTH_OUT_{i} {dout}")
        hpp(f"#define {partition_name_upper}_HEIGHT_OUT_{i} {hout}")
        hpp(f"#define {partition_name_upper}_WIDTH_OUT_{i} {wout}", newlines=2)

    for i, cfin in enumerate(coarse_factor_in):
        hpp(f"#define {partition_name_upper}_STREAMS_IN_{i} {cfin}")
    for i, cfout in enumerate(coarse_factor_out):
        hpp(f"#define {partition_name_upper}_STREAMS_OUT_{i} {cfout}", newlines=2)

    hpp(f"typedef ap_fixed<16,8,AP_RND, AP_SAT> \t{partition_name_lower}_input_t;"
        )
    hpp(
        f"typedef ap_fixed<16,8,AP_RND, AP_SAT> \t{partition_name_lower}_output_t;",
        newlines=3,
    )

    hpp(f"void {partition_name_lower}(")
    for i in range(len(coarse_factor_in)):
        hpp(f"\tstream_t({partition_name_lower}_input_t) in_{i}[{partition_name_upper}_STREAMS_IN_{i}],")
    for i in range(len(coarse_factor_out)):
        if i == len(coarse_factor_out) - 1:
            hpp(f"\tstream_t({partition_name_lower}_output_t) out_{i}[{partition_name_upper}_STREAMS_OUT_{i}]\n);")
        else:
            hpp(f"\tstream_t({partition_name_lower}_output_t) out_{i}[{partition_name_upper}_STREAMS_OUT_{i}],")

    hpp.close()


def generate_top_level_cpp(partition_name: str, model_name: str, layers_config: dict, input_nodes: list, output_nodes: list):

    coarse_factor_in = []
    for node in input_nodes:
        coarse_factor_in.append(layers_config[node]["coarse_factor"] if "coarse_factor" in layers_config[node] else layers_config[node]["coarse_in_factor"])

    coarse_factor_out = []
    for node in output_nodes:
        coarse_factor_out.append(layers_config[node]["coarse_factor"] if "coarse_factor" in layers_config[node] else layers_config[node]["coarse_out_factor"])

    partition_name_lower = partition_name.lower()
    partition_name_upper = partition_name.upper()

    cpp = CppFile(
        os.path.join(
            os.getcwd(),
            "generated_files",
            model_name,
            partition_name,
            "src", f"{partition_name_lower}_top.cpp",
        ))

    cpp(f'#include "{partition_name_lower}_top.hpp"', newlines=2)

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

    cpp(f"void {partition_name_lower}_top(")
    for i in range(len(coarse_factor_in)):
        cpp(f"\tstream_t(axi_stream_t) in_{i}[{partition_name_upper}_STREAMS_IN_{i}],")
    for i in range(len(coarse_factor_out)):
        if i == len(coarse_factor_out) - 1:
            cpp(f"\tstream_t(axi_stream_t) out_{i}[{partition_name_upper}_STREAMS_OUT_{i}]\n)")
        else:
            cpp(f"\tstream_t(axi_stream_t) out_{i}[{partition_name_upper}_STREAMS_OUT_{i}],")
    with cpp.block(''):

        for i in range(len(coarse_factor_in)):
            cpp(f"#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=in_{i} register")
        for i in range(len(coarse_factor_out)):
            cpp(f"#pragma HLS INTERFACE mode=axis register_mode=both depth=32 port=out_{i} register")
        cpp("#pragma HLS INTERFACE mode=s_axilite bundle=control port=return", newlines=2)

        for i in range(len(coarse_factor_in)):
            cpp(f"#pragma HLS ARRAY_PARTITION variable=in_{i} complete dim=0")
        for i in range(len(coarse_factor_out)):
            if i == len(coarse_factor_out) - 1:
                cpp(f"#pragma HLS ARRAY_PARTITION variable=out_{i} complete dim=0", newlines=2)
            else:
                cpp(f"#pragma HLS ARRAY_PARTITION variable=out_{i} complete dim=0")

        for i in range(len(coarse_factor_in)):
            cpp(f"stream_t({partition_name_lower}_data_t) in_{i}_stream[{partition_name_upper}_STREAMS_IN_{i}];")
        for i in range(len(coarse_factor_out)):
            if i == len(coarse_factor_out) - 1:
                cpp(f"stream_t({partition_name_lower}_data_t) out_{i}_stream[{partition_name_upper}_STREAMS_OUT_{i}];", newlines=2)
            else:
                cpp(f"stream_t({partition_name_lower}_data_t) out_{i}_stream[{partition_name_upper}_STREAMS_OUT_{i}];")

        for i in range(len(coarse_factor_in)):
            cpp(f"#pragma HLS ARRAY_PARTITION variable=in_{i}_stream complete dim=0")
        for i in range(len(coarse_factor_out)):
            if i == len(coarse_factor_out) - 1:
                cpp(f"#pragma HLS ARRAY_PARTITION variable=out_{i}_stream complete dim=0", newlines=2)
            else:
                cpp(f"#pragma HLS ARRAY_PARTITION variable=out_{i}_stream complete dim=0")

        for i in range(len(coarse_factor_in)):
            cpp(f"const int pixel_loop_in_{i} = {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_IN_{i}, {partition_name_upper}_STREAMS_IN_{i})*{partition_name_upper}_DEPTH_IN_{i}*{partition_name_upper}_HEIGHT_IN_{i}*{partition_name_upper}_WIDTH_IN_{i};")

        for i in range(len(coarse_factor_out)):
            if i == len(coarse_factor_out) - 1:
                cpp(f"const int pixel_loop_out_{i} = {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_OUT_{i}, {partition_name_upper}_STREAMS_OUT_{i})*{partition_name_upper}_DEPTH_OUT_{i}*{partition_name_upper}_HEIGHT_OUT_{i}*{partition_name_upper}_WIDTH_OUT_{i};", newlines=2)
            else:
                cpp(f"const int pixel_loop_out_{i} = {partition_name_upper}_BATCH_SIZE*DIVIDE({partition_name_upper}_CHANNELS_OUT_{i}, {partition_name_upper}_STREAMS_OUT_{i})*{partition_name_upper}_DEPTH_OUT_{i}*{partition_name_upper}_HEIGHT_OUT_{i}*{partition_name_upper}_WIDTH_OUT_{i};")

        cpp("#pragma HLS DATAFLOW", newlines=2)

        for i in range(len(coarse_factor_in)):
            cpp(f"axis_to_stream<pixel_loop_in_{i}, {partition_name_upper}_STREAMS_IN_{i}, {partition_name_lower}_data_t, axi_stream_t>(in_{i}, in_{i}_stream);")

        cpp(f"{partition_name_lower}(")
        for i in range(len(coarse_factor_in)):
            cpp(f"\tin_{i}_stream,")
        for i in range(len(coarse_factor_out)):
            if i == len(coarse_factor_out) - 1:
                cpp(f"\tout_{i}_stream\n\t);")
            else:
                cpp(f"\tout_{i}_stream,")

        for i in range(len(coarse_factor_out)):
            cpp(f"stream_to_axis<pixel_loop_out_{i}, {partition_name_upper}_STREAMS_OUT_{i}, {partition_name_lower}_data_t, axi_stream_t>(out_{i}_stream, out_{i});")

    cpp.close()

def generate_top_level_hpp(partition_name: str, model_name: str, layers_config: dict, input_nodes: list, output_nodes: list):
    batch_size = layers_config[input_nodes[0]]["batch_size"]

    channels_in = []
    depth_in = []
    height_in = []
    width_in = []
    coarse_factor_in = []

    for node in input_nodes:
        channels_in.append(layers_config[node]["channels_in"] if "channels_in" in layers_config[node] else layers_config[node]["features_in"])
        depth_in.append(layers_config[node]["depth_in"] if "depth_in" in layers_config[node] else 1)
        height_in.append(layers_config[node]["height_in"] if "height_in" in layers_config[node] else 1)
        width_in.append(layers_config[node]["width_in"] if "width_in" in layers_config[node] else 1)
        coarse_factor_in.append(layers_config[node]["coarse_factor"] if "coarse_factor" in layers_config[node] else layers_config[node]["coarse_in_factor"])

    channels_out = []
    depth_out = []
    height_out = []
    width_out = []
    coarse_factor_out = []

    for node in output_nodes:
        channels_out.append(layers_config[node]["channels_out"] if "channels_out" in layers_config[node] else layers_config[node]["features_out"])
        depth_out.append(layers_config[node]["depth_out"] if "depth_out" in layers_config[node] else 1)
        height_out.append(layers_config[node]["height_out"] if "height_out" in layers_config[node] else 1)
        width_out.append(layers_config[node]["width_out"] if "width_out" in layers_config[node] else 1)
        coarse_factor_out.append(layers_config[node]["coarse_factor"] if "coarse_factor" in layers_config[node] else layers_config[node]["coarse_out_factor"])

    partition_name_lower = partition_name.lower()
    partition_name_upper = partition_name.upper()

    hpp = CppFile(
        os.path.join(
            os.getcwd(),
            "generated_files",
            model_name,
            partition_name,
            "src", f"{partition_name_lower}_top.hpp",
        ))

    hpp("#pragma once", newlines=2)

    hpp(f'#include "{partition_name_lower}.hpp"', newlines=2)

    hpp(f"void {partition_name_lower}_top(")
    for i in range(len(coarse_factor_in)):
        hpp(f"\tstream_t(axi_stream_t) in_{i}[{partition_name_upper}_STREAMS_IN_{i}],")
    for i in range(len(coarse_factor_out)):
        if i == len(coarse_factor_out) - 1:
            hpp(f"\tstream_t(axi_stream_t) out_{i}[{partition_name_upper}_STREAMS_OUT_{i}]\n);")
        else:
            hpp(f"\tstream_t(axi_stream_t) out_{i}[{partition_name_upper}_STREAMS_OUT_{i}],")

    hpp.close()

def generate_top_level_files(partition_name: str, model_name: str, branch_depth: dict, partition_structure: dict, layers_config: dict):
    if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "src")):
        os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "src"))

    header_files = [
        h_file.replace("globalaveragepool", "gap") for h_file in os.listdir(
            os.path.join(os.getcwd(), "generated_files", model_name, partition_name, "src"))
        if h_file.endswith(".hpp")
    ]

    mem_input_nodes = partition_structure['input_nodes']
    mem_output_nodes = partition_structure['output_nodes']
    input_nodes = []
    for node in mem_input_nodes:
        assert len(partition_structure['layers'][node]['out_nodes']) == 1, "Memory input node should have only one output node"
        input_nodes.append(partition_structure['layers'][node]['out_nodes'][0])
    output_nodes = []
    for node in mem_output_nodes:
        in_nodes = partition_structure['layers'][node]['in_nodes']
        assert len(in_nodes) == 1, "Memory output node should have only one input node"
        if partition_structure['layers'][in_nodes[0]]['type'] == "Split":
            output_nodes.append(partition_structure['layers'][in_nodes[0]]['ref_layer'])
        elif partition_structure['layers'][in_nodes[0]]['type'] == "Squeeze":
            output_nodes.append(partition_structure['layers'][in_nodes[0]]['ref_layer_out'])
        else:
            output_nodes.append(in_nodes[0])

    generate_partition_level_hpp(
        partition_name,
        model_name,
        layers_config,
        header_files,
        input_nodes,
        output_nodes,
    )

    generate_partition_level_cpp(
        partition_name,
        model_name,
        layers_config,
        partition_structure,
        branch_depth,
        input_nodes,
        output_nodes,
    )

    generate_top_level_hpp(
        partition_name,
        model_name,
        layers_config,
        input_nodes,
        output_nodes,
    )

    generate_top_level_cpp(
        partition_name,
        model_name,
        layers_config,
        input_nodes,
        output_nodes,
    )
