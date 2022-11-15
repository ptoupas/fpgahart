import os
import re

import numpy as np
from layers.codegen import *

from fpga_hart.utils.utils import (generate_supportive_layer_config,
                                   get_branch_edges, get_input_node,
                                   get_output_node)


def convert_name(name):
    numbers = re.findall(r"\d+", name)
    for n in numbers:
        if name.endswith(n):
            name = name.replace(n, f"_{n}")
        else:
            name = name.replace(n, f"_{n}_")
    if "split" in name:
        name = name.replace("split", "split_")
    if "squeeze" in name:
        name = name.replace("squeeze", "squeeze_")
    return name


def get_nodes_and_fifos(graph, config):
    nodes_with_fifos = {}
    for start_node in graph.nodes:
        start_node_name = start_node.lower().replace("_", "")

        prev_nodes = list(graph.predecessors(start_node))
        in_fifo_names = []
        if prev_nodes:
            sorted_inputs = []
            for pn in prev_nodes:
                in_fifo_name = pn.lower().replace("_",
                                                  "") + "_" + start_node_name
                sorted_inputs.append((
                    in_fifo_name.replace("globalaveragepool", "gap"),
                    config[pn]["shape_in"],
                ))
            sorted_inputs = sorted(sorted_inputs,
                                   key=lambda x: np.prod(np.array(x[1])),
                                   reverse=True)
            for si in sorted_inputs:
                in_fifo_names.append(si[0])
        else:
            in_fifo_names.append("in")

        next_nodes = list(graph.successors(start_node))
        out_fifo_names = []
        if next_nodes:
            for nn in next_nodes:
                out_fifo_name = start_node_name + "_" + nn.lower().replace(
                    "_", "")
                out_fifo_names.append(
                    out_fifo_name.replace("globalaveragepool", "gap"))
        else:
            out_fifo_names.append("out")

        nodes_with_fifos[start_node] = {
            "in_fifos": in_fifo_names,
            "out_fifos": out_fifo_names,
        }

    unique_fifos = []
    for node in [*nodes_with_fifos]:
        in_fifos = nodes_with_fifos[node]["in_fifos"]
        out_fifos = nodes_with_fifos[node]["out_fifos"]
        for fifo in in_fifos:
            if fifo not in unique_fifos and fifo != "in":
                unique_fifos.append(fifo.replace("globalaveragepool", "gap"))
        for fifo in out_fifos:
            if fifo not in unique_fifos and fifo != "out":
                unique_fifos.append(fifo.replace("globalaveragepool", "gap"))

    split_squeeze_nodes = []
    for node in [*nodes_with_fifos]:
        if "Split" in node or "Squeeze" in node:
            split_squeeze_nodes.append(node)

    nodes_in_order = []
    for node in [*nodes_with_fifos]:
        if not ("Split" in node or "Squeeze" in node):
            nodes_in_order.append(node)
            for sqn in split_squeeze_nodes:
                if "Squeeze" in sqn:
                    sqn_decompose = sqn.split("_")
                    sqn_decompose = (
                        f"{sqn_decompose[0]}_{sqn_decompose[1]}_{sqn_decompose[2]}"
                    )
                else:
                    sqn_decompose = sqn
                if node in sqn_decompose:
                    nodes_in_order.append(sqn)

    return nodes_with_fifos, nodes_in_order, unique_fifos


def generate_top_level_cpp(partition_name: str, model_name: str, layers_config: dict, partition_structure: dict, branch_depth: dict, input_nodes: list, output_nodes: list):

    coarse_factor_in = []
    for node in input_nodes:
        coarse_factor_in.append(layers_config[node]["coarse_factor"] if "coarse_factor" in layers_config[node] else layers_config[node]["coarse_in_factor"])

    coarse_factor_out = []
    for node in output_nodes:
        coarse_factor_out.append(layers_config[node]["coarse_factor"] if "coarse_factor" in layers_config[node] else layers_config[node]["coarse_out_factor"])

    # nodes_with_fifos, nodes_in_order, unique_fifos = get_nodes_and_fifos(
    #     graph, layers_config)
    # branch_edges = get_branch_edges(graph)

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

        cpp("#pragma HLS DATAFLOW", newlines=2)

        # for fifo in unique_fifos:
        #     fifo_out_lower = convert_name(fifo.split("_")[1]).lower()
        #     fifo_out_upper = fifo_out_lower.upper()
        #     data_type = f"{fifo_out_lower}_data_t"
        #     coarse = (f"{fifo_out_upper}_COARSE" if (
        #         not ("conv" in fifo_out_lower or "squeeze" in fifo_out_lower)
        #         or "split" in fifo_out_lower) else
        #               f"{fifo_out_upper}_COARSE_IN")

        #     cpp(f"stream_t({data_type}) {fifo}[{coarse}];")
        #     cpp(f"#pragma HLS STREAM variable={fifo}")
        #     cpp(
        #         f"#pragma HLS ARRAY_PARTITION variable={fifo} complete dim=0",
        #         newlines=2,
        #     )

        # for node in nodes_in_order:
        #     node_name = node.lower() + "_layer"
        #     node_name = node_name.replace("globalaveragepool", "gap")
        #     in_fifos = ",".join(nodes_with_fifos[node]["in_fifos"])
        #     out_fifos = ",".join(nodes_with_fifos[node]["out_fifos"])

        #     cpp(f"{node_name}({in_fifos},{out_fifos});", newlines=2)

    cpp.close()


def generate_top_level_hpp(partition_name: str, model_name: str, layers_config: dict, header_files: list, input_nodes: list, output_nodes: list):
    batch_size = layers_config[input_nodes[0]]["batch_size"]

    channels_in = []
    depth_in = []
    height_in = []
    width_in = []
    coarse_factor_in = []

    for node in input_nodes:
        channels_in.append(layers_config[node]["channels_in"])
        depth_in.append(layers_config[node]["depth_in"])
        height_in.append(layers_config[node]["height_in"])
        width_in.append(layers_config[node]["width_in"])
        coarse_factor_in.append(layers_config[node]["coarse_factor"] if "coarse_factor" in layers_config[node] else layers_config[node]["coarse_in_factor"])

    channels_out = []
    depth_out = []
    height_out = []
    width_out = []
    coarse_factor_out = []

    for node in output_nodes:
        channels_out.append(layers_config[node]["channels_out"])
        depth_out.append(layers_config[node]["depth_out"])
        height_out.append(layers_config[node]["height_out"])
        width_out.append(layers_config[node]["width_out"])
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


    generate_top_level_hpp(
        partition_name,
        model_name,
        layers_config,
        header_files,
        input_nodes,
        output_nodes,
    )

    generate_top_level_cpp(
        partition_name,
        model_name,
        layers_config,
        partition_structure,
        branch_depth,
        input_nodes,
        output_nodes,
    )
