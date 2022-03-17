from layers.codegen import *
from fpgaHART.utils.utils import get_input_node, get_output_node, get_branch_edges, add_supportive_nodes_config
import numpy as np
import os
import re

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
                in_fifo_name = pn.lower().replace("_", "") + "_" + start_node_name
                sorted_inputs.append((in_fifo_name.replace("globalaveragepool", "gap"), config[pn]['shape_in']))
            sorted_inputs = sorted(sorted_inputs, key=lambda x: np.prod(np.array(x[1])), reverse=True)
            for si in sorted_inputs:
                in_fifo_names.append(si[0])
        else:
            in_fifo_names.append("in")

        next_nodes = list(graph.successors(start_node))
        out_fifo_names = []
        if next_nodes:
            for nn in next_nodes:
                out_fifo_name = start_node_name + "_" + nn.lower().replace("_", "")
                out_fifo_names.append(out_fifo_name.replace("globalaveragepool", "gap"))
        else:
            out_fifo_names.append("out")
        
        nodes_with_fifos[start_node] = {"in_fifos": in_fifo_names, "out_fifos": out_fifo_names}

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
                    sqn_decompose = f"{sqn_decompose[0]}_{sqn_decompose[1]}_{sqn_decompose[2]}"
                else:
                    sqn_decompose = sqn
                if node in sqn_decompose:
                    nodes_in_order.append(sqn)

    return nodes_with_fifos, nodes_in_order, unique_fifos

def generate_top_level_cpp(graph, layers_config, branch_depth, partition_name, prefix):
    nodes_with_fifos, nodes_in_order, unique_fifos = get_nodes_and_fifos(graph, layers_config)
    branch_edges = get_branch_edges(graph)

    partition_name_lower = partition_name.lower()
    partition_name_upper = partition_name.upper()

    cpp = CppFile(os.path.join(os.getcwd(), "generated_files", f"{prefix}/{partition_name}", f"{partition_name_lower}.cpp"))

    cpp(f"#include \"{partition_name_lower}.hpp\"", newlines=2)

    with cpp.block(f"void {partition_name_lower}_top(\n\
        stream_t({partition_name_lower}_input_t) in[{partition_name_upper}_STREAMS_IN],\n\
        stream_t({partition_name_lower}_output_t) out[{partition_name_upper}_STREAMS_OUT])"):

        cpp("#pragma HLS INTERFACE ap_ctrl_chain port=return")
        cpp("#pragma HLS INTERFACE axis register_mode=both depth=2 port=in register")
        cpp("#pragma HLS INTERFACE axis register_mode=both depth=2 port=out register", newlines=2)

        cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

        cpp("#pragma HLS DATAFLOW", newlines=2)

        for fifo in unique_fifos:
            fifo_out_lower = convert_name(fifo.split("_")[1]).lower()
            fifo_out_upper = fifo_out_lower.upper()
            data_type = f"{fifo_out_lower}_data_t"
            coarse = f"{fifo_out_upper}_COARSE" if (not ("conv" in fifo_out_lower or "squeeze" in fifo_out_lower) or "split" in fifo_out_lower) else f"{fifo_out_upper}_COARSE_IN"

            cpp(f"stream_t({data_type}) {fifo}[{coarse}];")
            cpp(f"#pragma HLS STREAM variable={fifo}")
            cpp(f"#pragma HLS ARRAY_PARTITION variable={fifo} complete dim=0", newlines=2)

        for node in nodes_in_order:
            node_name = node.lower() + "_layer"
            node_name = node_name.replace("globalaveragepool", "gap")
            in_fifos = ",".join(nodes_with_fifos[node]["in_fifos"])
            out_fifos = ",".join(nodes_with_fifos[node]["out_fifos"])
            
            cpp(f"{node_name}({in_fifos},{out_fifos});", newlines=2)

    cpp.close()

def generate_top_level_hpp(include_files, in_name, in_config, out_name, out_config, partition_name, prefix):
    batch_size = in_config['shape_in'][0]

    channels_in = in_config['shape_in'][1]
    depth_in = in_config['shape_in'][2]
    height_in = in_config['shape_in'][3]
    width_in = in_config['shape_in'][4]
    coarse_factor_in = in_config['coarse_factor'] if 'coarse_factor' in in_config else in_config['coarse_in_factor']

    channels_out = out_config['shape_out'][1]
    depth_out = out_config['shape_out'][2]
    height_out = out_config['shape_out'][3]
    width_out = out_config['shape_out'][4]
    coarse_factor_out = out_config['coarse_factor'] if 'coarse_factor' in out_config else out_config['coarse_out_factor']

    partition_name_lower = partition_name.lower()
    partition_name_upper = partition_name.upper()

    layer_in_name_lower = in_name.lower()
    layer_out_name_lower = out_name.lower()

    hpp = CppFile(os.path.join(os.getcwd(), "generated_files", f"{prefix}/{partition_name}", f"{partition_name_lower}.hpp"))
    
    hpp("#pragma once", newlines=2)
    hpp("#include \"common_.hpp\"")
    for include_file in include_files:
        hpp(f"#include \"{include_file}\"")
    hpp("\n")

    hpp(f"#define {partition_name_upper}_BATCH_SIZE {batch_size}", newlines=2)

    hpp(f"#define {partition_name_upper}_CHANNELS_IN {channels_in}")
    hpp(f"#define {partition_name_upper}_DEPTH_IN {depth_in}")
    hpp(f"#define {partition_name_upper}_HEIGHT_IN {height_in}")
    hpp(f"#define {partition_name_upper}_WIDTH_IN {width_in}", newlines=2)

    hpp(f"#define {partition_name_upper}_CHANNELS_OUT {channels_out}")
    hpp(f"#define {partition_name_upper}_DEPTH_OUT {depth_out}")
    hpp(f"#define {partition_name_upper}_HEIGHT_OUT {height_out}")
    hpp(f"#define {partition_name_upper}_WIDTH_OUT {width_out}", newlines=2)

    hpp(f"#define {partition_name_upper}_STREAMS_IN {coarse_factor_in}")
    hpp(f"#define {partition_name_upper}_STREAMS_OUT {coarse_factor_out}", newlines=2)

    hpp(f"typedef {layer_in_name_lower}_data_t \t{partition_name_lower}_input_t;")
    hpp(f"typedef {layer_out_name_lower}_data_t \t{partition_name_lower}_output_t;", newlines=3)

    hpp(f"void {partition_name_lower}_top(\n\
        stream_t({partition_name_lower}_input_t) in[{partition_name_upper}_STREAMS_IN],\n\
        stream_t({partition_name_lower}_output_t) out[{partition_name_upper}_STREAMS_OUT]);")
    
    hpp.close()

def generate_top_level_files(graph, branch_depth, layers_config, partition_name, prefix):
    if not os.path.exists(os.path.join(os.getcwd(), "generated_files", f"{prefix}/{partition_name}")):
        os.makedirs(os.path.join(os.getcwd(), "generated_files", f"{prefix}/{partition_name}"))

    header_files = [h_file.replace("globalaveragepool", "gap") for h_file in os.listdir(os.path.join(os.getcwd(), "generated_files", f"{prefix}/{partition_name}")) if h_file.endswith(".hpp")]
    input_node = get_input_node(graph)
    output_node = get_output_node(graph)

    generate_top_level_hpp(header_files, input_node, layers_config[input_node], output_node, layers_config[output_node], partition_name, prefix)
    generate_top_level_cpp(graph, add_supportive_nodes_config(graph, layers_config.copy()), branch_depth, partition_name, prefix)