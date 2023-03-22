import math
from copy import deepcopy

import networkx as nx
import numpy as np

from fpga_hart import _logger
from fpga_hart.layers.activation_3d import Activation3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise_3d import ElementWise3DLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap_3d import GAP3DLayer
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.utils.graph_manipulation import (add_off_chip_connections,
                                                get_input_nodes,
                                                get_output_nodes,
                                                get_split_points)


def get_minimum_resource_utilization(hw_layer, gap_approx=False):
    if isinstance(hw_layer, Convolutional3DLayer):
        initial_filters = deepcopy(hw_layer.filters)
        coarsein_min = 1 / np.int32(hw_layer.channels)
        coarseout_min = 1 / np.int32(hw_layer.filters)
        fine_min = 1 / np.prod(np.array(hw_layer.kernel_shape))
        dsp_util, bram_util, pipeline_depth = hw_layer.get_resource_util(f_fine = fine_min,
                                        f_coarseIn = coarsein_min,
                                        f_coarseOut= coarseout_min)
    elif isinstance(hw_layer, Pooling3DLayer):
        initial_filters = deepcopy(hw_layer.channels)
        coarseinout_min = 1 / np.int32(hw_layer.channels)
        fine_min = 1 / np.prod(np.array(hw_layer.kernel_shape))
        dsp_util, bram_util, pipeline_depth = hw_layer.get_resource_util(f_fine = fine_min,
                                        f_coarse_inout = coarseinout_min)
    elif isinstance(hw_layer, Activation3DLayer):
        initial_filters = deepcopy(hw_layer.filters)
        coarseinout_min = 1 / np.int32(hw_layer.channels)
        dsp_util, bram_util, pipeline_depth = hw_layer.get_resource_util(f_coarse_inout = coarseinout_min,
                                                         supported_ops = [hw_layer.op_type])
    elif isinstance(hw_layer, ElementWise3DLayer):
        initial_filters = deepcopy(hw_layer.filters)
        coarseinout_min = 1 / np.int32(hw_layer.channels_1)
        dsp_util, bram_util, pipeline_depth = hw_layer.get_resource_util(f_coarse_inout = coarseinout_min,
                                                         supported_ops = [hw_layer.op_type])
    elif isinstance(hw_layer, FCLayer):
        initial_filters = deepcopy(hw_layer.dim_out)
        coarsein_min = 1 / np.int32(hw_layer.dim_in)
        coarseout_min = 1 / np.int32(hw_layer.dim_out)
        dsp_util, bram_util, pipeline_depth = hw_layer.get_resource_util(f_coarseIn = coarsein_min,
                                                         f_coarseOut= coarseout_min)
    elif isinstance(hw_layer, GAP3DLayer):
        initial_filters = deepcopy(hw_layer.filters)
        coarseinout_min = 1 / np.int32(hw_layer.channels)
        dsp_util, bram_util, pipeline_depth = hw_layer.get_resource_util(f_coarse_inout = coarseinout_min,
                                                         supported_ops = [],
                                                         gap_approx = gap_approx)
    else:
        raise ValueError(f"Layer type {type(hw_layer)} not supported.")

    # if not (isinstance(hw_layer, Convolutional3DLayer) or isinstance(hw_layer, Pooling3DLayer)):
    #     return 0, dsp_util, initial_filters
    return bram_util, dsp_util, pipeline_depth, initial_filters

def get_extra_mem_connections(graph, node_list):
    extra_inputs, extra_outputs = [], []
    for node in node_list:
        if graph.out_degree[node] > 1:
            for out in graph.successors(node):
                if out not in node_list:
                    extra_outputs.append(node)
                    break
        if graph.in_degree[node] > 1:
            for inp in graph.predecessors(node):
                if inp not in node_list:
                    extra_inputs.append(node)
    return extra_inputs, extra_outputs

def update_nodes_shapes(graph, wr_f, old_filters, old_layer):
    new_filters = math.floor(old_filters/wr_f)
    update_valid = False

    for layer in nx.topological_sort(graph):
        if layer == old_layer:
            update_valid = True
        if update_valid:
            hw = graph.nodes[layer]["hw"]
            new_shape_out = deepcopy(hw.output_shape)
            new_shape_out[1] = new_filters
            assert (new_filters*wr_f + old_filters%wr_f) == old_filters, f"All nodes in a graph with weights reloading should have the same number of filters. {new_filters*wr_f} + {old_filters%wr_f} != {old_filters} on layer {layer}"
            if isinstance(hw, ElementWise3DLayer):
                new_shape_in_1 = deepcopy(hw.input_shape_1)
                new_shape_in_1[1] = new_shape_out[1]
                new_shape_in_2 = deepcopy(hw.input_shape_2)
                new_shape_in_2[1] = new_shape_out[1]
                hw.update_shapes(new_shape_in_1, new_shape_in_2, new_shape_out)
            elif isinstance(hw, Convolutional3DLayer):
                hw.update_shapes(hw.input_shape, new_shape_out)
            else:
                new_shape_in = deepcopy(hw.input_shape)
                new_shape_in[1] = new_shape_out[1]
                hw.update_shapes(new_shape_in, new_shape_out)

def calculate_wr_factor(graph, max_BRAM_util):
    weights_reloading = 1

    wr_layers = []
    for layer in reversed(list(nx.topological_sort(graph))):
        wr_layers.append(layer)
        if graph.nodes[layer]["type"] == "Conv":
            break
    wr_layers = wr_layers[::-1]

    total_bram_util = 0
    for layer in nx.topological_sort(graph):
        if layer == wr_layers[0]:
            break
        hw = graph.nodes[layer]["hw"]
        bram_util, _, _, _ = get_minimum_resource_utilization(hw)
        total_bram_util += bram_util

    for layer in wr_layers:
        hw = graph.nodes[layer]["hw"]
        bram_util, _, _, _ = get_minimum_resource_utilization(hw)
        if (total_bram_util + bram_util) > max_BRAM_util:
            initial_filters = deepcopy(hw.filters)
            for f in range(1,initial_filters):
            # for f in utils.get_factors(initial_filters)[1:]:
                update_nodes_shapes(graph=graph, wr_f=f, old_filters=initial_filters, old_layer=layer)
                bram_util_wr, _, _, _ = get_minimum_resource_utilization(hw)
                if (total_bram_util + bram_util_wr) < max_BRAM_util:
                    weights_reloading = f if initial_filters%f == 0 else (f+1)
                    break
            if weights_reloading == 1:
                _logger.error(f"Layer {layer} does not fit in the device even after weights reloading")
                return -1
        else:
            total_bram_util += bram_util
    return weights_reloading

def check_partition_fitting(graph, partition_composer, max_BRAM_util, word_bytes, bram_type, brams_total, mem_words_per_cycle, result=[], original_graph=None, gap_approx=False):
    if original_graph is None:
        original_graph = deepcopy(graph)
    sort_order = list(nx.topological_sort(original_graph))

    _, min_bram_util = get_worst_case_buffering(deepcopy(graph), partition_composer, mem_words_per_cycle, word_bytes, bram_type, brams_total, gap_approx)
    if min_bram_util > max_BRAM_util:
        for sp in reversed(get_split_points(graph)):
            ancestors = list(nx.ancestors(graph, sp))
            ancestors.sort(key=lambda val: sort_order.index(val))
            subgraph_nodes = deepcopy(ancestors)
            subgraph_nodes.append(sp)

            #TODO: I dont like this, but it works. It should be improved since it is a workaround
            if len(subgraph_nodes) == 1 and not "Conv" in subgraph_nodes[0]:
                descendants = list(nx.descendants(graph, sp))
                descendants.sort(key=lambda val: sort_order.index(val))
                convs_added = 0
                for d in descendants:
                    if "Conv" in d and graph.in_degree[d] <= 1:
                        if convs_added >= 1:
                            break
                        subgraph_nodes.append(d)
                        convs_added += 1
                    else:
                        subgraph_nodes.append(d)

            graph_1 = graph.subgraph(subgraph_nodes).copy()
            graph_2 = graph.copy()
            graph_2.remove_nodes_from(graph_1.nodes())
            _, min_bram_util = get_worst_case_buffering(deepcopy(graph_2), partition_composer, mem_words_per_cycle, word_bytes, bram_type, brams_total, gap_approx)

            if min_bram_util <= max_BRAM_util:
                extra_inputs, extra_outputs = get_extra_mem_connections(original_graph, subgraph_nodes)
                weights_reloading = calculate_wr_factor(graph_1, max_BRAM_util)
                result.append([graph_1, extra_inputs, extra_outputs, weights_reloading])

                if len(graph_2.nodes()) > 0:
                    check_partition_fitting(graph_2, partition_composer, max_BRAM_util, word_bytes, bram_type, brams_total, mem_words_per_cycle, result, original_graph, gap_approx=gap_approx)
                break
        if min_bram_util > max_BRAM_util:
            raise ValueError(f"Graph {graph.nodes()} does not fit in the device even after partitioning based on branch buffering")
    else:
        for layer in nx.topological_sort(graph):
            hw = graph.nodes[layer]["hw"]
            bram_util, _, _, _ = get_minimum_resource_utilization(hw)

            min_bram_util += bram_util
            if min_bram_util > max_BRAM_util:

                ancestors = list(nx.ancestors(graph, layer))
                ancestors.sort(key=lambda val: sort_order.index(val))
                descendants = list(nx.descendants(graph, layer))
                descendants.sort(key=lambda val: sort_order.index(val))

                extra_inputs = []
                extra_outputs = []
                if ancestors:
                    subgraph_nodes = deepcopy(ancestors)

                    append_current = True
                    for ancestor in subgraph_nodes:
                        if "GlobalAveragePool" in ancestor or "Conv" in ancestor:
                            append_current = False
                            break

                    if append_current:
                        subgraph_nodes.append(layer)

                        for descendant in descendants:
                            if "GlobalAveragePool" in descendant or "Conv" in descendant:
                                break
                            if "Add" in descendant or "Mul" in descendant:
                                split_points = get_split_points(graph)
                                if not split_points:
                                    subgraph_nodes.append(descendant)
                                else:
                                    #TODO: Check to which split point the descendant is connected and check only that
                                    if len(split_points) > 1:
                                        raise ValueError("More than one split point found. Not supported yet.")
                                    if not list(nx.all_simple_paths(graph, split_points[0], descendant)):
                                        subgraph_nodes.append(descendant)
                                        break

                    extra_inputs, extra_outputs = get_extra_mem_connections(original_graph, subgraph_nodes)
                    graph_1 = graph.subgraph(subgraph_nodes).copy()
                    weights_reloading = calculate_wr_factor(graph_1, max_BRAM_util)
                    result.append([graph_1, extra_inputs, extra_outputs, weights_reloading])
                else:
                    subgraph_nodes = [layer]

                    for descendant in descendants:
                        if "GlobalAveragePool" in descendant or "Conv" in descendant:
                            break
                        subgraph_nodes.append(descendant)

                    extra_inputs, extra_outputs = get_extra_mem_connections(original_graph, subgraph_nodes)
                    graph_1 = graph.subgraph(subgraph_nodes).copy()
                    weights_reloading = calculate_wr_factor(graph_1, max_BRAM_util)
                    result.append([graph_1, extra_inputs, extra_outputs, weights_reloading])

                graph_2 = graph.copy()
                graph_2.remove_nodes_from(graph_1.nodes())
                if len(graph_2.nodes()) > 0:
                    check_partition_fitting(graph_2, partition_composer, max_BRAM_util, word_bytes, bram_type, brams_total, mem_words_per_cycle, result, original_graph, gap_approx=gap_approx)
                break
        if min_bram_util <= max_BRAM_util:
            extra_inputs, extra_outputs = get_extra_mem_connections(original_graph, graph.nodes())
            weights_reloading = calculate_wr_factor(graph, max_BRAM_util)
            result.append([graph, extra_inputs, extra_outputs, weights_reloading])

    return result

def get_off_chip_mem_connections(graph):
    input_nodes = get_input_nodes(graph)
    output_nodes = get_output_nodes(graph)

    mem_conns = dict({"inputs": [], "outputs": []})
    for n in nx.topological_sort(graph):
        if n in input_nodes:
            mem_conns["inputs"].append(n)
            if graph.nodes[n]["layer_mode"] == "merge" and graph.in_degree(n) == 0:
                mem_conns["inputs"].append(n)
        if n in output_nodes:
            mem_conns["outputs"].append(n)
        if graph.nodes[n]["layer_mode"] == "split" and graph.out_degree(n) <= 1 and n not in output_nodes:
            mem_conns["outputs"].append(n)
    return mem_conns["inputs"], mem_conns["outputs"]

def get_worst_case_buffering(graph, partition_composer, mem_words_per_cycle, word_bytes, bram_type, brams_total, gap_approx):
    # branch_edges = get_branch_start_end_points(graph)

    # branch_buffer = 0
    # for (splt, mrg) in branch_edges:
    #     all_paths = [p for p in nx.all_simple_paths(graph, splt, mrg)]
    #     num_sub_branches = len(all_paths) - 2

    #     shortest_path = nx.shortest_path(graph, splt, mrg)
    #     merge_node = shortest_path[-1]
    #     pre_merge_node = shortest_path[-2]
    #     assert (graph.nodes[pre_merge_node]["hw"].output_shape
    #                 == graph.nodes[merge_node]["hw"].input_shape
    #             ), "Layers input and output shapes does not match"
    #     #TODO: This is the work case scenario for buffering the whole feature map. A more accurate design would be to calculate the depths for each layer in each branch and accumulate the depths to get the total buffer depth which will be the minimum between the work case scenario and the actual buffer depth.
    #     branch_buffer += np.prod(np.array(graph.nodes[pre_merge_node]["hw"].output_shape))
    # branch_buffer *= 0.5

    comb_config = {}
    for node in nx.topological_sort(graph):
        op_type = graph.nodes[node]['type']
        if op_type == "GlobalAveragePool":
            channels = graph.nodes[node]["hw"].input_shape[1]
            comb_config[node] = [1/channels]
        elif op_type == "Conv":
            channels = graph.nodes[node]["hw"].input_shape[1]
            filters = graph.nodes[node]["hw"].output_shape[1]
            kernel_size = np.prod(np.array(graph.nodes[node]["hw"].kernel_shape))
            comb_config[node] = [1/kernel_size, 1/channels, 1/filters]
        elif op_type == "Pooling":
            channels = graph.nodes[node]["hw"].input_shape[1]
            kernel_size = np.prod(np.array(graph.nodes[node]["hw"].kernel_shape))
            comb_config[node] = [1/kernel_size, 1/channels]
        elif op_type == "Activation":
            channels = graph.nodes[node]["hw"].input_shape[1]
            comb_config[node] = [1/channels]
        elif op_type == "ElementWise":
            channels = graph.nodes[node]["hw"].input_shape[1]
            comb_config[node] = [1/channels]
        elif op_type == "BatchNormalization":
            channels = graph.nodes[node]["hw"].input_shape[1]
            comb_config[node] = [1/channels]
        elif op_type == "Gemm":
            channels = graph.nodes[node]["hw"].input_shape[1]
            filters = graph.nodes[node]["hw"].output_shape[1]
            comb_config[node] = [1/channels, 1/filters]
        else:
            assert False, "Not supported layer"
        comb_config[node]

    # nodes_in = get_input_nodes(graph)
    # nodes_out = get_output_nodes(graph)
    nodes_in, nodes_out = get_off_chip_mem_connections(graph)
    num_mem_connections = len(nodes_in) + len(nodes_out)

    # TODO: This might not be totally correct. We should split the memory bandwidth between input and output in half and then split either input or output in half again if there are extra connections.
    mem_bw_in = [mem_words_per_cycle/num_mem_connections for _ in range(len(nodes_in))]
    mem_bw_out = [mem_words_per_cycle/num_mem_connections for _ in range(len(nodes_out))]
    read_points, write_points = add_off_chip_connections(
                graph, nodes_in, nodes_out, gap_approx)

    dp_info = partition_composer.get_design_point(
        graph.copy(),
        comb_config,
        mem_bw_in,
        mem_bw_out,
        read_points,
        write_points,
        gap_approx=gap_approx
    )

    branch_buffer_new = 0
    for k, v in partition_composer.preliminary_branch_depth.items():
        branch_buffer_new += v['depth']
    bram_util = ((branch_buffer_new * word_bytes / (bram_type * 1024) ) / brams_total) * 100

    if not dp_info['config'] and bram_util <= 90.0:
        return 0, 0

    return branch_buffer_new, bram_util