import math
import random
from collections import deque
from typing import Tuple

import networkx as nx
import numpy as np

import wandb
from fpga_hart.layers.gap_3d import GAP3DLayer
from fpga_hart.layers.memory_interface import MemoryNode


def has_gap(graph: nx.DiGraph) -> bool:
    result = False
    for node in graph.nodes:
        hw = graph.nodes[node]["hw"]
        if isinstance(hw, GAP3DLayer):
            result = True
            break
    return result

def split_graph(
    graph: nx.DiGraph,
    word_bytes: float,
    bram_Kbytes: float,
    bram: int,
    max_BRAM_util: float,
) -> Tuple[nx.DiGraph, nx.DiGraph, int, int, list, list]:
    """
    Create a 1st break point of the graph right before the mul operation of squeeze excitation module.
    Then search (using simulated annealing) from which point in the graph we will start the 2nd phase of the execution (or we will reconfigure the whole FPGA). Depending on the point we might need to store extra intermediate results of phase1 and read them from off-chip memory during phase 2. The other alternative will be to re-compute some layers but reduce the off-chip memory transfers.
    """
    mem_in_1, mem_out_1, mem_in_2, mem_out_2 = [], [], [], []

    merge_nodes = [n for n in graph.nodes if graph.in_degree[n] > 1]
    split_nodes = [n for n in graph.nodes if graph.out_degree[n] > 1]
    break_node_gap = ""
    for node in graph.nodes:
        op_type = graph.nodes[node]["type"]
        hw = graph.nodes[node]["hw"]
        if (
            op_type == "ElementWise"
            and graph.in_degree[node] > 1
            and hw.op_type == "Mul"
        ):
            break_node_gap = node

    phase_1 = deque()
    check_node = break_node_gap
    predec_nodes = [n for n in graph.predecessors(check_node)]
    while predec_nodes:
        if len(predec_nodes) > 1:
            for pn in predec_nodes:
                op_type = graph.nodes[pn]["type"]
                if op_type == "Activation":
                    phase_1.appendleft(pn)
                    check_node = pn
                    predec_nodes = [n for n in graph.predecessors(check_node)]
                    break
        else:
            curr_node = predec_nodes[0]
            phase_1.appendleft(curr_node)
            check_node = curr_node
            predec_nodes = [n for n in graph.predecessors(check_node)]

    phase_1_graph_frozen = graph.subgraph(list(phase_1))
    phase_1_graph = nx.DiGraph(phase_1_graph_frozen)
    phase_1_edges = list(phase_1_graph.edges)

    for node in list(phase_1_graph.nodes()):
        if node in merge_nodes and not phase_1_graph.in_degree[node] > 1:
            mem_in_1.append(node)
        if phase_1_graph.in_degree[node] == 0:
            mem_in_1.append(node)
        if phase_1_graph.out_degree[node] == 0:
            mem_out_1.append(node)

    phase_2_graph = graph.copy()
    gap_index = [i for i, n in enumerate(list(phase_1)) if "Global" in n][0]
    phase_2_graph.remove_nodes_from(list(phase_1)[gap_index:])
    phase_2_graph.remove_edges_from(phase_1_edges[gap_index - 1 :])

    phase_2_read_point = random.choice([*[-1], *list(range(gap_index))])
    split_graph_start_point = 0
    split_nodes_ind = [i for i, n in enumerate(list(phase_1)) if n in split_nodes]
    for i in split_nodes_ind:
        if i < phase_2_read_point:
            split_graph_start_point = i
            # TODO: What if we have more than 1 point before the phase_2_read_point node?
            break
    if phase_2_read_point >= 0:
        phase_2_read_node = [
            n for i, n in enumerate(list(phase_1)) if phase_2_read_point == i
        ][0]
        mem_out_1.append(phase_2_read_node)

    if split_graph_start_point > 0:
        phase_2_graph.remove_nodes_from(
            list(phase_1)[split_graph_start_point + 1 : phase_2_read_point + 1]
        )
        phase_2_graph.remove_edges_from(
            phase_1_edges[split_graph_start_point + 1 : phase_2_read_point + 1]
        )
    else:
        phase_2_graph.remove_nodes_from(list(phase_1)[: phase_2_read_point + 1])
        phase_2_graph.remove_edges_from(phase_1_edges[: phase_2_read_point + 1])

    for node in list(phase_2_graph.nodes()):
        if node in merge_nodes and not phase_2_graph.in_degree[node] > 1:
            mem_in_2.append(node)
        if (
            phase_2_graph.in_degree[node] == 0
        ):  # and node in [n for n in graph.successors(phase_2_read_node)]:
            mem_in_2.append(node)
        if phase_2_graph.out_degree[node] == 0:
            mem_out_2.append(node)

    branch_edges_1 = get_branch_edges(phase_1_graph)
    branch_edges_2 = get_branch_edges(phase_2_graph)
    # Worst case scenario
    branch_buffer_1 = 0
    for edge in branch_edges_1:
        max_shape = 0
        for pair in edge:
            assert (
                phase_1_graph.nodes[pair[0]]["hw"].output_shape
                == phase_1_graph.nodes[pair[1]]["hw"].input_shape_1
                or phase_1_graph.nodes[pair[0]]["hw"].output_shape
                == phase_1_graph.nodes[pair[1]]["hw"].input_shape_2
            ), "Layers input and output shapes does not match"
            max_shape = max(
                max_shape,
                np.prod(np.array(phase_1_graph.nodes[pair[0]]["hw"].output_shape[1:])),
            )
        branch_buffer_1 += max_shape
    # Worst case scenario
    branch_buffer_2 = 0
    for edge in branch_edges_2:
        max_shape = 0
        for pair in edge:
            assert (
                phase_2_graph.nodes[pair[0]]["hw"].output_shape
                == phase_2_graph.nodes[pair[1]]["hw"].input_shape_1
                or phase_2_graph.nodes[pair[0]]["hw"].output_shape
                == phase_2_graph.nodes[pair[1]]["hw"].input_shape_2
            ), "Layers input and output shapes does not match"
            max_shape = max(
                max_shape,
                np.prod(np.array(phase_2_graph.nodes[pair[0]]["hw"].output_shape[1:])),
            )
        branch_buffer_2 += max_shape

    mem_kb = ((branch_buffer_1 + branch_buffer_2) * word_bytes) / 1e3
    mem_bram = math.ceil(mem_kb / bram_Kbytes)
    branch_bram_util = (mem_bram / bram) * 100
    if branch_bram_util > max_BRAM_util:
        raise ValueError(
            "BRAM utilization is {}%. Buffering cant be used in one of the splitted graphs.".format(
                branch_bram_util
            )
        )
    return (
        phase_1_graph,
        phase_2_graph,
        branch_buffer_1,
        branch_buffer_2,
        mem_in_1,
        mem_out_1,
        mem_in_2,
        mem_out_2,
    )


def add_node_to_position(
    G: nx.DiGraph,
    new_node: str,
    connect_node: str,
    connect_pos: str,
    is_input: bool = False,
    is_output: bool = False,
) -> None:
    if connect_pos == "pre":
        edge = (new_node, connect_node)
    elif connect_pos == "post":
        edge = (connect_node, new_node)
    else:
        raise Exception(f"Invalid connect_pos {connect_pos}")
    old_nodes = G.copy().nodes()
    nodes = list(G.nodes())
    edges = list(G.edges())
    if is_input:
        new_nodes = [new_node] + nodes
        new_edges = [edge] + edges
    elif is_output:
        new_nodes = nodes + [new_node]
        new_edges = edges + [edge]
    else:
        node_idx = nodes.index(connect_node)
        new_nodes = nodes.copy()
        if connect_pos == "pre":
            new_nodes.insert(node_idx, new_node)
        elif connect_pos == "post":
            new_nodes.insert(node_idx + 1, new_node)
        new_edges = edges.copy()
        new_edges.append(edge)
    G.remove_nodes_from(nodes)
    G.remove_edges_from(edges)
    G.add_nodes_from(new_nodes)
    G.add_edges_from(new_edges)

    for node in G.nodes():
        if node in old_nodes:
            G.nodes._nodes[node] = old_nodes[node]
        else:
            if "Mem_in" in node:
                if old_nodes[edge[1]]["type"] == "ElementWise":
                    extra_inputs = [n for n in G.predecessors(edge[1]) if not n == node]
                    for e in extra_inputs:
                        assert (
                            not old_nodes[e]["type"] == "ElementWise"
                        ), "Current graph sequence cannot be handled."
                        shape_e = old_nodes[e]["hw"].output_shape
                        if shape_e == old_nodes[edge[1]]["hw"].input_shape_1:
                            shape = old_nodes[edge[1]]["hw"].input_shape_2
                        elif shape_e == old_nodes[edge[1]]["hw"].input_shape_2:
                            shape = old_nodes[edge[1]]["hw"].input_shape_1
                        else:
                            raise Exception("Invalid shape for Memory in node")
                    if not extra_inputs:
                        shape = old_nodes[edge[1]]["hw"].input_shape_1
                else:
                    shape = old_nodes[edge[1]]["hw"].input_shape
                G.nodes._nodes[node] = {
                    "type": "mem_in",
                    "hw": MemoryNode("in", shape),
                }
            elif "Mem_out" in node:
                shape = old_nodes[edge[0]]["hw"].output_shape
                G.nodes._nodes[node] = {
                    "type": "mem_out",
                    "hw": MemoryNode("out", shape),
                }
    del old_nodes


def add_off_chip_connections(
    graph: nx.DiGraph,
    in_connections: list,
    out_connections: list,
    gap_approx: bool = False,
) -> Tuple[list, list]:
    read_points = []
    write_points = []

    mem_in_count = 1
    mem_out_count = 1
    input_nodes = []
    output_nodes = []
    for n in nx.topological_sort(graph):
        edges_in = graph.in_edges(n)
        edges_out = graph.out_edges(n)
        if not edges_in:
            input_nodes.append(n)
        if not edges_out:
            output_nodes.append(n)

    for in_n in input_nodes:
        repetitions = 1
        # TODO: This only works with elementwise nodes accepting 2 inputs, need to be generalized for N inputs
        if graph.nodes[in_n]["type"] == "ElementWise":
            repetitions = 2
        for _ in range(repetitions):
            read_points.append(in_n)
            if not 'mem_in' in in_n.lower():
                add_node_to_position(
                    G=graph,
                    new_node="Mem_in{}".format(mem_in_count),
                    connect_node=in_n,
                    connect_pos="pre",
                )
            mem_in_count += 1

    for out_n in output_nodes:
        write_points.append(out_n)
        if not 'mem_out' in out_n.lower():
            add_node_to_position(
                G=graph,
                new_node="Mem_out{}".format(mem_out_count),
                connect_node=out_n,
                connect_pos="post",
            )
        mem_out_count += 1

    for con_in in in_connections:
        if con_in in input_nodes:
            continue
        if not 'mem_in' in con_in.lower():
            add_node_to_position(
                G=graph,
                new_node="Mem_in{}".format(mem_in_count),
                connect_node=con_in,
                connect_pos="pre",
            )
        read_points.append(con_in)
        mem_in_count += 1

    for con_out in out_connections:
        if con_out in output_nodes:
            continue
        if not 'mem_out' in con_out.lower():
            add_node_to_position(
                G=graph,
                new_node="Mem_out{}".format(mem_out_count),
                connect_node=con_out,
                connect_pos="post",
            )
        write_points.append(con_out)
        mem_out_count += 1

    return read_points, write_points

def get_graph_structure(graph: nx.DiGraph, config: dict) -> dict:
    graph_structure = {}
    graph_structure["input_nodes"] = get_input_nodes(graph)
    graph_structure["output_nodes"] = get_output_nodes(graph)
    layers_sub_dict = {}
    for n in nx.topological_sort(graph):
        layers_sub_dict[n] = {}
        layers_sub_dict[n]["type"] = graph.nodes[n]["type"]
        layers_sub_dict[n]["in_edges"] = list(graph.in_edges(n))
        layers_sub_dict[n]["out_edges"] = list(graph.out_edges(n))
        layers_sub_dict[n]["in_nodes"] = list(graph.predecessors(n))
        layers_sub_dict[n]["out_nodes"] = list(graph.successors(n))
        layers_sub_dict[n]["split_node"] = True if graph.out_degree(n) > 1 else False
        layers_sub_dict[n]["merge_node"] = True if graph.in_degree(n) > 1 else False
        if "Mem_in" in n or "Mem_out" in n:
            layers_sub_dict[n]["streams_in"] = 1
            layers_sub_dict[n]["streams_out"] = 1
        else:
            if "Conv" in graph.nodes[n]["type"] or "Gemm" in graph.nodes[n]["type"]:
                layers_sub_dict[n]["streams_in"] = config[n]["coarse_in_factor"]
                layers_sub_dict[n]["streams_out"] = config[n]["coarse_out_factor"]
            else:
                layers_sub_dict[n]["streams_in"] = config[n]["coarse_factor"]
                layers_sub_dict[n]["streams_out"] = config[n]["coarse_factor"]
    graph_structure["layers"] = layers_sub_dict

    update_graph_structure_split_layers(graph_structure)
    update_graph_structure_squeeze_layers(graph_structure)

    return graph_structure

def update_graph_structure_split_layers(graph_structure: dict) -> None:
    # create split layers
    split_nodes = {}
    for node, config in graph_structure["layers"].items():
        if config['type'] == 'mem_in' or config['type'] == 'mem_out':
            continue
        if config['split_node']:
            split_node_name = "Split_{}".format(node)
            split_nodes[split_node_name] = {}
            split_nodes[split_node_name]["type"] = 'Split'
            split_nodes[split_node_name]["ref_layer"] = node
            split_nodes[split_node_name]["in_edges"] = [(node, split_node_name)]
            split_nodes[split_node_name]["out_edges"] = [(split_node_name, graph_structure["layers"][node]["out_nodes"][0]), (split_node_name, graph_structure["layers"][node]["out_nodes"][1])]
            split_nodes[split_node_name]["in_nodes"] = [node]
            split_nodes[split_node_name]["out_nodes"] = graph_structure["layers"][node]["out_nodes"]
            split_nodes[split_node_name]["split_node"] = True
            split_nodes[split_node_name]["merge_node"] = False
            split_nodes[split_node_name]["streams_in"] = graph_structure["layers"][node]["streams_in"]
            split_nodes[split_node_name]["streams_out"] = graph_structure["layers"][node]["streams_out"]

            next_nodes = config['out_nodes']
            for n in next_nodes:
                if n in graph_structure["layers"]:
                    graph_structure["layers"][n]['in_edges'][graph_structure["layers"][n]['in_edges'].index((node, n))] = (split_node_name, n)
                    graph_structure["layers"][n]['in_nodes'][graph_structure["layers"][n]['in_nodes'].index(node)] = split_node_name

            graph_structure["layers"][node]['out_edges'] = [(node, split_node_name)]
            graph_structure["layers"][node]['out_nodes'] = [split_node_name]
            graph_structure["layers"][node]['split_node'] = False

    graph_structure["layers"] |= split_nodes

def update_graph_structure_squeeze_layers(graph_structure: dict) -> None:
    # create squeeze layers
    squeeze_nodes = {}
    for node, config in graph_structure["layers"].items():
        if config['type'] == 'mem_in' or config['type'] == 'mem_out':
            continue

        prev_nodes = config['in_nodes']
        next_nodes = config['out_nodes']

        for p in prev_nodes:
            if p not in graph_structure["layers"] or graph_structure["layers"][p]["type"] == 'mem_in':
                continue
            if graph_structure["layers"][p]["streams_out"] != config["streams_in"]:
                squeeze_node_name = "Squeeze_{}_{}".format(p, node)
                squeeze_nodes[squeeze_node_name] = {}
                squeeze_nodes[squeeze_node_name]["type"] = 'Squeeze'
                squeeze_nodes[squeeze_node_name]["ref_layer_in"] = p
                squeeze_nodes[squeeze_node_name]["ref_layer_out"] = node
                squeeze_nodes[squeeze_node_name]["in_edges"] = [(p, squeeze_node_name)]
                squeeze_nodes[squeeze_node_name]["out_edges"] = [(squeeze_node_name, node)]
                squeeze_nodes[squeeze_node_name]["in_nodes"] = [p]
                squeeze_nodes[squeeze_node_name]["out_nodes"] = [node]
                squeeze_nodes[squeeze_node_name]["split_node"] = False
                squeeze_nodes[squeeze_node_name]["merge_node"] = False
                squeeze_nodes[squeeze_node_name]["streams_in"] = graph_structure["layers"][p]["streams_out"]
                squeeze_nodes[squeeze_node_name]["streams_out"] = config["streams_in"]

                graph_structure["layers"][node]['in_edges'][graph_structure["layers"][node]['in_edges'].index((p,node))] = (squeeze_node_name, node)
                graph_structure["layers"][node]['in_nodes'][graph_structure["layers"][node]['in_nodes'].index(p)] = squeeze_node_name
                graph_structure["layers"][p]['out_edges'][graph_structure["layers"][p]['out_edges'].index((p,node))] = (p, squeeze_node_name)
                graph_structure["layers"][p]['out_nodes'][graph_structure["layers"][p]['out_nodes'].index(node)]  = squeeze_node_name

        for n in next_nodes:
            if n not in graph_structure["layers"] or graph_structure["layers"][n]["type"] == 'mem_out':
                continue
            if graph_structure["layers"][n]["streams_in"] != config["streams_out"]:
                squeeze_node_name = "Squeeze_{}_{}".format(node, n)
                squeeze_nodes[squeeze_node_name] = {}
                squeeze_nodes[squeeze_node_name]["type"] = 'Squeeze'
                squeeze_nodes[squeeze_node_name]["ref_layer_in"] = node
                squeeze_nodes[squeeze_node_name]["ref_layer_out"] = n
                squeeze_nodes[squeeze_node_name]["in_edges"] = [(node, squeeze_node_name)]
                squeeze_nodes[squeeze_node_name]["out_edges"] = [(squeeze_node_name, n)]
                squeeze_nodes[squeeze_node_name]["in_nodes"] = [node]
                squeeze_nodes[squeeze_node_name]["out_nodes"] = [n]
                squeeze_nodes[squeeze_node_name]["split_node"] = False
                squeeze_nodes[squeeze_node_name]["merge_node"] = False
                squeeze_nodes[squeeze_node_name]["streams_in"] = config["streams_out"]
                squeeze_nodes[squeeze_node_name]["streams_out"] = graph_structure["layers"][n]["streams_in"]

                graph_structure["layers"][node]['out_edges'][graph_structure["layers"][node]['out_edges'].index((node, n))] = (node, squeeze_node_name)
                graph_structure["layers"][node]['out_nodes'][graph_structure["layers"][node]['out_nodes'].index(n)] = squeeze_node_name
                graph_structure["layers"][n]['in_edges'][graph_structure["layers"][n]['in_edges'].index((node, n))] = (squeeze_node_name, n)
                graph_structure["layers"][n]['in_nodes'][graph_structure["layers"][n]['in_nodes'].index(node)] = squeeze_node_name

    graph_structure["layers"] |= squeeze_nodes

def update_graph(graph, split_points=None, squeeze_layers=None):
    if split_points is None and squeeze_layers is None:
        return graph

    if split_points is not None:
        new_nodes = []
        new_edges = []
        for sp in split_points:
            new_node_name = "Split_" + sp
            new_nodes.append(new_node_name)

            next_nodes = list(graph.successors(sp))

            edges_out = list(graph.out_edges(sp))

            assert (
                len(next_nodes) > 1 and len(edges_out) > 1
            ), "Split point {} cannot have only one successor".format(sp)

            graph.remove_edges_from(edges_out)

            edge = (sp, new_node_name)
            new_edges.append(edge)
            for nd in next_nodes:
                edge = (new_node_name, nd)
                new_edges.append(edge)

        if new_nodes or new_edges:
            graph.update(edges=new_edges, nodes=new_nodes)

    if squeeze_layers is not None:
        new_nodes = []
        new_edges = []
        for sl in squeeze_layers:
            new_node_name = "Squeeze_" + sl[0] + "_" + sl[1]
            new_nodes.append(new_node_name)

            prev_nodes = list(graph.predecessors(sl[1]))

            edges_in = list(graph.in_edges(sl[1]))

            for ei in edges_in:
                if sl[0] in ei[0] and sl[1] in ei[1]:
                    graph.remove_edge(ei[0], ei[1])

            edge = (new_node_name, sl[1])
            new_edges.append(edge)
            for pn in prev_nodes:
                if sl[0] in pn:
                    edge = (pn, new_node_name)
                    new_edges.append(edge)

        if new_nodes or new_edges:
            graph.update(edges=new_edges, nodes=new_nodes)

    return graph

def get_branch_start_end_points(graph):
    result = []
    def traverse_branch_bw(graph, mp, result):
        for pred in graph.predecessors(mp):
            prev_node = pred
            while True:
                if graph.out_degree[prev_node] > 1:
                    if (prev_node, mp) not in result:
                        result.append((prev_node, mp))
                    break

                if graph.in_degree[prev_node] == 1:
                    prev_node = list(graph.predecessors(prev_node))[0]
                elif graph.in_degree[prev_node] > 1:
                    prev_node = traverse_branch_bw(graph, prev_node, result)
                    assert len(list(graph.predecessors(prev_node))) == 1, "Split layer before split layer is not supported"
                    prev_node = list(graph.predecessors(prev_node))[0]
                elif graph.in_degree[prev_node] == 0:
                    if (prev_node, mp) not in result:
                        result.append((prev_node, mp))
                    break
        if "Mem_in" in prev_node and prev_node in list(graph.predecessors(mp)):
            assert len(list(graph.predecessors(mp))) == 2, f"If prev_node is a mem_in node {mp} should have atleast 2 predecessors since it is a merge point"
            predecessors = list(graph.predecessors(mp))
            predecessors.remove(prev_node)
            prev_node = predecessors[0]
        return prev_node

    merge_points = get_merge_points(graph)
    for mp in merge_points:
        traverse_branch_bw(graph, mp, result)

    return result

    # split_points = get_split_points(graph)
    # for sp in split_points:
    #     merge_point = None
    #     next_node = sp
    #     extra_split_points = 0
    #     while True:
    #         if graph.out_degree[next_node] == 1:
    #             next_node = list(graph.successors(next_node))[0]
    #         elif graph.out_degree[next_node] > 1:
    #             extra_split_points += 1
    #             next_node = list(graph.successors(next_node))[0]
    #         else:
    #             break

    #         if graph.in_degree[next_node] > 1:
    #             extra_split_points -= 1
    #             if extra_split_points == 0:
    #                 merge_point = next_node
    #                 break
    #     result.append((sp, merge_point))

def get_nodes_sorted(graph):
    g_sorted = nx.topological_sort(graph)
    return list(g_sorted)

def get_out_streams(layer_graph, node_out):
    for v in layer_graph.values():
        if v["node_out"] == node_out:
            return v["streams_out"]
    assert False, "Cannot find node {} in the layer graph.".format(node_out)

def get_input_nodes(graph):
    input_nodes = []
    for node in graph.nodes():
        if graph.in_degree[node] == 0 or (graph.nodes[node]['type'] == 'ElementWise' and graph.in_degree[node] == 1):
            input_nodes.append(node)
    return input_nodes

def get_output_nodes(graph):
    output_nodes = []
    for node in graph.nodes():
        if graph.out_degree[node] == 0:
            output_nodes.append(node)
    return output_nodes

def visualize_graph(graph: nx.DiGraph, path: str, enable_wandb: bool, graph_name: str, valid: bool = True) -> None:
    PG = nx.nx_pydot.to_pydot(graph)
    if not valid:
        PG.set_bgcolor("lightpink")
    PG.write_png(path + ".png")
    if enable_wandb:
        wandb.log({graph_name: wandb.Image(path + ".png")})


def get_split_points(graph):
    split_points = []
    for node in nx.topological_sort(graph):
        if graph.out_degree[node] > 1:
            split_points.append(node)
        if graph.nodes[node]["type"] == "GlobalAveragePool":
            split_points.insert(0, node)
    return split_points

def get_merge_points(graph):
    merge_points = []
    for node in nx.topological_sort(graph):
        if graph.in_degree[node] > 1:
            merge_points.append(node)
    return merge_points

def get_branch_edges(graph):
    merge_points = get_merge_points(graph)

    branch_edges = []
    for mrg in merge_points:
        branch_edges.append(list(graph.in_edges(mrg)))

    return branch_edges

def remove_off_chip_mem_connections(graph):
    nodes_to_remove = []
    for node in graph.nodes():
        if graph.nodes[node]["type"] == "mem_in" or graph.nodes[node]["type"] == "mem_out":
            nodes_to_remove.append(node)

    graph.remove_nodes_from(nodes_to_remove)

    return graph