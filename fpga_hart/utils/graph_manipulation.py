import math
import random
from collections import deque
from typing import Tuple

import networkx as nx
import numpy as np

from fpga_hart.layers.gap import GAPLayer
from fpga_hart.layers.memory_interface import MemoryNode
from fpga_hart.utils import utils


def has_gap(graph: nx.DiGraph) -> bool:
    result = False
    for node in graph.nodes:
        hw = graph.nodes[node]["hw"]
        if isinstance(hw, GAPLayer):
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

    branch_edges_1 = utils.get_branch_edges(phase_1_graph)
    branch_edges_2 = utils.get_branch_edges(phase_2_graph)
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

    # if not in_connections and not out_connections:
    for in_n in input_nodes:
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
        if not 'mem_out' in con_out.lower():
            add_node_to_position(
                G=graph,
                new_node="Mem_out{}".format(mem_out_count),
                connect_node=con_out,
                connect_pos="post",
            )
        write_points.append(con_out)
        mem_out_count += 1

    # if gap_approx:
    #     next_nodes = []
    #     gap_nodes = []
    #     for n in graph.nodes():
    #         if graph.nodes[n]['type'] == 'GlobalAveragePool':
    #             next_nodes.append(list(graph.successors(n))[0])
    #             gap_nodes.append(n)
    #             graph.remove_edge(n, list(graph.successors(n))[0])

    #     for n, g in zip(next_nodes, gap_nodes):
    #         read_points.append(n)
    #         add_node_to_position(G=graph, new_node='Mem_in{}'.format(mem_in_count), connect_node=n, connect_pos='pre')
    #         mem_in_count += 1
    #         write_points.append(g)
    #         add_node_to_position(G=graph, new_node='Mem_out{}'.format(mem_out_count), connect_node=g, connect_pos='post')
    #         mem_out_count += 1

    return read_points, write_points


def visualize_graph(graph: nx.DiGraph, path: str) -> None:
    PG = nx.nx_pydot.to_pydot(graph)
    PG.write_png(path + ".png")


def get_input_nodes(graph):
    input_nodes = []
    for node in graph.nodes():
        if graph.in_degree[node] == 0:
            input_nodes.append(node)
    return input_nodes

def get_output_nodes(graph):
    output_nodes = []
    for node in graph.nodes():
        if graph.out_degree[node] == 0:
            output_nodes.append(node)
    return output_nodes
