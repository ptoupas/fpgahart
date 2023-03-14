import math
from collections import deque

import networkx as nx
import numpy as np

from fpga_hart.layers.activation_3d import Activation3DLayer
from fpga_hart.layers.base_layer_3d import BaseLayer3D
from fpga_hart.layers.batchnorm_3d import BatchNorm3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise_3d import ElementWise3DLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap_3d import GAP3DLayer
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.layers.squeeze_excitation import SqueezeExcitationLayer
from fpga_hart.utils import graph_manipulation, utils
from fpga_hart.utils.matrix_balancing import balance_memory_rates

np.set_printoptions(precision=5, suppress=True, linewidth=250)
np.seterr(divide="ignore", invalid="ignore")

DEBUG = False


class PartitionComposer(BaseLayer3D):
    def __init__(self, max_DSP_util, max_BRAM_util):
        super().__init__(max_DSP_util=max_DSP_util, max_BRAM_util=max_BRAM_util)
        self.preliminary_branch_depth = {}

    def update_layer(self):
        self.full_rate_in = []
        self.full_rate_out = []
        self.max_parallel_muls = 0
        self.max_parallel_adds = 0
        self.memory = 0
        self.memoryKB = 0
        self.depth = 0
        self.branch_depth = 0
        self.data_size_in = 0
        self.data_size_out = 0
        self.mem_bd_in = []
        self.mem_bd_out = []
        self.config = []
        self.structure = {}
        self.dsps_util = 0
        self.dsps_raw = 0
        self.bram_util = 0
        self.bram_raw = 0
        self.latency_sec = 0
        self.latency_cycles = 0
        self.throughput_ops = 0
        self.throughput_vols = 0
        self.total_ops = 0
        self.max_latency_nodes = None

    def get_total_workload(self, graph, wr_factor=1):
        total_wl = 0
        update_valid = False
        for node in nx.topological_sort(graph):
            op_type = graph.nodes[node]["type"]
            hw = graph.nodes[node]["hw"]

            if wr_factor > 1 and "Conv" in op_type:
                update_valid = True

            if not op_type == "mem_in" and not op_type == "mem_out":
                if update_valid:
                    total_wl += hw.get_total_workload() * wr_factor
                else:
                    total_wl += hw.get_total_workload()

        return total_wl

    def get_dp_info(self):
        dp_info = {}

        dp_info["latency(C)"] = self.latency_cycles
        dp_info["latency(S)"] = self.latency_sec
        dp_info["GOP/s"] = self.throughput_ops * 1e-9
        dp_info["vols/s"] = self.throughput_vols
        dp_info["GOPs"] = self.total_ops * 1e-9
        dp_info["DSP"] = self.dsps_util
        dp_info["DSP_RAW"] = self.dsps_raw
        dp_info["BRAM"] = self.bram_util
        dp_info["BRAM_RAW"] = self.bram_raw
        dp_info["rateIn"] = self.full_rate_in
        dp_info["rateOut"] = self.full_rate_out
        dp_info["depth"] = self.depth
        dp_info["branch_depth"] = self.branch_depth
        dp_info["muls"] = self.max_parallel_muls
        dp_info["adds"] = self.max_parallel_adds
        dp_info["memWords"] = self.memory
        dp_info["memKBs"] = self.memoryKB
        dp_info["dataSizeIn"] = (self.data_size_in * self.word_bytes) / 1e6
        dp_info["dataSizeOut"] = (self.data_size_out * self.word_bytes) / 1e6
        dp_info["memBoundedIn"] = self.mem_bd_in
        dp_info["memBoundedOut"] = self.mem_bd_out
        dp_info["slowestNodes"] = self.max_latency_nodes
        dp_info["config"] = self.config
        dp_info["structure"] = self.structure

        return dp_info

    @staticmethod
    def calculate_branch_buffering(graph):
        branch_buffering = {}
        branch_edges = graph_manipulation.get_branch_start_end_points(graph)
        if branch_edges and (branch_edges[0][0] == None or branch_edges[0][1] == None):
            return branch_buffering
        unconnected_branches = {}
        for (in_point, end_point) in branch_edges:
            num_paths = len(
                list(nx.all_simple_paths(graph, source=in_point, target=end_point))
            )
            paths = []
            depths = []
            for path in nx.all_simple_paths(graph, source=in_point, target=end_point):
                depth_branch = 0
                split_node = path[0]
                merge_node = path[-1]
                assert (
                    in_point == split_node and end_point == merge_node
                ), "Branch edges are wrongly defined"
                for p in path[1:-1]:
                    depth_branch += graph.nodes[p]["hw"].depth
                depths.append(depth_branch)
                paths.append(path)

            if num_paths > 2:
                longest_idx = np.argmax(depths)
                shortest_idx = np.argmin(depths)
                final_depth = (
                    min(
                        abs(depths[longest_idx] - depths[shortest_idx]),
                        np.product(graph.nodes[end_point]["hw"].input_shape),
                    )
                    + 2
                )
                branch_buffering[f"{in_point}_{end_point}"] = {"start": in_point, "end": end_point, "conn": paths[shortest_idx][-2], "depth": int(final_depth)}
            elif num_paths == 2:
                longest_idx = np.argmax(depths)
                shortest_idx = np.argmin(depths)
                final_depth = (
                    min(
                        abs(depths[longest_idx] - depths[shortest_idx]),
                        np.product(graph.nodes[end_point]["hw"].input_shape),
                    )
                    + 2
                )
                assert paths[0][-1] == paths[1][-1], "Paths should end at the same node"
                if paths[0][0] == paths[1][0] and not 'Mem_in' in paths[0][0]:
                    branch_buffering[f"{in_point}_{end_point}"] = {"start": in_point, "end": end_point, "conn": paths[shortest_idx][-2], "depth": int(final_depth)}
                else:
                    unconnected_branches[f"{in_point}_{end_point}"] = {"start": in_point, "end": end_point, "conn": paths[longest_idx][-2], "depth": depths[longest_idx], "path": paths[longest_idx]}
            else:
                final_depth = (
                    min(
                        abs(depths[0]),
                        np.product(graph.nodes[end_point]["hw"].input_shape),
                    )
                    + 2
                )
                unconnected_branches[f"{in_point}_{end_point}"] = {"start": in_point, "end": end_point, "conn": paths[0][-2], "depth": final_depth, "path": paths[0]}

        if unconnected_branches:
            remove_keys = []
            unconn_paths = []
            keys = []
            for k, v in unconnected_branches.items():
                if len(v["path"]) == 2 and "Mem_in" in v["path"][0]:
                    remove_keys.append(k)
                unconn_paths.append(v["path"])
                keys.append(k)
            for i in range(len(unconn_paths)):
                for j in range(len(unconn_paths)):
                    if i != j:
                        flag = (all(x in unconn_paths[i] for x in unconn_paths[j]))
                        if flag:
                            remove_keys.append(keys[j])
            for k in remove_keys:
                unconnected_branches.pop(k)

            if len(unconnected_branches) > 1:
                start_p =[]
                end_p = []
                conn_p = []
                depths = []
                assert len(unconnected_branches) == 2, "Only two unconnected branches are supported"
                for k, v in unconnected_branches.items():
                    start_p.append(v["start"])
                    end_p.append(v["end"])
                    conn_p.append(v["conn"])
                    depths.append(v["depth"])
                assert end_p[0] == end_p[1], "Unconnected branches should end at the same point"
                longest_idx = np.argmax(depths)
                shortest_idx = np.argmin(depths)
                branch_buffering[f"{start_p[shortest_idx]}_{end_p[shortest_idx]}"] = {"start": start_p[shortest_idx], "end": end_p[shortest_idx], "conn": conn_p[shortest_idx], "depth": int(depths[longest_idx]-depths[shortest_idx])}
        return branch_buffering

    @staticmethod
    def find_node_idx(graph, request_node):
        for n, node in enumerate(graph.nodes):
            if node == request_node:
                return n
        return -1

    def get_design_point(
        self,
        graph,
        comb: dict(),
        mem_bw_in: list(),
        mem_bw_out: list,
        read_mem_points: list,
        write_mem_points: list,
        gap_approx=False,
        branch_mem=0,
        wr_factor: int = 1,
    ):
        assert len(mem_bw_in) == len(
            read_mem_points
        ), "Input memory break points and memory configuration does not match."
        assert len(mem_bw_out) == len(
            write_mem_points
        ), "Output memory break points and memory configuration does not match."

        self.update_layer()

        off_chip_mem_in = deque()
        off_chip_mem_out = deque()
        for i in range(len(mem_bw_in)):
            off_chip_mem_in.appendleft(mem_bw_in[i] * self.mem_words_per_cycle)
        for i in range(len(mem_bw_out)):
            off_chip_mem_out.appendleft(mem_bw_out[i] * self.mem_words_per_cycle)

        num_layers = graph.number_of_nodes()

        gamma_matrix = np.zeros(shape=(num_layers - 1, num_layers), dtype=float)

        graph_idx = {}
        for i, n in enumerate(nx.topological_sort(graph)):
            graph_idx[n] = i

        graph_layers = {}
        reduce_factor = 0
        for i, n in enumerate(nx.topological_sort(graph)):
            if not (
                graph.nodes[n]["type"] == "mem_in"
                or graph.nodes[n]["type"] == "mem_out"
            ):
                graph_layers[i - reduce_factor] = n
            else:
                reduce_factor += 1

        total_muls = 0
        total_adds = 0

        layer_fifos_arrays = {
            "branch_buffering": 0,
        }

        total_depth = 0
        total_brams = 0
        config = {}
        layers_ii = []
        for n, node in enumerate(nx.topological_sort(graph)):
            if DEBUG:
                print("*" * 50)
                print("Processing node: {}".format(node))
            op_type = graph.nodes[node]["type"]
            hw = graph.nodes[node]["hw"]
            node_predecessors = list(graph.predecessors(node))
            # node_successors = list(graph.successors(node))

            if op_type == "mem_in":
                assert (
                    not node in comb.keys()
                ), f"Memory IN node: {node} cannot have configuration."
                gamma_matrix[n, n] = off_chip_mem_in.pop()
                graph.nodes[node]["prod_rate"] = gamma_matrix[n, n]
                continue

            if op_type == "mem_out":
                assert (
                    not node in comb.keys()
                ), f"Memory OUT node: {node} cannot have configuration."
                gamma_matrix[graph_idx[node_predecessors[0]], n] = -off_chip_mem_out.pop()
                curr_layer_rate = gamma_matrix[graph_idx[node_predecessors[0]], n]
                graph.nodes[node]["cons_rate"] = gamma_matrix[graph_idx[node_predecessors[0]], n]
                continue

            assert (
                node in comb.keys()
            ), f"Node: {node} does not have configuration. Please check the graph creation."
            c = comb[node]

            curr_layer_rate = 1000000
            if graph.in_degree(node) == 1:
                assert (
                    len(node_predecessors) == 1
                ), f"Node: {node} must have exactly one predecessor. Graph inconsistency."
                prev_layer_rate_1 = graph.nodes[node_predecessors[0]]["prod_rate"]
                prev_layer_rate_2 = None
            elif graph.in_degree(node) == 2:
                assert (
                    len(node_predecessors) == 2
                ), f"Node: {node} must have exactly two predecessors. Graph inconsistency."
                prev_layer_rate_1 = graph.nodes[node_predecessors[0]]["prod_rate"]
                prev_layer_rate_2 = graph.nodes[node_predecessors[1]]["prod_rate"]
            else:
                raise Exception(
                    f"Node: {node} has more than 2 predecessors. This kind of connection is not yet supported."
                )

            if isinstance(hw, GAP3DLayer):
                dp_info = hw.get_design_point(
                    coarse_inout=c[0],
                    mem_bw_in=curr_layer_rate,
                    mem_bw_out=curr_layer_rate,
                    gap_approx=gap_approx,
                    ignore_bw_util=True,
                )
                config[node] = utils.generate_layer_config(hw, c, wr_factor=wr_factor)
            elif isinstance(hw, Convolutional3DLayer):
                dp_info = hw.get_design_point(
                    f_fine=c[0],
                    f_coarseIn=c[1],
                    f_coarseOut=c[2],
                    mem_bw_in=curr_layer_rate,
                    mem_bw_out=curr_layer_rate,
                    ignore_bw_util=True,
                )
                config[node] = utils.generate_layer_config(hw, c, wr_factor=wr_factor)
            elif isinstance(hw, Pooling3DLayer):
                dp_info = hw.get_design_point(
                    f_fine=c[0],
                    f_coarse_inout=c[1],
                    mem_bw_in=curr_layer_rate,
                    mem_bw_out=curr_layer_rate,
                    ignore_bw_util=True,
                )
                config[node] = utils.generate_layer_config(hw, c, wr_factor=wr_factor)
            elif isinstance(hw, Activation3DLayer):
                dp_info = hw.get_design_point(
                    coarse_inout=c[0],
                    mem_bw_in=curr_layer_rate,
                    mem_bw_out=curr_layer_rate,
                    ignore_bw_util=True,
                )
                config[node] = utils.generate_layer_config(hw, c, wr_factor=wr_factor)
            elif isinstance(hw, ElementWise3DLayer):
                if hw.broadcasting:
                    prev_nodes = [pn for pn in graph.predecessors(node)]
                    prev_nodes_out_shapes = [
                        graph.nodes[pn]["hw"].output_shape for pn in prev_nodes
                    ]
                    node_fs = prev_nodes[
                        prev_nodes_out_shapes.index(max(prev_nodes_out_shapes))
                    ]
                    node_rs = prev_nodes[
                        prev_nodes_out_shapes.index(min(prev_nodes_out_shapes))
                    ]
                    prev_layer_rate_1 = graph.nodes[node_fs]["prod_rate"]
                    prev_layer_rate_2 = graph.nodes[node_rs]["prod_rate"]
                dp_info = hw.get_design_point(
                    coarse_inout=c[0],
                    mem_bw_in_1=curr_layer_rate,
                    mem_bw_in_2=curr_layer_rate,
                    mem_bw_out=curr_layer_rate,
                    ignore_bw_util=True,
                )
                config[node] = utils.generate_layer_config(hw, c, wr_factor=wr_factor)
            elif isinstance(hw, BatchNorm3DLayer):
                dp_info = hw.get_design_point(
                    coarse_inout=c[0],
                    mem_bw_in=curr_layer_rate,
                    mem_bw_out=curr_layer_rate,
                    ignore_bw_util=True,
                )
                config[node] = utils.generate_layer_config(hw, c, wr_factor=wr_factor)
            elif isinstance(hw, SqueezeExcitationLayer):
                dp_info = hw.get_design_point(
                    f_gap_coarsein=c[0],
                    f_gap_coarseout=c[1],
                    f_fine_1=c[2],
                    f_coarseIn_1=c[3],
                    f_coarseOut_1=c[4],
                    f_relu_cinout=c[5],
                    f_fine_2=c[6],
                    f_coarseIn_2=c[7],
                    f_coarseOut_2=c[8],
                    f_sigm_cinout=c[9],
                    f_mul_coarsein1=c[10],
                    f_mul_coarsein2=c[11],
                    f_mul_coarseout=c[12],
                    mem_bw_in=curr_layer_rate,
                    mem_bw_out=curr_layer_rate,
                )
                config[node] = utils.generate_layer_config(hw, c, wr_factor=wr_factor)
            elif isinstance(hw, FCLayer):
                dp_info = hw.get_design_point(
                    coarse_in=c[0],
                    coarse_out=c[1],
                    mem_bw_in=curr_layer_rate,
                    mem_bw_out=curr_layer_rate,
                    ignore_bw_util=True,
                )
                config[node] = utils.generate_layer_config(hw, c, wr_factor=wr_factor)
            else:
                assert False, "Not supported layer"

            if isinstance(hw, ElementWise3DLayer):
                if not dp_info["config"]:
                    self.update_layer()
                    if DEBUG:
                        mem_kb = (dp_info["memWords"] * self.word_bytes) / 1e3
                        mem_bram = math.ceil(mem_kb / self.bram_Kbytes)
                        curr_bram_util = (mem_bram / self.bram) * 100
                        curr_dsps_util = (dp_info["muls"] / self.dsp) * 100
                        print(
                            f"{node}: Discarding design point. DSPS={curr_dsps_util}, BRAM={curr_bram_util}"
                        )
                    return self.get_dp_info()
                (
                    latency_cycles,
                    full_rate_in_1,
                    full_rate_in_2,
                    full_rate_out,
                    muls,
                    adds,
                    memory,
                    bram_raw,
                    depth,
                    mem_bd_in_1,
                    mem_bd_in_2,
                    mem_bd_out,
                ) = (
                    dp_info["latency(C)"],
                    dp_info["rateIn"][0],
                    dp_info["rateIn"][1],
                    dp_info["rateOut"][0],
                    dp_info["muls"],
                    dp_info["adds"],
                    dp_info["memWords"],
                    dp_info["BRAM_RAW"],
                    dp_info["depth"],
                    dp_info["memBoundedIn"][0],
                    dp_info["memBoundedIn"][0],
                    dp_info["memBoundedOut"],
                )
                if hw.broadcasting:
                    cp1 = graph_idx[node_fs]
                    cp2 = graph_idx[node_rs]
                else:
                    cp1 = graph_idx[node_predecessors[0]]
                    cp2 = graph_idx[node_predecessors[1]]
                gamma_matrix[cp1, n] = -full_rate_in_1
                gamma_matrix[cp2, n] = -full_rate_in_2
                gamma_matrix[n, n] = full_rate_out
                graph.nodes[node]["cons_rate_1"] = full_rate_in_1
                graph.nodes[node]["cons_rate_2"] = full_rate_in_2
                graph.nodes[node]["prod_rate"] = full_rate_out
            else:
                if not dp_info["config"]:
                    self.update_layer()
                    if DEBUG:
                        mem_kb = (dp_info["memWords"] * self.word_bytes) / 1e3
                        mem_bram = math.ceil(mem_kb / self.bram_Kbytes)
                        curr_bram_util = (mem_bram / self.bram) * 100
                        curr_dsps_util = (dp_info["muls"] / self.dsp) * 100
                        print(
                            f"{node}: Discarding design point. DSPS={curr_dsps_util}, BRAM={curr_bram_util}"
                        )
                    return self.get_dp_info()
                (
                    latency_cycles,
                    full_rate_in,
                    full_rate_out,
                    muls,
                    adds,
                    memory,
                    bram_raw,
                    depth,
                    mem_bd_in,
                    mem_bd_out,
                ) = (
                    dp_info["latency(C)"],
                    dp_info["rateIn"][0],
                    dp_info["rateOut"][0],
                    dp_info["muls"],
                    dp_info["adds"],
                    dp_info["memWords"],
                    dp_info["BRAM_RAW"],
                    dp_info["depth"],
                    dp_info["memBoundedIn"][0],
                    dp_info["memBoundedOut"][0],
                )
                cp = graph_idx[node_predecessors[0]]
                gamma_matrix[cp, n] = -full_rate_in
                gamma_matrix[n, n] = full_rate_out
                graph.nodes[node]["cons_rate"] = full_rate_in
                graph.nodes[node]["prod_rate"] = full_rate_out

            layers_ii.append(latency_cycles - depth)

            total_muls += muls
            total_adds += adds
            total_brams += bram_raw
            total_depth += depth
            curr_bram_util = (total_brams / self.bram) * 100
            curr_dsps_util = (total_muls / self.dsp) * 100

            if DEBUG:
                print(
                    f"{node} - Latency(C)={latency_cycles}, Latency(C)-Depth={latency_cycles-depth}, DSPs={muls}, BRAM={bram_raw}, Depth={depth}, Total Depth={total_depth}, Total DSPs={total_muls}, Total BRAM={total_brams}, BRAM Util={curr_bram_util}, DSP Util={curr_dsps_util}"
                )

            # if (
            #     not dp_info["config"]
            #     or curr_dsps_util >= self.max_DSP_util
            #     or curr_bram_util >= self.max_BRAM_util
            # ):
            #     self.update_layer()
            #     if DEBUG:
            #         print(
            #             f"{node}: Discarding design point. DSPS={curr_dsps_util}, BRAM={curr_bram_util}"
            #         )
            #     return self.get_dp_info()

        assert (
            len(off_chip_mem_in) == 0
        ), "Off-chip memory IN points left hanging. Wrong configuration of the graph."
        assert (
            len(off_chip_mem_out) == 0
        ), "Off-chip memory OUT points left hanging. Wrong configuration of the graph."

        layer_fifos_arrays["branch_buffering"] = self.calculate_branch_buffering(
            graph
        )
        self.preliminary_branch_depth = layer_fifos_arrays["branch_buffering"]

        if DEBUG:
            print(f"Branch buffering: {layer_fifos_arrays['branch_buffering']}")
            print("Γ:\n{}".format(gamma_matrix))
        gamma_matrix_balanced = balance_memory_rates(gamma_matrix.copy())
        # gamma_matrix_balanced = balance_matrix(gamma_matrix_balanced)

        # TODO: Properly find whether the graph is memory bounding and in which input/output (from gama matrix balancing)
        mem_bounded_out = []
        mem_bounded_in = []
        shapes_in = []
        shapes_out = []
        rates_in = []
        rates_out = []
        mem_conns_in = []
        mem_conns_out = []
        for n, node in enumerate(nx.topological_sort(graph)):
            if graph.nodes[node]["type"] == "mem_in":
                nn = graph_idx[list(graph.successors(node))[0]]
                if gamma_matrix_balanced[n, n] < abs(gamma_matrix_balanced[n, nn]):
                    mem_bounded_in.append(True)
                    if DEBUG:
                        print(
                            f"Memory in node {n} with rate {gamma_matrix_balanced[n,n]} -> {gamma_matrix_balanced[n,nn]}"
                        )
                else:
                    mem_bounded_in.append(False)
                    gamma_matrix_balanced[n, n] = abs(gamma_matrix_balanced[n, nn])
                rates_in.append(gamma_matrix_balanced[n, n])
                shapes_in.append(graph.nodes[node]["hw"].output_shape)
                mem_conns_in.append([n, n])
            if graph.nodes[node]["type"] == "mem_out":
                pn = graph_idx[list(graph.predecessors(node))[0]]
                if (
                    abs(gamma_matrix_balanced[pn, n])
                    < gamma_matrix_balanced[pn, pn]
                ):
                    mem_bounded_out.append(True)
                    if DEBUG:
                        print(
                            f"Memory out node {n} with rate {gamma_matrix_balanced[pn,n]} -> {gamma_matrix_balanced[pn,pn]}"
                        )
                else:
                    mem_bounded_out.append(False)
                    gamma_matrix_balanced[pn, n] = -gamma_matrix_balanced[pn, pn]
                rates_out.append(abs(gamma_matrix_balanced[pn, n]))
                shapes_out.append(graph.nodes[node]["hw"].input_shape)
                mem_conns_out.append([pn, n])

        if DEBUG:
            print("Γ Balanced:\n{}".format(gamma_matrix_balanced))
        workload_matrix = self.get_workload_matrix(graph, num_layers)
        if DEBUG:
            print("WL:\n{}".format(workload_matrix))
        ii_matrix = np.nan_to_num(workload_matrix / gamma_matrix_balanced)
        if DEBUG:
            print("II:\n{}".format(ii_matrix))

        batch_size = 1
        (
            latency_sec,
            latency_cycles,
            thr_in,
            thr_out,
            dsps_util,
            dsps_raw,
            bram_util,
            bram_raw,
            memKBs,
        ) = self.get_performance(
            workload_matrix,
            ii_matrix,
            mem_conns_in,
            mem_conns_out,
            total_muls,
            total_adds,
            layer_fifos_arrays,
            total_brams,
            total_depth,
            graph,
            config,
            batch=batch_size,
            per_layer_ii=layers_ii,
            wr_factor=wr_factor,
        )
        slowest_nodes_idxs = np.array(layers_ii).argsort()[::-1][:n].tolist()[:3]
        slowest_nodes_names = [graph_layers[n] for n in slowest_nodes_idxs[:3]]
        self.max_latency_nodes = slowest_nodes_names

        total_ops = self.get_total_workload(graph, wr_factor=wr_factor) * batch_size
        throughput_ops = total_ops / latency_sec

        #TODO: double check if this is actually correct. Every input througput should be equal to every output?
        thr_out_vols = []
        for idx_in, (i, j) in enumerate(mem_conns_in):
            curr_thr_in = thr_in[idx_in] / workload_matrix[i, j]
            for idx_out, (k, h) in enumerate(mem_conns_out):
                curr_thr_out = thr_out[idx_out] / workload_matrix[k, h]
                thr_out_vols.append(curr_thr_out)
                assert math.isclose(curr_thr_in, curr_thr_out), "Thoughputs missmatch between {} IN and {} OUT connections. thr in = {}, thr out = {}.".format(curr_thr_in, curr_thr_out)

        if (
            dsps_util < self.max_DSP_util
            and bram_util < self.max_BRAM_util
        ):
            self.full_rate_in = rates_in
            self.full_rate_out = rates_out
            self.max_parallel_muls = total_muls
            self.max_parallel_adds = total_adds
            self.depth = total_depth
            self.branch_depth = layer_fifos_arrays["branch_buffering"]
            self.mem_bd_in = mem_bounded_in
            self.mem_bd_out = mem_bounded_out
            self.data_size_in = sum(map(lambda x: np.prod(np.array(x[1:])), shapes_in))
            self.data_size_out = sum(
                map(lambda x: np.prod(np.array(x[1:])), shapes_out)
            )

            self.total_ops = total_ops
            self.config = config
            self.structure = graph_manipulation.get_graph_structure(graph, config)
            self.memoryKB = memKBs
            self.dsps_util = dsps_util
            self.dsps_raw = dsps_raw
            self.bram_util = bram_util
            self.bram_raw = bram_raw
            self.latency_sec = latency_sec
            self.latency_cycles = int(latency_cycles)
            self.throughput_ops = throughput_ops
            #TODO: Is this correct? Do we have a throughput out that is the sum of all out throughputs to the mem?
            self.throughput_vols = np.sum(np.array(thr_out_vols))
            if DEBUG:
                print(
                    "GOPs/s={:.2f}, DSPS={}({:.2f}), BRAM={}({:.2f}), depth={}, latency(s)={:.2f}, latency(c)={:.2f}, mem bounded in = {}, mem bounded out = {}".format(
                        throughput_ops * 1e-9,
                        dsps_raw,
                        dsps_util,
                        bram_raw,
                        bram_util,
                        total_depth,
                        latency_sec,
                        int(latency_cycles),
                        mem_bounded_in,
                        mem_bounded_out,
                    )
                )
        else:
            self.update_layer()
            if DEBUG:
                print(f"Discarding design point. DSPS={dsps_util}, BRAM={bram_util}")
        return self.get_dp_info()

    def get_workload_matrix(self, graph, num_layers):
        graph_idx = {}
        for i, n in enumerate(nx.topological_sort(graph)):
            graph_idx[n] = i

        workload_matrix = np.zeros(shape=(num_layers - 1, num_layers), dtype=float)

        for n, node in enumerate(nx.topological_sort(graph)):

            op_type = graph.nodes[node]["type"]
            hw = graph.nodes[node]["hw"]

            if op_type == "mem_in":
                workload_matrix[n, n] = np.prod(np.array(hw.output_shape[1:]))
                continue

            if op_type == "mem_out":
                pn = graph_idx[list(graph.predecessors(node))[0]]
                workload_matrix[pn, n] = np.prod(np.array(hw.input_shape[1:]))
                continue

            if isinstance(hw, ElementWise3DLayer):
                cp1 = graph_idx[list(graph.in_edges(node))[0][0]]
                cp2 = graph_idx[list(graph.in_edges(node))[1][0]]
                workload_matrix[cp1, n] = np.prod(np.array(hw.input_shape_1[1:]))
                workload_matrix[cp2, n] = np.prod(np.array(hw.input_shape_2[1:]))
                workload_matrix[n, n] = np.prod(np.array(hw.output_shape[1:]))
            else:
                cp = graph_idx[list(graph.in_edges(node))[0][0]]
                workload_matrix[cp, n] = np.prod(np.array(hw.input_shape[1:]))
                workload_matrix[n, n] = np.prod(np.array(hw.output_shape[1:]))

        return workload_matrix

    def get_performance(
        self,
        workload_matrix,
        ii,
        mem_conns_in,
        mem_conns_out,
        muls,
        adds,
        layer_fifos_arrays,
        layer_brams,
        depth,
        graph,
        config,
        batch=1,
        per_layer_ii=None,
        wr_factor=1
    ):
        if wr_factor > 1:
            conv_nodes_count = 0
            for n, node in enumerate(nx.topological_sort(graph)):
                hw = graph.nodes[node]["hw"]
                if isinstance(hw, Convolutional3DLayer):
                    wr_kernel_shape = [hw.filters, hw.channels] + hw.kernel_shape
                    conv_nodes_count += 1
            if conv_nodes_count > 1:
                raise ValueError(f"Partition with weights reloading should not have more than 1 Conv layers. Currently {conv_nodes_count}.")
        else:
            wr_kernel_shape = [1, 1, 1, 1, 1]

        mem_kb_total = 0
        bram_raw_out = layer_brams

        if "branch_buffering" in layer_fifos_arrays:
            for _, v in layer_fifos_arrays["branch_buffering"].items():
                merge_node = v["end"]
                curr_depth = v["depth"]
                # depth_per_fifo = math.ceil(curr_depth/config[merge_node]['coarse_factor'])
                bram_raw_out += (
                    self.bram_stream_resource_model(curr_depth, 16)
                    * config[merge_node]["coarse_factor"]
                )

        bram_util = (bram_raw_out / self.bram) * 100
        dsps_util = (muls / self.dsp) * 100
        dsps_raw_out = muls

        if per_layer_ii is not None:
            latency_cycles = int(max(per_layer_ii)) * batch + depth * wr_factor + (wr_factor - 1) * np.prod(np.array(wr_kernel_shape))
        else:
            latency_cycles = int(np.max(np.abs(ii))) * batch + depth * wr_factor + (wr_factor - 1) * np.prod(np.array(wr_kernel_shape))
        latency_sec = latency_cycles / self.cycles_per_sec
        if DEBUG:
            print(
                "Gama matrix calculated: {} - Max II over layers: {} = {}".format(
                    int(np.max(np.abs(ii))) * batch + depth,
                    int(max(per_layer_ii)) * batch + depth,
                    (int(np.max(np.abs(ii))) * batch + depth)
                    - (int(max(per_layer_ii)) * batch + depth),
                )
            )

        thr_in = []
        thr_out = []
        for (i, j) in mem_conns_in:
            thr_in_tmp = (batch * workload_matrix[i, j]) / latency_sec  # Input words per second
            thr_in.append(thr_in_tmp)
        for (i, j) in mem_conns_out:
            thr_out_tmp = (batch * workload_matrix[i, j]) / latency_sec  # Output words per second
            thr_out.append(thr_out_tmp)

        return (
            latency_sec,
            latency_cycles,
            thr_in,
            thr_out,
            dsps_util,
            dsps_raw_out,
            bram_util,
            bram_raw_out,
            mem_kb_total,
        )
