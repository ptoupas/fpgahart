import copy
import itertools
import json
import logging
import math
import os
import random
import time
from collections import deque
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
import scipy.constants as sc
import wandb
from fpga_hart import _logger

from ..layers.activation import ActivationLayer
from ..layers.base_layer import BaseLayer
from ..layers.batchnorm_3d import BatchNorm3DLayer
from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.elemwise import ElementWiseLayer
from ..layers.fully_connected import FCLayer
from ..layers.gap import GAPLayer
from ..layers.memory_interface import MemoryNode
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..partitions.partition_compose import PartitionComposer
from ..utils import utils


class SimulatedAnnealing(BaseLayer):
    def __init__(
        self,
        graph,
        branch_mem=0,
        t_min=1e-6,
        t_max=7,
        iterationPerTemp=15,
        cooling_rate=0.99,
        best_of_iter=1,
        partition_name="",
        gap_approx=False,
        ml_flow_id=None,
        wandb_config=None,
    ):
        self.wandb_config = wandb_config
        super().__init__(
            max_DSP_util=95.0
            if self.wandb_config == None
            else self.wandb_config.max_dsp_util,
            max_BRAM_util=95.0
            if self.wandb_config == None
            else self.wandb_config.max_bram_util,
        )
        # _logger.setLevel(level=logging.DEBUG)
        if self.wandb_config != None:
            self.wandb_config.update(
                {
                    "Device": self.fpga_device,
                    "Clock frequency": self.clock_freq,
                    "DSPs": self.dsp,
                    "BRAMs": self.bram,
                    "Memory BW": self.mem_bw,
                }
            )

        self.gap_approx = gap_approx
        self.ml_flow_id = ml_flow_id
        self.part_name = partition_name

        self.graph = graph

        mem_kb = (branch_mem * self.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.bram_Kbytes)
        self.branch_bram_util = (mem_bram / self.bram) * 100
        self.branch_mem = branch_mem

        # Simulate Annealing Variables
        self.k = sc.Boltzmann
        self.t_min = (
            t_min
            if self.wandb_config == None
            else self.wandb_config.simulatedAnnealing["t_min"]
        )
        self.t_max = (
            t_max
            if self.wandb_config == None
            else self.wandb_config.simulatedAnnealing["t_max"]
        )
        self.cooling_rate = (
            cooling_rate
            if self.wandb_config == None
            else self.wandb_config.simulatedAnnealing["cooling_rate"]
        )
        self.iterationPerTemp = (
            iterationPerTemp
            if self.wandb_config == None
            else self.wandb_config.simulatedAnnealing["iterationPerTemp"]
        )
        self.best_of_iter = (
            best_of_iter
            if self.wandb_config == None
            else self.wandb_config.simulatedAnnealing["best_of_iter"]
        )
        self.param_changes = 0
        self.freeze_param = False

        self.partition_composer = PartitionComposer(
            max_DSP_util=self.max_DSP_util, max_BRAM_util=self.max_BRAM_util
        )

    def has_gap(self):
        result = False
        for node in self.graph.nodes:
            op_type = self.graph.nodes[node]["type"]
            hw = self.graph.nodes[node]["hw"]
            if isinstance(hw, GAPLayer):
                result = True
                break
        return result

    def split_graph(self):
        """
        Create a 1st break point of the graph right before the mul operation of squeeze excitation module.
        Then search (using simulated annealing) from which point in the graph we will start the 2nd phase of the execution (or we will reconfigure the whole FPGA). Depending on the point we might need to store extra intermediate results of phase1 and read them from off-chip memory during phase 2. The other alternative will be to re-compute some layers but reduce the off-chip memory transfers.
        """
        mem_in_1, mem_out_1, mem_in_2, mem_out_2 = [], [], [], []

        merge_nodes = [n for n in self.graph.nodes if self.graph.in_degree[n] > 1]
        split_nodes = [n for n in self.graph.nodes if self.graph.out_degree[n] > 1]
        break_node_gap = ""
        for node in self.graph.nodes:
            op_type = self.graph.nodes[node]["type"]
            hw = self.graph.nodes[node]["hw"]
            if (
                op_type == "ElementWise"
                and self.graph.in_degree[node] > 1
                and hw.type == "Mul"
            ):
                break_node_gap = node

        phase_1 = deque()
        check_node = break_node_gap
        predec_nodes = [n for n in self.graph.predecessors(check_node)]
        while predec_nodes:
            if len(predec_nodes) > 1:
                for pn in predec_nodes:
                    op_type = self.graph.nodes[pn]["type"]
                    if op_type == "Activation":
                        phase_1.appendleft(pn)
                        check_node = pn
                        predec_nodes = [n for n in self.graph.predecessors(check_node)]
                        break
            else:
                curr_node = predec_nodes[0]
                phase_1.appendleft(curr_node)
                check_node = curr_node
                predec_nodes = [n for n in self.graph.predecessors(check_node)]

        phase_1_graph_frozen = self.graph.subgraph(list(phase_1))
        phase_1_graph = nx.DiGraph(phase_1_graph_frozen)
        phase_1_edges = list(phase_1_graph.edges)

        mem_in_1_count = 1
        for node in list(phase_1_graph.nodes()):
            if node in merge_nodes and not phase_1_graph.in_degree[node] > 1:
                mem_in_1.append(node)
            if phase_1_graph.in_degree[node] == 0:
                mem_in_1.append(node)
            if phase_1_graph.out_degree[node] == 0:
                mem_out_1.append(node)

        phase_2_graph = self.graph.copy()
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

        mem_in_2_count = 1
        for node in list(phase_2_graph.nodes()):
            if node in merge_nodes and not phase_2_graph.in_degree[node] > 1:
                mem_in_2.append(node)
            if (
                phase_2_graph.in_degree[node] == 0
            ):  # and node in [n for n in self.graph.successors(phase_2_read_node)]:
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
                    np.prod(
                        np.array(phase_1_graph.nodes[pair[0]]["hw"].output_shape[1:])
                    ),
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
                    np.prod(
                        np.array(phase_2_graph.nodes[pair[0]]["hw"].output_shape[1:])
                    ),
                )
            branch_buffer_2 += max_shape

        mem_kb = ((branch_buffer_1 + branch_buffer_2) * self.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.bram_Kbytes)
        branch_bram_util = (mem_bram / self.bram) * 100
        if branch_bram_util > self.max_BRAM_util:
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

    def validate_configs(self, graph_1_dp, graph_2_dp):
        g_1_dsp_util = graph_1_dp["DSP"]
        g_2_dsp_util = graph_2_dp["DSP"]

        g_1_bram_util = graph_1_dp["BRAM"]
        g_2_bram_util = graph_2_dp["BRAM"]

        if g_1_dsp_util + g_2_dsp_util >= self.max_DSP_util:
            return False
        if g_1_bram_util + g_2_bram_util >= self.max_BRAM_util:
            return False

        return True

    def run_optimizer_double_graph(self):
        (
            graph_1,
            graph_2,
            bb1,
            bb2,
            mem_in_1,
            mem_out_1,
            mem_in_2,
            mem_out_2,
        ) = self.split_graph()

        nIN1 = len(mem_in_1)
        nOUT1 = len(mem_out_1)
        assert nIN1 > 0 and nOUT1 > 0, "No memory in/out nodes found"
        config_1, mem_bw_1 = self.generate_random_config(
            target_graph=graph_1, n_in=nIN1, n_out=nOUT1
        )
        cost_1, dp_info_1 = self.get_cost(
            config_1,
            mem_bw_1,
            target_graph=graph_1,
            branch_mem_update=bb1,
            mem_in_conns=mem_in_1,
            mem_out_conns=mem_out_1,
        )

        nIN2 = len(mem_in_2)
        nOUT2 = len(mem_out_2)
        assert nIN2 > 0 and nOUT2 > 0, "No memory in/out nodes found"
        config_2, mem_bw_2 = self.generate_random_config(
            target_graph=graph_2, n_in=nIN2, n_out=nOUT2
        )
        cost_2, dp_info_2 = self.get_cost(
            config_2,
            mem_bw_2,
            target_graph=graph_2,
            branch_mem_update=bb2,
            mem_in_conns=mem_in_2,
            mem_out_conns=mem_out_2,
        )

        if cost_1 is None or cost_2 is None:
            for _ in range(500):
                (
                    graph_1,
                    graph_2,
                    bb1,
                    bb2,
                    mem_in_1,
                    mem_out_1,
                    mem_in_2,
                    mem_out_2,
                ) = self.split_graph()

                nIN1 = len(mem_in_1)
                nOUT1 = len(mem_out_1)
                assert nIN1 > 0 and nOUT1 > 0, "No memory in/out nodes found"
                config_1, mem_bw_1 = self.generate_random_config(
                    target_graph=graph_1, n_in=nIN1, n_out=nOUT1
                )
                cost_1, dp_info_1 = self.get_cost(
                    config_1,
                    mem_bw_1,
                    target_graph=graph_1,
                    branch_mem_update=bb1,
                    mem_in_conns=mem_in_1,
                    mem_out_conns=mem_out_1,
                )

                nIN2 = len(mem_in_2)
                nOUT2 = len(mem_out_2)
                assert nIN2 > 0 and nOUT2 > 0, "No memory in/out nodes found"
                config_2, mem_bw_2 = self.generate_random_config(
                    target_graph=graph_2, n_in=nIN2, n_out=nOUT2
                )
                cost_2, dp_info_2 = self.get_cost(
                    config_2,
                    mem_bw_2,
                    target_graph=graph_2,
                    branch_mem_update=bb2,
                    mem_in_conns=mem_in_2,
                    mem_out_conns=mem_out_2,
                )
                if (
                    cost_1 is not None
                    and cost_2 is not None
                    and self.validate_configs(dp_info_1, dp_info_2)
                ):
                    break
            if cost_1 is None or cost_2 is None:
                return None, None, None

        prev_state_1 = config_1
        prev_state_2 = config_2
        prev_cost = cost_1 + cost_2
        solution_dp_1 = dp_info_1
        solution_dp_2 = dp_info_2
        solution_mem_1 = mem_bw_1
        solution_mem_2 = mem_bw_2

        current_temp = self.t_max

        print(f"Temperature  |  Latency")
        split_graph_count = 0
        while current_temp > self.t_min:
            for i in range(self.iterationPerTemp):
                if split_graph_count % 1000 == 0:
                    (
                        graph_1,
                        graph_2,
                        bb1,
                        bb2,
                        mem_in_1,
                        mem_out_1,
                        mem_in_2,
                        mem_out_2,
                    ) = self.split_graph()

                prev_state_1 = self.fix_inconsistent_config(prev_state_1, graph_1)
                prev_state_2 = self.fix_inconsistent_config(prev_state_2, graph_2)

                nIN1 = len(mem_in_1)
                nOUT1 = len(mem_out_1)
                assert nIN1 > 0 and nOUT1 > 0, "No memory in/out nodes found"
                new_state_1, new_mem_bw_1 = self.generate_random_config(
                    target_graph=graph_1,
                    neighbours=True,
                    prev_state=prev_state_1,
                    n_in=nIN1,
                    n_out=nOUT1,
                )
                new_cost_1, new_dp_info_1 = self.get_cost(
                    new_state_1,
                    new_mem_bw_1,
                    target_graph=graph_1,
                    branch_mem_update=bb1,
                    mem_in_conns=mem_in_1,
                    mem_out_conns=mem_out_1,
                )

                nIN2 = len(mem_in_2)
                nOUT2 = len(mem_out_2)
                assert nIN2 > 0 and nOUT2 > 0, "No memory in/out nodes found"
                new_state_2, new_mem_bw_2 = self.generate_random_config(
                    target_graph=graph_2,
                    neighbours=True,
                    prev_state=prev_state_2,
                    n_in=nIN2,
                    n_out=nOUT2,
                )
                new_cost_2, new_dp_info_2 = self.get_cost(
                    new_state_2,
                    new_mem_bw_2,
                    target_graph=graph_2,
                    branch_mem_update=bb2,
                    mem_in_conns=mem_in_2,
                    mem_out_conns=mem_out_2,
                )

                if (
                    new_cost_1 is None
                    or new_cost_2 is None
                    or not self.validate_configs(new_dp_info_1, new_dp_info_2)
                ):
                    continue
                split_graph_count += 1

                new_cost = new_cost_1 + new_cost_2
                cost_diff = prev_cost - new_cost
                if cost_diff >= 0:
                    prev_state_1 = copy.deepcopy(new_state_1)
                    prev_state_2 = copy.deepcopy(new_state_2)
                    prev_cost = copy.deepcopy(new_cost)
                    solution_mem_1, solution_dp_1 = (
                        copy.deepcopy(new_mem_bw_1),
                        copy.deepcopy(new_dp_info_1),
                    )
                    solution_mem_2, solution_dp_2 = (
                        copy.deepcopy(new_mem_bw_2),
                        copy.deepcopy(new_dp_info_2),
                    )
                    self.visualize_graph(
                        graph_1,
                        os.getcwd()
                        + "/fpga_modeling_reports/partition_graphs/"
                        + self.part_name
                        + "_phase_1",
                    )
                    self.visualize_graph(
                        graph_2,
                        os.getcwd()
                        + "/fpga_modeling_reports/partition_graphs/"
                        + self.part_name
                        + "_phase_2",
                    )
                else:
                    if random.uniform(0, 1) < math.exp(
                        (cost_diff / (self.k * current_temp))
                    ):
                        prev_state_1 = copy.deepcopy(new_state_1)
                        prev_state_2 = copy.deepcopy(new_state_2)
                        prev_cost = copy.deepcopy(new_cost)
                        solution_mem_1, solution_dp_1 = (
                            copy.deepcopy(new_mem_bw_1),
                            copy.deepcopy(new_dp_info_1),
                        )
                        solution_mem_2, solution_dp_2 = (
                            copy.deepcopy(new_mem_bw_2),
                            copy.deepcopy(new_dp_info_2),
                        )
                        self.visualize_graph(
                            graph_1,
                            os.getcwd()
                            + "/fpga_modeling_reports/partition_graphs/"
                            + self.part_name
                            + "_phase_1",
                        )
                        self.visualize_graph(
                            graph_2,
                            os.getcwd()
                            + "/fpga_modeling_reports/partition_graphs/"
                            + self.part_name
                            + "_phase_2",
                        )

            current_temp *= self.cooling_rate
            # current_temp -= self.cooling_rate
            # current_temp = current_temp/(1 + self.cooling_rate*current_temp)
            print(f"{current_temp:.5e}\t{prev_cost:.5e}", end="\r")

        print(
            f"\n\nLatency: {prev_cost}. volumes/s: {1/prev_cost}.\nSolution 1: Memory IN {list(np.array(solution_mem_1[0]) * self.mem_words_per_cycle)}, Memory OUT {list(np.array(solution_mem_1[1]) * self.mem_words_per_cycle)}\nSolution 2: Memory IN {list(np.array(solution_mem_2[0]) * self.mem_words_per_cycle)}, Memory OUT {list(np.array(solution_mem_2[1]) * self.mem_words_per_cycle)}."
        )
        print(
            "Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}\nPartition Configuration: {}".format(
                solution_dp_1["latency(C)"],
                solution_dp_1["latency(S)"],
                solution_dp_1["GOP/s"],
                solution_dp_1["vols/s"],
                solution_dp_1["DSP"],
                solution_dp_1["BRAM"],
                solution_dp_1["rateIn"],
                solution_dp_1["rateOut"],
                solution_dp_1["depth"],
                solution_dp_1["muls"],
                solution_dp_1["adds"],
                solution_dp_1["memWords"],
                solution_dp_1["memKBs"],
                solution_dp_1["memBoundedIn"],
                solution_dp_1["memBoundedOut"],
                solution_dp_1["config"],
            )
        )
        print(
            "Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}\nPartition Configuration: {}".format(
                solution_dp_2["latency(C)"],
                solution_dp_2["latency(S)"],
                solution_dp_2["GOP/s"],
                solution_dp_2["vols/s"],
                solution_dp_2["DSP"],
                solution_dp_2["BRAM"],
                solution_dp_2["rateIn"],
                solution_dp_2["rateOut"],
                solution_dp_2["depth"],
                solution_dp_2["muls"],
                solution_dp_2["adds"],
                solution_dp_2["memWords"],
                solution_dp_2["memKBs"],
                solution_dp_2["memBoundedIn"],
                solution_dp_2["memBoundedOut"],
                solution_dp_2["config"],
            )
        )
        print("*" * 60)
        return (
            self.mem_words_per_cycle,
            [solution_mem_1, solution_mem_2],
            [solution_dp_1, solution_dp_2],
        )

    def initialize_optimizer(self):
        self.freeze_param = True

        config, mem_bw, _, _ = self.generate_random_config()
        cost, dp_info = self.get_cost(config, mem_bw)
        slowest_nodes = None

        if cost is None:
            start_time = time.time()
            while time.time() - start_time < 90.0:
                # config, mem_bw, _, _ = self.generate_random_config(
                #     seconds_passed=math.ceil(time.time() - start_time)
                # )
                config, mem_bw, _, _ = self.generate_random_config()
                cost, dp_info = self.get_cost(config, mem_bw)

                if cost is not None:
                    slowest_nodes = dp_info["slowestNodes"]
                    break
            if cost is None:
                print("No configuration found after 90 seconds. Aborting...")
                return None, None, None, None, None

        prev_state = config
        prev_cost = cost
        solution_dp = dp_info
        solution_mem = mem_bw

        return prev_state, prev_cost, solution_dp, solution_mem, slowest_nodes

    def run_optimizer(self):
        if self.has_gap() and self.branch_bram_util > self.max_BRAM_util:
            return self.run_optimizer_double_graph()

        best_solution_mem = None
        best_solution_dp = None
        best_latency = 1000
        for _ in range(self.best_of_iter):

            (
                config,
                cost,
                dp_info,
                mem_bw,
                slowest_nodes,
            ) = self.initialize_optimizer()
            if config == None:
                return None, None, None

            prev_state = config
            prev_cost = cost
            solution_dp = dp_info
            solution_mem = mem_bw

            current_temp = self.t_max
            # first_restart, second_restart, third_restart = True, True, True
            count = 0
            print(
                f"Temperature  |  Latency     |   Count   |   Param Count   |   Param %"
            )
            while current_temp > self.t_min:
                # wandb.log({"running_temp": current_temp, "running_latency": prev_cost})
                for _ in range(self.iterationPerTemp):
                    count += 1
                    (
                        new_state,
                        new_mem_bw,
                        param_count,
                        param_perc,
                    ) = self.generate_random_config(
                        neighbours=True,
                        prev_state=prev_state,
                        slowest_nodes=slowest_nodes,
                    )
                    new_cost, new_dp_info = self.get_cost(new_state, new_mem_bw)
                    if new_cost is not None:
                        slowest_nodes = new_dp_info["slowestNodes"]

                    if new_cost is None:
                        continue

                    cost_diff = prev_cost - new_cost
                    if cost_diff >= 0:
                        prev_state = copy.deepcopy(new_state)
                        prev_cost = copy.deepcopy(new_cost)
                        solution_mem, solution_dp = (
                            copy.deepcopy(new_mem_bw),
                            copy.deepcopy(new_dp_info),
                        )
                    else:
                        if random.uniform(0, 1) < math.exp(cost_diff / current_temp):
                            prev_state = copy.deepcopy(new_state)
                            prev_cost = copy.deepcopy(new_cost)
                            solution_mem, solution_dp = (
                                copy.deepcopy(new_mem_bw),
                                copy.deepcopy(new_dp_info),
                            )

                current_temp *= self.cooling_rate
                # if current_temp <= 0.01 and first_restart:
                #     current_temp *= 100
                #     first_restart = False
                # elif current_temp <= 0.001 and second_restart:
                #     current_temp *= 1000
                #     second_restart = False
                # elif current_temp <= 0.0001 and third_restart:
                #     current_temp *= 1000
                #     third_restart = False
                print(
                    f"{current_temp:.5e}\t{prev_cost:.5e}\t{count:5d}\t{param_count:5d}\t\t\t{param_perc:.3f}",
                    end="\r",
                )

            print(
                f"{current_temp:.5e}\t{prev_cost:.5e}\t{count:5d}\t{param_count:5d}\t\t\t{param_perc:.3f}"
            )

            if prev_cost < best_latency:
                best_latency = prev_cost
                best_solution_mem = solution_mem
                best_solution_dp = solution_dp

        print(
            f"\n\nLatency: {best_latency}.\nFinal Memory IN {list(np.array(best_solution_mem[0]) * self.mem_words_per_cycle)}, Memory OUT {list(np.array(best_solution_mem[1]) * self.mem_words_per_cycle)}."
        )
        print(
            "Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={}({:.2f}), BRAM(%)={}({:.2f}), rateIn={}, RateOut={}, Depth={}({}), Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}\nPartition Configuration: {}".format(
                best_solution_dp["latency(C)"],
                best_solution_dp["latency(S)"],
                best_solution_dp["GOP/s"],
                best_solution_dp["vols/s"],
                best_solution_dp["DSP_RAW"],
                best_solution_dp["DSP"],
                best_solution_dp["BRAM_RAW"],
                best_solution_dp["BRAM"],
                best_solution_dp["rateIn"],
                best_solution_dp["rateOut"],
                best_solution_dp["depth"],
                best_solution_dp["branch_depth"],
                best_solution_dp["muls"],
                best_solution_dp["adds"],
                best_solution_dp["memWords"],
                best_solution_dp["memKBs"],
                best_solution_dp["memBoundedIn"],
                best_solution_dp["memBoundedOut"],
                best_solution_dp["config"],
            )
        )
        print("*" * 60)
        return self.mem_words_per_cycle, [best_solution_mem], [best_solution_dp]

    def get_cost(
        self,
        config,
        mem_bw,
        target_graph=None,
        branch_mem_update=None,
        mem_in_conns=[],
        mem_out_conns=[],
    ):
        """
        We should be able to choose whether we want to optimize for latency or throughput
        """
        if target_graph is None:
            graph = self.graph.copy()
        else:
            graph = target_graph
        if branch_mem_update is None:
            branch_mem = self.branch_mem
        else:
            branch_mem = branch_mem_update

        comb_config = {}
        for k, v in config.items():
            if v["op_type"] == "GlobalAveragePool":
                comb_config[k] = [v["coarse_inout"]]
            elif v["op_type"] == "Conv":
                comb_config[k] = [v["fine"], v["coarse_in"], v["coarse_out"]]
            elif v["op_type"] == "Activation":
                comb_config[k] = [v["coarse_inout"]]
            elif v["op_type"] == "ElementWise":
                comb_config[k] = [v["coarse_inout"]]
            elif v["op_type"] == "BatchNormalization":
                comb_config[k] = [v["coarse_inout"]]
            elif v["op_type"] == "Gemm":
                comb_config[k] = [v["coarse_in"], v["coarse_out"]]
            else:
                assert False, "Not supported layer"

        mem_in = mem_in_conns
        mem_out = mem_out_conns
        read_points, write_points = self.add_off_chip_connections(
            graph, mem_in, mem_out
        )
        dp_info = self.partition_composer.get_design_point(
            graph.copy(),
            comb_config,
            mem_bw[0],
            mem_bw[1],
            read_points,
            write_points,
            gap_approx=self.gap_approx,
            branch_mem=branch_mem,
        )
        # self.visualize_graph(graph, os.getcwd() + '/fpga_modeling_reports/partition_graphs/' + self.part_name + '_int')
        if dp_info["config"]:
            return dp_info["latency(S)"], dp_info
            # return -dp_info['GOP/s'], dp_info
        return None, None

    @staticmethod
    def visualize_graph(graph, path):
        PG = nx.nx_pydot.to_pydot(graph)
        PG.write_png(path + ".png")

    def add_off_chip_connections(self, graph, in_connections, out_connections):
        read_points = []
        write_points = []

        mem_in_count = 1
        mem_out_count = 1
        for n in graph.nodes():
            edges_in = graph.in_edges(n)
            edges_out = graph.out_edges(n)
            if not edges_in:
                input_node = n
            if not edges_out:
                output_node = n

        if not in_connections and not out_connections:
            read_points.append(input_node)
            self.add_node_to_position(
                G=graph,
                new_node="Mem_in{}".format(mem_in_count),
                connect_node=input_node,
                connect_pos="pre",
            )
            mem_in_count += 1
            write_points.append(output_node)
            self.add_node_to_position(
                G=graph,
                new_node="Mem_out{}".format(mem_out_count),
                connect_node=output_node,
                connect_pos="post",
            )
            mem_out_count += 1

        for con_in in in_connections:
            self.add_node_to_position(
                G=graph,
                new_node="Mem_in{}".format(mem_in_count),
                connect_node=con_in,
                connect_pos="pre",
            )
            read_points.append(con_in)
            mem_in_count += 1

        for con_out in out_connections:
            self.add_node_to_position(
                G=graph,
                new_node="Mem_out{}".format(mem_out_count),
                connect_node=con_out,
                connect_pos="post",
            )
            write_points.append(con_out)
            mem_out_count += 1

        # if self.gap_approx:
        #     next_nodes = []
        #     gap_nodes = []
        #     for n in graph.nodes():
        #         if graph.nodes[n]['type'] == 'GlobalAveragePool':
        #             next_nodes.append(list(graph.successors(n))[0])
        #             gap_nodes.append(n)
        #             graph.remove_edge(n, list(graph.successors(n))[0])

        #     for n, g in zip(next_nodes, gap_nodes):
        #         read_points.append(n)
        #         self.add_node_to_position(G=graph, new_node='Mem_in{}'.format(mem_in_count), connect_node=n, connect_pos='pre')
        #         mem_in_count += 1
        #         write_points.append(g)
        #         self.add_node_to_position(G=graph, new_node='Mem_out{}'.format(mem_out_count), connect_node=g, connect_pos='post')
        #         mem_out_count += 1

        return read_points, write_points

    @staticmethod
    def add_node_to_position(
        G, new_node, connect_node, connect_pos, is_input=False, is_output=False
    ):
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
                        extra_inputs = [
                            n for n in G.predecessors(edge[1]) if not n == node
                        ]
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

    @staticmethod
    def get_mem_bw_feasible(n_in=0, n_out=0, gap_approx=False):
        # if gap_approx:
        #     n_in += 1
        #     n_out += 1

        rand_vals = []
        for i in range(n_in + n_out):
            rand_vals.append(random.randint(1, 100))

        total_sum = np.sum(np.array(rand_vals))

        perc = []
        for i in range(n_in + n_out):
            perc.append(rand_vals[i] / total_sum)

        assert math.isclose(
            np.sum(np.array(perc)), 1.0
        ), "Sum of mem_in_1_perc, mem_in_2_perc and mem_out_perc should be 1"

        perc_in = perc[:n_in]
        perc_out = perc[n_in:]

        return perc_in, perc_out

    @staticmethod
    def fix_inconsistent_config(config, graph):
        new_config = config.copy()
        for node in config.keys():
            if not graph.has_node(node):
                del new_config[node]
        for node in graph.nodes():
            if not node in config.keys():
                op_type = graph.nodes[node]["type"]
                hw = graph.nodes[node]["hw"]
                if isinstance(hw, GAPLayer):
                    channels = hw.channels
                    coarse_inout_feasible = utils.get_factors(channels)
                    coarse_inout_factor = (
                        random.choice(coarse_inout_feasible) / channels
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
                elif isinstance(hw, Convolutional3DLayer):
                    channels = hw.channels
                    filters = hw.filters
                    kernel_size = hw.kernel_shape
                    coarse_in_feasible = utils.get_factors(channels)
                    coarse_out_feasible = utils.get_factors(filters)
                    fine_feasible = utils.get_fine_feasible(kernel_size)
                    coarse_in_factor = random.choice(coarse_in_feasible) / channels
                    coarse_out_factor = random.choice(coarse_out_feasible) / filters
                    fine_factor = random.choice(fine_feasible) / np.prod(
                        np.array(kernel_size)
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "fine": fine_factor,
                        "coarse_in": coarse_in_factor,
                        "coarse_out": coarse_out_factor,
                    }
                elif isinstance(hw, ActivationLayer):
                    channels = hw.channels
                    coarse_inout_feasible = utils.get_factors(channels)
                    coarse_inout_factor = (
                        random.choice(coarse_inout_feasible) / channels
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
                elif isinstance(hw, ElementWiseLayer):
                    channels = hw.channels_1
                    coarse_inout_feasible = utils.get_factors(channels)
                    coarse_inout_factor = (
                        random.choice(coarse_inout_feasible) / channels
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
                elif isinstance(hw, BatchNorm3DLayer):
                    channels = hw.channels
                    coarse_inout_feasible = utils.get_factors(channels)
                    coarse_inout_factor = (
                        random.choice(coarse_inout_feasible) / channels
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
                elif isinstance(hw, SqueezeExcitationLayer):
                    assert False, "Not supported layer (SqueezeExcitationLayer)"
                elif isinstance(hw, FCLayer):
                    dim_in = hw.dim_in
                    dim_out = hw.dim_out
                    coarse_in_feasible = utils.get_factors(dim_in)
                    coarse_out_feasible = utils.get_factors(dim_out)
                    coarse_in_factor = random.choice(coarse_in_feasible) / dim_in
                    coarse_out_factor = random.choice(coarse_out_feasible) / dim_out
                    new_config[node] = {
                        "op_type": op_type,
                        "coarse_in": coarse_in_factor,
                        "coarse_out": coarse_out_factor,
                    }
                elif isinstance(hw, MemoryNode):
                    continue
                else:
                    assert False, "Not supported layer"
        return new_config

    def generate_random_config(
        self,
        target_graph=None,
        neighbours=False,
        prev_state=None,
        slowest_nodes=None,
        seconds_passed=None,
        param_perc=0.3,
        n_in=1,
        n_out=1,
    ):
        if target_graph is None:
            graph = self.graph.copy()
        else:
            graph = target_graph

        # if neighbours:
        #     if not self.freeze_param:
        #         self.param_changes += 1
        #         param_perc = math.cos(self.param_changes/1500)
        #         if param_perc > 0.6:
        #             param_perc = 0.6
        #         elif param_perc < 0.3:
        #             param_perc = 0.3
        #             self.freeze_param = True
        #     else:
        #         param_perc = param_perc

        #     number_of_new_configs = math.ceil(graph.order() * param_perc)

        #     # without replacement
        #     config_nodes = random.sample(list(graph.nodes), number_of_new_configs)
        #     # with replacement
        #     # config_nodes = random.choices(list(graph.nodes), k=number_of_new_configs)

        #     config = prev_state
        # else:
        #     config_nodes = list(graph.nodes)
        #     config = {}

        neighbours = False

        config_nodes = list(graph.nodes)
        if slowest_nodes:
            config = prev_state.copy()
        else:
            config = {}

        for node in config_nodes:
            if slowest_nodes is not None and node not in slowest_nodes:
                continue

            op_type = graph.nodes[node]["type"]
            hw = graph.nodes[node]["hw"]
            if isinstance(hw, GAPLayer):
                channels = hw.channels
                coarse_inout_feasible = utils.get_factors(
                    channels, sec_passed=seconds_passed
                )
                coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove("op_type")
                    apply_transform = random.choice(transformations)
                    if apply_transform == "coarse_inout":
                        config[node][apply_transform] = coarse_inout_factor
                else:
                    config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
            elif isinstance(hw, Convolutional3DLayer):
                channels = hw.channels
                filters = hw.filters
                kernel_size = hw.kernel_shape
                coarse_in_feasible = utils.get_factors(
                    channels, sec_passed=seconds_passed
                )
                coarse_out_feasible = utils.get_factors(
                    filters, sec_passed=seconds_passed
                )
                fine_feasible = utils.get_fine_feasible(kernel_size)
                coarse_in_factor = random.choice(coarse_in_feasible) / channels
                coarse_out_factor = random.choice(coarse_out_feasible) / filters
                fine_factor = random.choice(fine_feasible) / np.prod(
                    np.array(kernel_size)
                )
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove("op_type")
                    apply_transform = random.choice(transformations)
                    if apply_transform == "coarse_in":
                        config[node][apply_transform] = coarse_in_factor
                    elif apply_transform == "coarse_out":
                        config[node][apply_transform] = coarse_out_factor
                    elif apply_transform == "fine":
                        config[node][apply_transform] = fine_factor
                else:
                    config[node] = {
                        "op_type": op_type,
                        "fine": fine_factor,
                        "coarse_in": coarse_in_factor,
                        "coarse_out": coarse_out_factor,
                    }
            elif isinstance(hw, ActivationLayer):
                channels = hw.channels
                coarse_inout_feasible = utils.get_factors(
                    channels, sec_passed=seconds_passed
                )
                coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove("op_type")
                    apply_transform = random.choice(transformations)
                    if apply_transform == "coarse_inout":
                        config[node][apply_transform] = coarse_inout_factor
                else:
                    config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
            elif isinstance(hw, ElementWiseLayer):
                channels = hw.channels_1
                coarse_inout_feasible = utils.get_factors(
                    channels, sec_passed=seconds_passed
                )
                coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove("op_type")
                    apply_transform = random.choice(transformations)
                    if apply_transform == "coarse_inout":
                        config[node][apply_transform] = coarse_inout_factor
                else:
                    config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
            elif isinstance(hw, BatchNorm3DLayer):
                channels = hw.channels
                coarse_inout_feasible = utils.get_factors(
                    channels, sec_passed=seconds_passed
                )
                coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove("op_type")
                    apply_transform = random.choice(transformations)
                    if apply_transform == "coarse_inout":
                        config[node][apply_transform] = coarse_inout_factor
                else:
                    config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
            elif isinstance(hw, SqueezeExcitationLayer):
                assert False, "Not supported layer (SqueezeExcitationLayer)"
            elif isinstance(hw, FCLayer):
                dim_in = hw.dim_in
                dim_out = hw.dim_out
                coarse_in_feasible = utils.get_factors(
                    dim_in, sec_passed=seconds_passed
                )
                coarse_out_feasible = utils.get_factors(
                    dim_out, sec_passed=seconds_passed
                )
                coarse_in_factor = random.choice(coarse_in_feasible) / dim_in
                coarse_out_factor = random.choice(coarse_out_feasible) / dim_out
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove("op_type")
                    apply_transform = random.choice(transformations)
                    if apply_transform == "coarse_in":
                        config[node][apply_transform] = coarse_in_factor
                    elif apply_transform == "coarse_out":
                        config[node][apply_transform] = coarse_out_factor
                else:
                    config[node] = {
                        "op_type": op_type,
                        "coarse_in": coarse_in_factor,
                        "coarse_out": coarse_out_factor,
                    }
            elif isinstance(hw, MemoryNode):
                continue
            else:
                assert False, "Not supported layer"

        mem_config_in, mem_config_out = self.get_mem_bw_feasible(
            n_in=n_in, n_out=n_out, gap_approx=self.gap_approx
        )

        return config, [mem_config_in, mem_config_out], self.param_changes, param_perc

    # TODO: Revise that to follow the changes on run_optimizer
    def run_optimizer_layer(self, layer):
        config, mem_bw = self.generate_random_config_layer(layer)
        cost, dp_info = self.get_cost_layer(config, mem_bw, layer)

        if cost is None:
            while cost is None:
                config, mem_bw = self.generate_random_config_layer(layer)
                cost, dp_info = self.get_cost_layer(config, mem_bw, layer)

        prev_state = config
        solution_dp = dp_info
        prev_cost = cost

        current_temp = self.t_max

        print(f"Temperature  |  Latency")
        while current_temp > self.t_min:

            for i in range(self.iterationPerTemp):
                new_state, new_mem_bw = self.generate_random_config_layer(layer)
                new_cost, new_dp_info = self.get_cost_layer(
                    new_state, new_mem_bw, layer
                )

                if new_cost is None:
                    continue

                cost_diff = prev_cost - new_cost
                if cost_diff >= 0:
                    prev_state = copy.deepcopy(new_state)
                    prev_cost = copy.deepcopy(new_cost)
                    solution_mem, solution_dp = (
                        copy.deepcopy(new_mem_bw),
                        copy.deepcopy(new_dp_info),
                    )
                else:
                    if random.uniform(0, 1) < math.exp(
                        (cost_diff / (self.k * current_temp))
                    ):
                        prev_state = copy.deepcopy(new_state)
                        prev_cost = copy.deepcopy(new_cost)
                        solution_mem, solution_dp = (
                            copy.deepcopy(new_mem_bw),
                            copy.deepcopy(new_dp_info),
                        )

            current_temp *= self.cooling_rate
            print(f"{current_temp:.5e}\t{prev_cost:.5e}", end="\r")

        print(
            f"\n\nLatency: {prev_cost}.\nFinal Memory IN {list(np.array(solution_mem[0]) * self.mem_words_per_cycle)}, Memory OUT {list(np.array(solution_mem[1]) * self.mem_words_per_cycle)}."
        )
        print(
            "Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={}({:.2f}), BRAM(%)={}({:.2f}), rateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}\nPartition Configuration: {}".format(
                solution_dp["latency(C)"],
                solution_dp["latency(S)"],
                solution_dp["GOP/s"],
                solution_dp["vols/s"],
                int((solution_dp["DSP"] / 100) * self.graph.nodes[layer]["hw"].dsp),
                solution_dp["DSP"],
                int((solution_dp["BRAM"] / 100) * self.graph.nodes[layer]["hw"].bram),
                solution_dp["BRAM"],
                solution_dp["rateIn"],
                solution_dp["rateOut"],
                solution_dp["depth"],
                solution_dp["muls"],
                solution_dp["adds"],
                solution_dp["memWords"],
                solution_dp["memKBs"],
                solution_dp["memBoundedIn"],
                solution_dp["memBoundedOut"],
                solution_dp["config"],
            )
        )
        print("*" * 60)

    def get_cost_layer(self, config, mem_bw, layer):
        hw = self.graph.nodes[layer]["hw"]
        if isinstance(hw, GAPLayer):
            dp_info = hw.get_design_point(
                config[0],
                hw.mem_words_per_cycle * mem_bw[0][0],
                hw.mem_words_per_cycle * mem_bw[1][0],
            )
        elif isinstance(hw, Convolutional3DLayer):
            dp_info = hw.get_design_point(
                config[0],
                config[1],
                config[2],
                hw.mem_words_per_cycle * mem_bw[0][0],
                hw.mem_words_per_cycle * mem_bw[1][0],
            )
        elif isinstance(hw, ActivationLayer):
            dp_info = hw.get_design_point(
                config[0],
                hw.mem_words_per_cycle * mem_bw[0][0],
                hw.mem_words_per_cycle * mem_bw[1][0],
            )
        elif isinstance(hw, ElementWiseLayer):
            dp_info = hw.get_design_point(
                config[0],
                hw.mem_words_per_cycle * mem_bw[0][0],
                hw.mem_words_per_cycle * mem_bw[0][1],
                hw.mem_words_per_cycle * mem_bw[1][0],
            )
        elif isinstance(hw, BatchNorm3DLayer):
            dp_info = hw.get_design_point(
                config[0],
                hw.mem_words_per_cycle * mem_bw[0][0],
                hw.mem_words_per_cycle * mem_bw[1][0],
            )
        elif isinstance(hw, SqueezeExcitationLayer):
            assert False, "Not supported layer (SqueezeExcitationLayer)"
        elif isinstance(hw, FCLayer):
            dp_info = hw.get_design_point(
                config[0],
                config[1],
                hw.mem_words_per_cycle * mem_bw[0][0],
                hw.mem_words_per_cycle * mem_bw[1][0],
            )
        else:
            assert False, "Not supported layer"

        if dp_info["config"]:
            return dp_info["latency(S)"], dp_info
            # return -dp_info['GOP/s'], dp_info
        return None, None

    def generate_random_config_layer(self, l):
        config = []
        hw = self.graph.nodes[l]["hw"]
        if isinstance(hw, GAPLayer):
            channels = hw.channels
            filters = hw.filters
            coarse_inout_feasible = utils.get_factors(channels)
            coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
            config = [coarse_inout_factor]
        elif isinstance(hw, Convolutional3DLayer):
            channels = hw.channels
            filters = hw.filters
            kernel_size = hw.kernel_shape
            coarse_in_feasible = utils.get_factors(channels)
            coarse_out_feasible = utils.get_factors(filters)
            fine_feasible = utils.get_fine_feasible(kernel_size)
            coarse_in_factor = random.choice(coarse_in_feasible) / channels
            coarse_out_factor = random.choice(coarse_out_feasible) / filters
            fine_factor = random.choice(fine_feasible) / np.prod(np.array(kernel_size))
            config = [coarse_in_factor, coarse_out_factor, fine_factor]
        elif isinstance(hw, ActivationLayer):
            channels = hw.channels
            coarse_inout_feasible = utils.get_factors(channels)
            coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
            config = [coarse_inout_factor]
        elif isinstance(hw, ElementWiseLayer):
            channels = hw.channels_1
            coarse_inout_feasible = utils.get_factors(channels)
            coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
            config = [coarse_inout_factor]
        elif isinstance(hw, BatchNorm3DLayer):
            channels = hw.channels
            coarse_inout_feasible = utils.get_factors(channels)
            coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
            config = [coarse_inout_factor]
        elif isinstance(hw, SqueezeExcitationLayer):
            assert False, "Not supported layer (SqueezeExcitationLayer)"
        elif isinstance(hw, FCLayer):
            dim_in = hw.dim_in
            dim_out = hw.dim_out
            coarse_in_feasible = utils.get_factors(dim_in)
            coarse_out_feasible = utils.get_factors(dim_out)
            coarse_in_factor = random.choice(coarse_in_feasible) / dim_in
            coarse_out_factor = random.choice(coarse_out_feasible) / dim_out
            config = [coarse_in_factor, coarse_out_factor]
        else:
            assert False, "Not supported layer"

        if isinstance(hw, ElementWiseLayer):
            mem_config_in, mem_config_out = self.get_mem_bw_feasible(n_in=2, n_out=1)
        else:
            mem_config_in, mem_config_out = self.get_mem_bw_feasible(n_in=1, n_out=1)

        return config, [mem_config_in, mem_config_out]

    def validate_building_blocks_setup(self, bblocks: list) -> bool:
        """
        Validate the building blocks setup by producing a valid scedulilng of the building blocks that
        can execute the complete network graph.
        """

        nodes = utils.get_nodes_sorted(self.graph)
        for n in nodes:
            bb = self.graph.nodes[n]["hw_type"]
            if bb in bblocks:
                continue
            else:
                _logger.critical(
                    msg=f"Hardware building block {bb} not found in existing building blocks list"
                )
                return False

        return True

    def generate_building_blocks(self) -> list:
        """
        Generate a set of hardware building blocks that can be used to execute the graph's workload.

        Returns:
            dict: bb_setup
        """

        # TODO: implement a fuction that generates building blocks comprising of a single layer or a set of layers.
        bblocks = ["Conv3DDw", "Conv3DPw", "Activation", "GlobalAveragePool"]
        assert self.validate_building_blocks_setup(
            bblocks
        ), "Invalid building blocks setup. Cannot find a valid scheduling."

        return bblocks

    def generate_building_blocks_config(
        self, bblocks: list, previous_config: dict = None
    ) -> dict:
        """
        Generate a configuration for each building block based on the min and max channels and filters values
        from all its instances across the graph of the network.

        Returns:
            dict: bb_setup
        """

        bb_setup = dict()
        total_dsp = 0
        total_bram = 0
        for bb in bblocks:
            if not bb in bb_setup.keys():
                bb_setup[bb] = dict()

            # print(f"Generating configuration for {bb}")
            shape_in, shape_out = utils.get_random_shape(
                self.graph, bb, previous_config=previous_config
            )
            _, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim = shape_in
            (
                _,
                channels_out_dim,
                depth_out_dim,
                height_out_dim,
                width_out_dim,
            ) = shape_out

            if bb == "Conv3DDw":
                bb_descriptor = {
                    "operation": "Conv",
                    "shape_in": [
                        [1, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim]
                    ],
                    "shape_out": [
                        1,
                        channels_in_dim,
                        depth_out_dim,
                        height_out_dim,
                        width_out_dim,
                    ],
                    "kernel": [channels_in_dim, 1, 3, 3, 3],
                    "bias": [channels_in_dim],
                    "padding": [1, 1, 1],
                    "stride": [1, 1, 1],
                    "groups": channels_in_dim,
                    "dilation": [1, 1, 1],
                    "branching": False,
                }
            elif bb == "Conv3DPw":
                bb_descriptor = {
                    "operation": "Conv",
                    "shape_in": [
                        [1, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim]
                    ],
                    "shape_out": [
                        1,
                        channels_out_dim,
                        depth_out_dim,
                        height_out_dim,
                        width_out_dim,
                    ],
                    "kernel": [channels_out_dim, channels_in_dim, 1, 1, 1],
                    "bias": [channels_out_dim],
                    "padding": [0, 0, 0],
                    "stride": [1, 1, 1],
                    "groups": 1,
                    "dilation": [1, 1, 1],
                    "branching": False,
                }
            elif bb == "Activation":
                bb_descriptor = {
                    "operation": "Activation",
                    "shape_in": [
                        [1, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim]
                    ],
                    "shape_out": [
                        1,
                        channels_in_dim,
                        depth_in_dim,
                        height_in_dim,
                        width_in_dim,
                    ],
                }
            elif bb == "GlobalAveragePool":
                bb_descriptor = {
                    "operation": "GlobalAveragePool",
                    "shape_in": [
                        [1, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim]
                    ],
                    "shape_out": [
                        1,
                        channels_in_dim,
                        1,
                        1,
                        1,
                    ],
                }

            if bb in ["Conv3DDw", "Conv3DPw"]:
                bb_setup[bb]["hw"] = Convolutional3DLayer(
                    self.max_DSP_util, self.max_BRAM_util, bb_descriptor
                )
            elif bb == "Activation":
                bb_setup[bb]["hw"] = ActivationLayer(
                    self.max_DSP_util, self.max_BRAM_util, bb_descriptor
                )
            elif bb == "GlobalAveragePool":
                bb_setup[bb]["hw"] = GAPLayer(
                    self.max_DSP_util, self.max_BRAM_util, bb_descriptor
                )

            dsp_util, bram_util = 100, 100
            stop_counter = 0
            while dsp_util > (self.max_DSP_util - total_dsp) or bram_util > (
                self.max_BRAM_util - total_bram
            ):

                if bb in ["Conv3DDw", "Conv3DPw"]:
                    coarse_in = (
                        np.random.choice(np.arange(channels_in_dim) + 1, replace=False)
                        / channels_in_dim
                    )
                    coarse_out = (
                        np.random.choice(np.arange(channels_out_dim) + 1, replace=False)
                        / channels_out_dim
                    )
                    if bb == "Conv3DDw":
                        coarse_out = coarse_in

                    dsp_util, bram_util = bb_setup[bb]["hw"].get_resource_util(
                        f_fine=1, f_coarseIn=coarse_in, f_coarseOut=coarse_out
                    )
                elif bb in ["Activation", "GlobalAveragePool"]:
                    coarse_inout = (
                        np.random.choice(np.arange(channels_in_dim) + 1, replace=False)
                        / channels_in_dim
                    )
                    dsp_util, bram_util = bb_setup[bb]["hw"].get_resource_util(
                        f_coarse_inout=coarse_inout
                    )
                stop_counter += 1
                if stop_counter > 50:
                    return None

            if bb in ["Conv3DDw", "Conv3DPw"]:
                bb_setup[bb]["HINT_shape_in"] = [
                    1,
                    channels_in_dim,
                    depth_in_dim,
                    height_in_dim,
                    width_in_dim,
                ]
                bb_setup[bb]["coarse_in"] = coarse_in
                bb_setup[bb]["streams_in"] = math.ceil(coarse_in * channels_in_dim)
                bb_setup[bb]["interleaving_in"] = 1 // coarse_in
                bb_setup[bb]["interleaving_pad_in"] = channels_in_dim % math.ceil(
                    channels_in_dim * coarse_in
                )
                bb_setup[bb]["HINT_shape_out"] = [
                    1,
                    channels_out_dim,
                    depth_out_dim,
                    height_out_dim,
                    width_out_dim,
                ]
                bb_setup[bb]["coarse_out"] = coarse_out
                bb_setup[bb]["streams_out"] = math.ceil(coarse_out * channels_out_dim)
                bb_setup[bb]["interleaving_out"] = 1 // coarse_out
                bb_setup[bb]["interleaving_pad_out"] = channels_out_dim % math.ceil(
                    channels_out_dim * coarse_out
                )
            elif bb in ["Activation", "GlobalAveragePool"]:
                bb_setup[bb]["HINT_shape_in"] = [
                    1,
                    channels_in_dim,
                    depth_in_dim,
                    height_in_dim,
                    width_in_dim,
                ]
                bb_setup[bb]["coarse_inout"] = coarse_inout
                bb_setup[bb]["streams_inout"] = math.ceil(
                    coarse_inout * channels_in_dim
                )
                bb_setup[bb]["interleaving_inout"] = 1 // coarse_inout
                bb_setup[bb]["interleaving_pad_inout"] = channels_in_dim % math.ceil(
                    channels_in_dim * coarse_inout
                )
            bb_setup[bb]["DSP_util"] = dsp_util
            bb_setup[bb]["BRAM_util"] = bram_util
            total_dsp += dsp_util
            total_bram += bram_util

        return bb_setup

    def get_cost_e2e(self, bblocks_config: dict) -> float:
        if bblocks_config is None:
            return None, None, None, None, None
        cost = 0.0
        avg_BW = 0.0
        scheduling = {}
        for node in self.graph.nodes:
            bb_type = self.graph.nodes[node]["hw_type"]
            hw = self.graph.nodes[node]["hw"]

            if len(hw.input_shape) < 5:
                shape_calls = 1
            else:
                # TODO: Search how to deal with cases where the output dimensions are also altered? Like in convolutional layers with stride > 1.
                shape_calls = math.ceil(
                    hw.input_shape[2]
                    / bblocks_config[bb_type]["HINT_shape_in"][2]
                    * hw.input_shape[3]
                    / bblocks_config[bb_type]["HINT_shape_in"][3]
                    * hw.input_shape[4]
                    / bblocks_config[bb_type]["HINT_shape_in"][4]
                )
            if bb_type in ["Conv3DDw", "Conv3DPw"]:
                in_calls = math.ceil(
                    hw.input_shape[1]
                    / (
                        bblocks_config[bb_type]["streams_in"]
                        * bblocks_config[bb_type]["interleaving_in"]
                    )
                )
                out_calls = math.ceil(
                    hw.output_shape[1]
                    / (
                        bblocks_config[bb_type]["streams_out"]
                        * bblocks_config[bb_type]["interleaving_out"]
                    )
                )
                total_block_calls = in_calls * out_calls * shape_calls
            elif bb_type in ["Activation", "GlobalAveragePool"]:
                inout_calls = math.ceil(
                    hw.input_shape[1]
                    / (
                        bblocks_config[bb_type]["streams_inout"]
                        * bblocks_config[bb_type]["interleaving_inout"]
                    )
                )
                total_block_calls = inout_calls * shape_calls

            # TODO: Update the shapes of the bblocks_config[bb_type]["hw"] to account for the overlapping regions because of feature map tilling
            if bb_type in ["Conv3DDw", "Conv3DPw"]:
                r = bblocks_config[bb_type]["hw"].get_design_point(
                    f_fine=1,
                    f_coarseIn=bblocks_config[bb_type]["coarse_in"],
                    f_coarseOut=bblocks_config[bb_type]["coarse_out"],
                    mem_bw_in=bblocks_config[bb_type]["hw"].mem_words_per_cycle / 2,
                    mem_bw_out=bblocks_config[bb_type]["hw"].mem_words_per_cycle / 2,
                )
            elif bb_type == "Activation":
                bblocks_config[bb_type]["hw"].activation_type = hw.activation_type
                r = bblocks_config[bb_type]["hw"].get_design_point(
                    coarse_inout=bblocks_config[bb_type]["coarse_inout"],
                    mem_bw_in=bblocks_config[bb_type]["hw"].mem_words_per_cycle / 2,
                    mem_bw_out=bblocks_config[bb_type]["hw"].mem_words_per_cycle / 2,
                )
                bblocks_config[bb_type]["hw"].activation_type = "Activation"
            elif bb_type == "GlobalAveragePool":
                r = bblocks_config[bb_type]["hw"].get_design_point(
                    coarse_inout=bblocks_config[bb_type]["coarse_inout"],
                    mem_bw_in=bblocks_config[bb_type]["hw"].mem_words_per_cycle / 2,
                    mem_bw_out=bblocks_config[bb_type]["hw"].mem_words_per_cycle / 2,
                )
            # TODO: Revert back the shapes of the bblocks_config[bb_type]["hw"]

            bblocks_config[bb_type]["MemBw_util"] = r["memBwUtil"]

            latency = r["latency(S)"] * math.ceil(total_block_calls)
            avg_BW += r["memBwUtil"]
            assert math.ceil(total_block_calls) > 0, "Zero calls aborting..."
            if bb_type in ["Conv3DDw", "Conv3DPw"]:
                scheduling[node] = {
                    "Block Type": bb_type,
                    "Times Called (channels)": in_calls,
                    "Times Called (filters)": out_calls,
                    "Times Called (tiles)": math.ceil(shape_calls),
                    "Latency": latency,
                    "Base Latency": r["latency(S)"],
                }
            elif bb_type in ["Activation", "GlobalAveragePool"]:
                scheduling[node] = {
                    "Block Type": bb_type,
                    "Times Called (channels)": inout_calls,
                    "Times Called (tiles)": math.ceil(shape_calls),
                    "Latency": latency,
                    "Base Latency": r["latency(S)"],
                }
            cost += latency

        avg_BW = avg_BW / len(self.graph.nodes)
        final_DSP = 0
        final_BRAM = 0
        for bb in bblocks_config:
            final_DSP += bblocks_config[bb]["DSP_util"]
            final_BRAM += bblocks_config[bb]["BRAM_util"]
        return cost, scheduling, final_DSP, final_BRAM, avg_BW

    def run_optimizer_latency(self):
        # wandb.config.update({"config_generation": "mape_c70_w100"})

        bblocks = self.generate_building_blocks()
        bblocks_config = self.generate_building_blocks_config(bblocks)

        cost, scheduling, dsp_util, bram_util, bw_util = self.get_cost_e2e(
            bblocks_config
        )

        if cost is None:
            for _ in range(100):
                bblocks_config = self.generate_building_blocks_config(bblocks)
                cost, scheduling, dsp_util, bram_util, bw_util = self.get_cost_e2e(
                    bblocks_config
                )
                if cost is not None:
                    break
            if cost is None:
                print("No configuration found in 100 iterations. Exiting...")
                return None

        prev_state = bblocks_config
        prev_scheduling = scheduling
        prev_cost = cost
        prev_dsp = dsp_util
        prev_bram = bram_util
        prev_bw = bw_util

        current_temp = self.t_max
        print(f"Temperature  |  Latency    ")
        while current_temp > self.t_min:
            if not self.wandb_config == None:
                log_dict = {}
                log_dict["temperature"] = current_temp
                log_dict["latency"] = prev_cost
                log_dict["dsp_util"] = prev_dsp
                log_dict["bram_util"] = prev_bram
                log_dict["mem_bw_util"] = prev_bw
                wandb.log(log_dict)

            for _ in range(self.iterationPerTemp):
                new_state = self.generate_building_blocks_config(
                    bblocks, previous_config=None
                )
                (
                    new_cost,
                    new_scheduling,
                    dsp_util,
                    bram_util,
                    bw_util,
                ) = self.get_cost_e2e(new_state)

                if new_cost is None:
                    continue

                cost_diff = prev_cost - new_cost
                if cost_diff >= 0:
                    prev_state = copy.deepcopy(new_state)
                    prev_cost = copy.deepcopy(new_cost)
                    prev_dsp = copy.deepcopy(dsp_util)
                    prev_bram = copy.deepcopy(bram_util)
                    prev_bw = copy.deepcopy(bw_util)
                    prev_scheduling = copy.deepcopy(new_scheduling)
                else:
                    if random.uniform(0, 1) < math.exp(cost_diff / current_temp):
                        prev_state = copy.deepcopy(new_state)
                        prev_cost = copy.deepcopy(new_cost)
                        prev_dsp = copy.deepcopy(dsp_util)
                        prev_bram = copy.deepcopy(bram_util)
                        prev_bw = copy.deepcopy(bw_util)
                        prev_scheduling = copy.deepcopy(new_scheduling)

            current_temp *= self.cooling_rate
            print(
                f"{current_temp:.5e}\t{prev_cost:.5e}",
                end="\r",
            )

        print(f"{current_temp:.5e}\t{prev_cost:.5e}")
        print(f"Optimization finished. Block setup: {prev_state}")
        print(f"Final per layer configuration: {prev_scheduling}")
        if not self.wandb_config == None:
            artifact = wandb.Artifact("scheduling", type="json")
            with artifact.new_file("scheduling.json") as f:
                json.dump(prev_scheduling, f)
            wandb.log_artifact(artifact)
