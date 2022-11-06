import copy
import json
import logging
import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import scipy.constants as sc

import wandb
from fpga_hart import _logger
from fpga_hart.layers.activation import ActivationLayer
from fpga_hart.layers.base_layer import BaseLayer
from fpga_hart.layers.batchnorm_3d import BatchNorm3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise import ElementWiseLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap import GAPLayer
from fpga_hart.layers.memory_interface import MemoryNode
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.layers.squeeze_excitation import SqueezeExcitationLayer
from fpga_hart.optimizer.optimizer_helper import check_partition_fitting
from fpga_hart.partitions.partition_compose import PartitionComposer
from fpga_hart.utils import utils
from fpga_hart.utils.graph_manipulation import (add_off_chip_connections,
                                                get_input_nodes,
                                                get_output_nodes, split_graph,
                                                visualize_graph)


class SimulatedAnnealing(BaseLayer):
    def __init__(
        self,
        graph,
        config,
        branch_mem=0,
        partition_name="",
        gap_approx=False,
        cnn_model_name="",
        enable_wandb=False,
    ):
        self.cnn_model_name = cnn_model_name
        self.config = config
        self.enable_wandb = enable_wandb
        super().__init__(
            max_DSP_util = self.config.max_dsp_util,
            max_BRAM_util = self.config.max_bram_util,
        )
        # _logger.setLevel(level=logging.DEBUG)

        self.gap_approx = gap_approx
        self.part_name = partition_name

        self.graph = graph

        mem_kb = (branch_mem * self.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.bram_Kbytes)
        self.branch_bram_util = (mem_bram / self.bram) * 100
        self.branch_mem = branch_mem

        # Simulate Annealing Variables
        self.k = sc.Boltzmann
        self.t_min = self.config.simulatedAnnealing["t_min"]
        self.t_max = self.config.simulatedAnnealing["t_max"]
        self.cooling_rate = self.config.simulatedAnnealing["cooling_rate"]
        self.iterationPerTemp = self.config.simulatedAnnealing["iterationPerTemp"]
        self.best_of_iter = self.config.simulatedAnnealing["best_of_iter"]
        self.block_gen = self.config.bblock_generation
        self.bblock_keep_percentage = self.config.bblock_keep_percentage
        self.use_arbitrary_shape = self.config.use_arbitrary_shape
        self.use_previous_config = self.config.use_previous_config
        self.chan_dist_thresh = self.config.chan_dist_thresh
        self.depth_dist_thresh = self.config.depth_dist_thresh
        self.height_dist_thresh = self.config.height_dist_thresh

        self.param_changes = 0
        self.freeze_param = False

        self.partition_composer = PartitionComposer(
            max_DSP_util=self.max_DSP_util, max_BRAM_util=self.max_BRAM_util
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
        ) = split_graph(
            self.graph, self.word_bytes, self.bram_Kbytes, self.bram, self.max_BRAM_util
        )

        nIN1 = len(mem_in_1)
        nOUT1 = len(mem_out_1)
        assert nIN1 > 0 and nOUT1 > 0, "No memory in/out nodes found"
        config_1, mem_bw_1 = self.generate_random_config(
            target_graph=graph_1, n_in=nIN1, n_out=nOUT1
        )
        cost_1, dp_info_1 = self.get_cost(
            config_1,
            mem_bw_1,
            len(mem_in_1),
            len(mem_out_1),
            target_graph=graph_1,
            branch_mem_update=bb1
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
            len(mem_in_2),
            len(mem_out_2),
            target_graph=graph_2,
            branch_mem_update=bb2
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
                ) = split_graph(
                    self.graph,
                    self.word_bytes,
                    self.bram_Kbytes,
                    self.bram,
                    self.max_BRAM_util,
                )

                nIN1 = len(mem_in_1)
                nOUT1 = len(mem_out_1)
                assert nIN1 > 0 and nOUT1 > 0, "No memory in/out nodes found"
                config_1, mem_bw_1 = self.generate_random_config(
                    target_graph=graph_1, n_in=nIN1, n_out=nOUT1
                )
                cost_1, dp_info_1 = self.get_cost(
                    config_1,
                    mem_bw_1,
                    len(mem_in_1),
                    len(mem_out_1),
                    target_graph=graph_1,
                    branch_mem_update=bb1
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
                    len(mem_in_2),
                    len(mem_out_2),
                    target_graph=graph_2,
                    branch_mem_update=bb2
                )
                if (
                    cost_1 is not None
                    and cost_2 is not None
                    and self.validate_configs(dp_info_1, dp_info_2)
                ):
                    break
            if cost_1 is None or cost_2 is None:
                return None, None, None, 0

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
                    ) = split_graph(
                        self.graph,
                        self.word_bytes,
                        self.bram_Kbytes,
                        self.bram,
                        self.max_BRAM_util,
                    )

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
                    len(mem_in_1),
                    len(mem_out_1),
                    target_graph=graph_1,
                    branch_mem_update=bb1
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
                    len(mem_in_2),
                    len(mem_out_2),
                    target_graph=graph_2,
                    branch_mem_update=bb2
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
                    if not os.path.exists(
                        os.getcwd() + "/fpga_modeling_reports/partition_graphs/"
                    ):
                        os.makedirs(
                            os.getcwd() + "/fpga_modeling_reports/partition_graphs/"
                        )
                    visualize_graph(
                        graph_1,
                        os.getcwd()
                        + "/fpga_modeling_reports/partition_graphs/"
                        + self.part_name
                        + "_phase_1",
                        self.enable_wandb,
                        f"{self.part_name}_phase_1",
                    )
                    visualize_graph(
                        graph_2,
                        os.getcwd()
                        + "/fpga_modeling_reports/partition_graphs/"
                        + self.part_name
                        + "_phase_2",
                        self.enable_wandb,
                        f"{self.part_name}_phase_2",
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
                        if not os.path.exists(
                            os.getcwd() + "/fpga_modeling_reports/partition_graphs/"
                        ):
                            os.makedirs(
                                os.getcwd() + "/fpga_modeling_reports/partition_graphs/"
                            )
                        visualize_graph(
                            graph_1,
                            os.getcwd()
                            + "/fpga_modeling_reports/partition_graphs/"
                            + self.part_name
                            + "_phase_1",
                            self.enable_wandb,
                            f"{self.part_name}_phase_1",
                        )
                        visualize_graph(
                            graph_2,
                            os.getcwd()
                            + "/fpga_modeling_reports/partition_graphs/"
                            + self.part_name
                            + "_phase_2",
                            self.enable_wandb,
                            f"{self.part_name}_phase_1",
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
            0
        )

    def initialize_optimizer(self, graph, read_points, write_points, wr_factor):
        self.freeze_param = True

        config, mem_bw, _, _ = self.generate_random_config(target_graph=graph)
        cost, dp_info = self.get_cost(config, mem_bw, read_points, write_points, target_graph=graph, wr_factor=wr_factor)
        slowest_nodes = None

        if cost is None:
            start_time = time.time()
            while time.time() - start_time < 90.0:
                x = float(time.time() - start_time)
                perc = 1/(1+math.exp(-0.1*(x-45)))
                config, mem_bw, _, _ = self.generate_random_config(
                    target_graph=graph,
                    keep_percentage=perc)
                cost, dp_info = self.get_cost(config, mem_bw, read_points, write_points, target_graph=graph, wr_factor=wr_factor)

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

    def initialize_optimizer_layer(self, layer, wr_factor: int = 1):
        config, mem_bw = self.generate_random_config_layer(layer)
        cost, dp_info = self.get_cost_layer(config, mem_bw, layer, wr_factor=wr_factor)

        if cost is None:
            start_time = time.time()
            while time.time() - start_time < 90.0:
                x = float(time.time() - start_time)
                perc = 1/(1+math.exp(-0.1*(x-45)))
                config, mem_bw = self.generate_random_config_layer(layer,
                    keep_percentage=perc)
                cost, dp_info = self.get_cost_layer(config, mem_bw, layer, wr_factor=wr_factor)

                if cost is not None:
                    break
            if cost is None:
                print("No configuration found after 90 seconds. Aborting...")
                return None, None, None

        return config, cost, dp_info

    def run_optimizer(self):
        # if has_gap(self.graph) and self.branch_bram_util > self.max_BRAM_util:
        #     return self.run_optimizer_double_graph()

        #TODO: Searching for partition fitting or not to the device we assume a lower bram utilization than the provided one from the user by 15 %.
        sub_partitions = check_partition_fitting(self.graph, self.max_BRAM_util-15, [])
        extra_reconfigurations = len(sub_partitions) - 1
        print(f'Splitting original partition into {len(sub_partitions)} sub-partitions. A number of {extra_reconfigurations} extra reconfiguration(s) will be added.')

        mem_bw_list = []
        dp_info_list = []
        wr_list = []
        for i, sp in enumerate(sub_partitions):
            graph = sp[0]
            mem_in = sp[1]
            mem_out = sp[2]
            weights_reloading = sp[3]

            read_points, write_points = add_off_chip_connections(
                graph, mem_in, mem_out, gap_approx=self.gap_approx
            )
            visualize_graph(graph, os.getcwd() + '/fpga_modeling_reports/' + self.cnn_model_name + '/partition_graphs/' + self.part_name + '_split' + str(i), self.enable_wandb, f"{self.part_name}_split_{i}",)

            best_solution_mem = None
            best_solution_dp = None
            best_latency = 1000
            for _ in range(self.best_of_iter):
                ( config,
                cost,
                dp_info,
                mem_bw,
                slowest_nodes,
                ) = self.initialize_optimizer(graph=graph, read_points=read_points, write_points=write_points, wr_factor=weights_reloading)

                if config == None:
                    return None, None, None, None, None

                prev_state = config
                prev_cost = cost
                solution_dp = dp_info
                solution_mem = mem_bw

                current_temp = self.t_max
                # first_restart, second_restart, third_restart = True, True, True
                count = 0
                print(f"Temperature  |  Latency     |   Count")
                while current_temp > self.t_min:

                    # if self.enable_wandb:
                    #     log_dict = {}
                    #     log_dict["temperature"] = current_temp
                    #     log_dict["latency"] = prev_cost
                    #     wandb.log(log_dict)

                    num_iterations = 0
                    timeout_tmr_start = time.time()
                    while num_iterations < self.iterationPerTemp and time.time() - timeout_tmr_start < 30.0:
                    # for _ in range(self.iterationPerTemp):
                        (
                            new_state,
                            new_mem_bw,
                            _,
                            _,
                        ) = self.generate_random_config(
                            neighbours=True,
                            prev_state=prev_state,
                            slowest_nodes=slowest_nodes,
                            target_graph=graph
                        )
                        new_cost, new_dp_info = self.get_cost(new_state, new_mem_bw, read_points, write_points, target_graph=graph, wr_factor=weights_reloading)
                        if new_cost is not None:
                            slowest_nodes = new_dp_info["slowestNodes"]

                        if new_cost is None:
                            continue
                        count += 1
                        num_iterations += 1

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
                        f"{current_temp:.5e}\t{prev_cost:.5e}\t{count:5d}",
                        end="\r",
                    )

                print(
                    f"{current_temp:.5e}\t{prev_cost:.5e}\t{count:5d}"
                )

                if prev_cost < best_latency:
                    best_latency = prev_cost
                    best_solution_mem = solution_mem
                    best_solution_dp = solution_dp

            mem_bw_list.append(best_solution_mem)
            dp_info_list.append(best_solution_dp)
            wr_list.append(weights_reloading)

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
        return self.mem_words_per_cycle, mem_bw_list, dp_info_list, extra_reconfigurations, wr_list

    def get_cost(
        self,
        config,
        mem_bw,
        read_points,
        write_points,
        target_graph=None,
        branch_mem_update=None,
        wr_factor=1
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
            elif v["op_type"] == "Pooling":
                comb_config[k] = [v["fine"], v["coarse_inout"]]
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

        dp_info = self.partition_composer.get_design_point(
            graph.copy(),
            comb_config,
            mem_bw[0],
            mem_bw[1],
            read_points,
            write_points,
            gap_approx=self.gap_approx,
            branch_mem=branch_mem,
            wr_factor=wr_factor
        )
        if dp_info["config"]:
            return dp_info["latency(S)"], dp_info
            # return -dp_info['GOP/s'], dp_info
        return None, None

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
                elif isinstance(hw, Pooling3DLayer):
                    channels = hw.channels
                    kernel_size = hw.kernel_shape
                    coarse_inout_feasible = utils.get_factors(channels)
                    fine_feasible = utils.get_fine_feasible(kernel_size)
                    coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
                    fine_factor = random.choice(fine_feasible) / np.prod(
                        np.array(kernel_size)
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "fine": fine_factor,
                        "coarse_inout": coarse_inout_factor,
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
        keep_percentage=None,
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
                    channels, keep_percentage=keep_percentage
                )
                coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
                if slowest_nodes and node in config.keys():
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
                    channels, keep_percentage=keep_percentage
                )
                coarse_out_feasible = utils.get_factors(
                    filters, keep_percentage=keep_percentage
                )
                fine_feasible = utils.get_fine_feasible(kernel_size, keep_percentage=keep_percentage)
                coarse_in_factor = random.choice(coarse_in_feasible) / channels
                coarse_out_factor = random.choice(coarse_out_feasible) / filters
                fine_factor = random.choice(fine_feasible) / np.prod(
                    np.array(kernel_size)
                )
                if slowest_nodes and node in config.keys():
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
            elif isinstance(hw, Pooling3DLayer):
                channels = hw.channels
                kernel_size = hw.kernel_shape
                coarse_inout_feasible = utils.get_factors(
                    channels, keep_percentage=keep_percentage
                )
                fine_feasible = utils.get_fine_feasible(kernel_size, keep_percentage=keep_percentage)
                coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
                fine_factor = random.choice(fine_feasible) / np.prod(
                    np.array(kernel_size)
                )
                if slowest_nodes and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove("op_type")
                    apply_transform = random.choice(transformations)
                    if apply_transform == "coarse_inout":
                        config[node][apply_transform] = coarse_inout_factor
                else:
                    config[node] = {
                        "op_type": op_type,
                        "fine": fine_factor,
                        "coarse_inout": coarse_inout_factor,
                    }
            elif isinstance(hw, ActivationLayer):
                channels = hw.channels
                coarse_inout_feasible = utils.get_factors(
                    channels, keep_percentage=keep_percentage
                )
                coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
                if slowest_nodes and node in config.keys():
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
                    channels, keep_percentage=keep_percentage
                )
                coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
                if slowest_nodes and node in config.keys():
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
                    channels, keep_percentage=keep_percentage
                )
                coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
                if slowest_nodes and node in config.keys():
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
                    dim_in, keep_percentage=keep_percentage
                )
                coarse_out_feasible = utils.get_factors(
                    dim_out, keep_percentage=keep_percentage
                )
                coarse_in_factor = random.choice(coarse_in_feasible) / dim_in
                coarse_out_factor = random.choice(coarse_out_feasible) / dim_out
                if slowest_nodes and node in config.keys():
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

        num_in_nodes = len(get_input_nodes(graph))
        num_out_nodes = len(get_output_nodes(graph))
        mem_config_in, mem_config_out = self.get_mem_bw_feasible(
            n_in=num_in_nodes, n_out=num_out_nodes, gap_approx=self.gap_approx
        )

        return config, [mem_config_in, mem_config_out], self.param_changes, param_perc

    def run_optimizer_layer(self, layer):
        """Run the optimizer for a single layer."""

        hw = self.graph.nodes[layer]["hw"]
        wr_factor = 1
        if isinstance(hw, Convolutional3DLayer):
            initial_filters = deepcopy(hw.filters)
            coarsein_min = 1 / np.int32(hw.channels)
            coarseout_min = 1 / np.int32(hw.filters)
            fine_min = 1 / np.prod(np.array(hw.kernel_shape))
            _, bram_util = hw.get_resource_util(f_fine = fine_min,
                                            f_coarseIn = coarsein_min,
                                            f_coarseOut= coarseout_min)
            print("Initial BRAM utilization: ", bram_util)
            if bram_util > self.max_BRAM_util:
                _logger.warning(f"Layer's ({layer}) minimum BRAM utilization is above the device's maximum on chip memory resources.\nSplit the layer execution into multiple instances (weights reloading).")
                for f in utils.get_factors(initial_filters)[1:]:
                    new_out_shape = deepcopy(hw.output_shape)
                    new_out_shape[1] = int(initial_filters/f)
                    hw.update_shapes(hw.input_shape, new_out_shape)
                    coarsein_min = 1 / np.int32(hw.channels)
                    coarseout_min = 1 / np.int32(hw.filters)
                    fine_min = 1 / np.prod(np.array(hw.kernel_shape))
                    _, bram_util = hw.get_resource_util(f_fine = fine_min,
                                                    f_coarseIn = coarsein_min,
                                                    f_coarseOut= coarseout_min)
                    if bram_util < self.max_BRAM_util:
                        wr_factor = f
                        break
                if wr_factor == 1:
                    return None

        config, cost, dp_info = self.initialize_optimizer_layer(layer, wr_factor=wr_factor)
        if config == None:
            return None

        prev_state = config
        solution_dp = dp_info
        prev_cost = cost

        current_temp = self.t_max

        print(f"Temperature  |  Latency")
        while current_temp > self.t_min:

            for i in range(self.iterationPerTemp):
                new_state, new_mem_bw = self.generate_random_config_layer(layer)
                new_cost, new_dp_info = self.get_cost_layer(
                    new_state, new_mem_bw, layer, wr_factor=wr_factor
                )

                if new_cost is None:
                    continue

                cost_diff = prev_cost - new_cost
                if cost_diff > 0:
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
        return solution_dp

    def get_cost_layer(self, config, mem_bw, layer, wr_factor: int=1):
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
                wr_factor=wr_factor
            )
        elif isinstance(hw, Pooling3DLayer):
            dp_info = hw.get_design_point(
                config[0],
                config[1],
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

    def generate_random_config_layer(self, l: str, keep_percentage: float = -1):
        config = []
        hw = self.graph.nodes[l]["hw"]
        if isinstance(hw, GAPLayer):
            channels = hw.channels
            filters = hw.filters
            coarse_inout_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
            coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
            config = [coarse_inout_factor]
        elif isinstance(hw, Convolutional3DLayer):
            channels = hw.channels
            filters = hw.filters
            kernel_size = hw.kernel_shape
            coarse_in_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
            coarse_out_feasible = utils.get_factors(filters, keep_percentage=keep_percentage)
            fine_feasible = utils.get_fine_feasible(kernel_size, keep_percentage=keep_percentage)
            coarse_in_factor = random.choice(coarse_in_feasible) / channels
            coarse_out_factor = random.choice(coarse_out_feasible) / filters
            fine_factor = random.choice(fine_feasible) / np.prod(np.array(kernel_size))
            config = [fine_factor, coarse_in_factor, coarse_out_factor]
        elif isinstance(hw, Pooling3DLayer):
            channels = hw.channels
            kernel_size = hw.kernel_shape
            coarse_inout_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
            fine_feasible = utils.get_fine_feasible(kernel_size, keep_percentage=keep_percentage)
            coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
            fine_factor = random.choice(fine_feasible) / np.prod(np.array(kernel_size))
            config = [coarse_inout_factor, fine_factor]
        elif isinstance(hw, ActivationLayer):
            channels = hw.channels
            coarse_inout_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
            coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
            config = [coarse_inout_factor]
        elif isinstance(hw, ElementWiseLayer):
            channels = hw.channels_1
            coarse_inout_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
            coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
            config = [coarse_inout_factor]
        elif isinstance(hw, BatchNorm3DLayer):
            channels = hw.channels
            coarse_inout_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
            coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
            config = [coarse_inout_factor]
        elif isinstance(hw, SqueezeExcitationLayer):
            assert False, "Not supported layer (SqueezeExcitationLayer)"
        elif isinstance(hw, FCLayer):
            dim_in = hw.dim_in
            dim_out = hw.dim_out
            coarse_in_feasible = utils.get_factors(dim_in, keep_percentage=keep_percentage)
            coarse_out_feasible = utils.get_factors(dim_out, keep_percentage=keep_percentage)
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
        _logger.info(msg=f"Validating building blocks setup... {bblocks}")
        for n in nodes:
            bb = self.graph.nodes[n]["hw_type"]
            if any(bb in block for block in bblocks):
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
        types = {}
        for n in self.graph.nodes:
            if self.graph.nodes[n]["hw_type"] not in types:
                if "Conv" in self.graph.nodes[n]["hw_type"] or "Pooling" in self.graph.nodes[n]["hw_type"]:
                    types[self.graph.nodes[n]["hw_type"]] = {
                        "Padding": [self.graph.nodes[n]["hw"].padding],
                        "Stride": [self.graph.nodes[n]["hw"].stride],
                    }
                else:
                    types[self.graph.nodes[n]["hw_type"]] = {}
            else:
                if "Conv" in self.graph.nodes[n]["hw_type"] or "Pooling" in self.graph.nodes[n]["hw_type"]:
                    types[self.graph.nodes[n]["hw_type"]]["Padding"].append(
                        self.graph.nodes[n]["hw"].padding
                    )
                    types[self.graph.nodes[n]["hw_type"]]["Stride"].append(
                        self.graph.nodes[n]["hw"].stride
                    )

        bblocks = []
        for t in types:
            if "Conv" in t or "Pooling" in t:
                final_padding = np.max(np.array(types[t]["Padding"]), axis=0).tolist()
                final_stride = np.min(np.array(types[t]["Stride"]), axis=0).tolist()
                bblocks.append(
                    f"{t}p{''.join([str(elem) for elem in final_padding])}s{''.join([str(elem) for elem in final_stride])}"
                )
            else:
                bblocks.append(t)

        bblocks, lookuptable = utils.combine_building_blocks(bblocks)

        # assert self.validate_building_blocks_setup(
        #     bblocks
        # ), "Invalid building blocks setup. Cannot find a valid scheduling."

        return bblocks, lookuptable

    def generate_building_blocks_config(
        self,
        bblocks: list,
        alignedfactors: bool,
        lookuptable: dict,
        previous_config: dict = None,
        initialization: bool = False,
    ) -> dict:
        """
        Generate a configuration for each building block based on the min and max channels and filters values
        from all its instances across the graph of the network.

        Returns:
            dict: bb_setup
        """
        if "Activation" in bblocks:
            activations_list = []
            for n in self.graph.nodes:
                if self.graph.nodes[n]["hw_type"] == "Activation":
                    if self.graph.nodes[n]['hw'].op_type not in activations_list:
                        activations_list.append(self.graph.nodes[n]['hw'].op_type)
        if "ElementWise" in bblocks:
            elementwise_list = []
            for n in self.graph.nodes:
                if self.graph.nodes[n]["hw_type"] == "ElementWise":
                    if self.graph.nodes[n]['hw'].op_type not in elementwise_list:
                        elementwise_list.append(self.graph.nodes[n]['hw'].op_type)

        total_dsp = 0
        total_bram = 0
        if (not initialization) and (not previous_config == None) and (bblocks == list(previous_config.keys())):
            bb_setup = deepcopy(previous_config)
            bb_choice = [bb for bb in bblocks]
            bblocks = random.choices(bb_choice, k=int(len(bb_choice)*self.bblock_keep_percentage))
            for b in bb_setup:
                total_dsp += bb_setup[b]["DSP_util"]
                total_bram += bb_setup[b]["BRAM_util"]
        else:
            bb_setup = dict()

        for bb in bblocks:
            if not bb in bb_setup.keys():
                bb_setup[bb] = dict()

            dsp_util, bram_util = 100, 100
            stop_counter = 0
            while dsp_util > (self.max_DSP_util - total_dsp) or bram_util > (
                self.max_BRAM_util - total_bram
            ):
                if self.use_arbitrary_shape:
                    shape_in, shape_out = utils.get_random_arbitrary_shape(
                        self.graph, bb, lookuptable, previous_config=previous_config, chan_dist_thresh=self.chan_dist_thresh, depth_dist_thresh=self.depth_dist_thresh, height_dist_thresh=self.height_dist_thresh
                    )
                else:
                    shape_in, shape_out = utils.get_random_shape(
                        self.graph, bb, lookuptable, previous_config=previous_config, chan_dist_thresh=self.chan_dist_thresh, depth_dist_thresh=self.depth_dist_thresh,height_dist_thresh=self.height_dist_thresh
                    )
                if bb != "Gemm":
                    _, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim = shape_in
                    (
                        _,
                        channels_out_dim,
                        depth_out_dim,
                        height_out_dim,
                        width_out_dim,
                    ) = shape_out
                else:
                    channels_in_dim, channels_out_dim = shape_in[1], shape_out[1]
                    depth_in_dim, height_in_dim, width_in_dim = 1, 1, 1
                    depth_out_dim, height_out_dim, width_out_dim = 1, 1, 1

                bb_descriptor = utils.generate_description_from_type(
                    bb,
                    channels_in_dim,
                    depth_in_dim,
                    height_in_dim,
                    width_in_dim,
                    channels_out_dim,
                    depth_out_dim,
                    height_out_dim,
                    width_out_dim,
                )

                if "Conv" in bb:
                    bb_setup[bb]["hw"] = Convolutional3DLayer(
                        self.max_DSP_util, self.max_BRAM_util, bb_descriptor
                    )
                if "Pooling" in bb:
                    bb_setup[bb]["hw"] = Pooling3DLayer(
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
                elif bb == "ElementWise":
                    bb_setup[bb]["hw"] = ElementWiseLayer(
                        self.max_DSP_util, self.max_BRAM_util, bb_descriptor
                    )
                elif bb == "Gemm":
                    bb_setup[bb]["hw"] = FCLayer(
                        self.max_DSP_util, self.max_BRAM_util, bb_descriptor
                    )

                if "Conv" in bb:
                    if alignedfactors:
                        coarse_in = (
                            random.choice(utils.get_factors(channels_in_dim))
                            / channels_in_dim
                        )
                        coarse_out = (
                            random.choice(utils.get_factors(channels_out_dim))
                            / channels_out_dim
                        )
                    else:
                        coarse_in = (
                            np.random.choice(
                                np.arange(channels_in_dim) + 1, replace=False
                            )
                            / channels_in_dim
                        )
                        coarse_out = (
                            np.random.choice(
                                np.arange(channels_out_dim) + 1, replace=False
                            )
                            / channels_out_dim
                        )
                    if "Conv3DDw" in bb:
                        coarse_out = coarse_in
                    assert coarse_in > 0 and coarse_in <= 1, "Invalid coarse in."
                    assert coarse_out > 0 and coarse_out <= 1, "Invalid coarse out."
                    # TODO: Add fine factor random generation for Conv3D ops
                    dsp_util, bram_util = bb_setup[bb]["hw"].get_resource_util(
                        f_fine=1, f_coarseIn=coarse_in, f_coarseOut=coarse_out
                    )
                elif "Pooling" in bb:
                    if alignedfactors:
                        coarse_inout = (
                            random.choice(utils.get_factors(channels_in_dim))
                            / channels_in_dim
                        )
                    else:
                        coarse_inout = (
                            np.random.choice(
                                np.arange(channels_in_dim) + 1, replace=False
                            )
                            / channels_in_dim
                        )
                    assert coarse_inout > 0 and coarse_inout <= 1, "Invalid coarse factor."
                    # TODO: Add fine factor random generation for Pooling3D ops
                    dsp_util, bram_util = bb_setup[bb]["hw"].get_resource_util(
                        f_fine=1, f_coarse_inout=coarse_inout
                    )
                elif bb in ["Activation", "GlobalAveragePool", "ElementWise"]:
                    if alignedfactors:
                        coarse_inout = (
                            random.choice(utils.get_factors(channels_in_dim))
                            / channels_in_dim
                        )
                    else:
                        coarse_inout = (
                            np.random.choice(
                                np.arange(channels_in_dim) + 1, replace=False
                            )
                            / channels_in_dim
                        )
                    assert coarse_inout > 0 and coarse_inout <= 1, "Invalid coarse factor."
                    if bb == "Activation":
                        supported_ops = deepcopy(activations_list)
                    elif bb == "ElementWise":
                        supported_ops = deepcopy(elementwise_list)
                    elif bb == "GlobalAveragePool":
                        supported_ops = []

                    dsp_util, bram_util = bb_setup[bb]["hw"].get_resource_util(
                        f_coarse_inout=coarse_inout, supported_ops=supported_ops
                    )
                elif bb == "Gemm":
                    if alignedfactors:
                        coarse_in = (
                            random.choice(utils.get_factors(channels_in_dim))
                            / channels_in_dim
                        )
                        coarse_out = (
                            random.choice(utils.get_factors(channels_out_dim))
                            / channels_out_dim
                        )
                    else:
                        coarse_in = (
                            np.random.choice(
                                np.arange(channels_in_dim) + 1, replace=False
                            )
                            / channels_in_dim
                        )
                        coarse_out = (
                            np.random.choice(
                                np.arange(channels_out_dim) + 1, replace=False
                            )
                            / channels_out_dim
                        )
                    assert coarse_in > 0 and coarse_in <= 1, "Invalid coarse in."
                    assert coarse_out > 0 and coarse_out <= 1, "Invalid coarse out."
                    dsp_util, bram_util = bb_setup[bb]["hw"].get_resource_util(
                        f_coarseIn=coarse_in, f_coarseOut=coarse_out
                    )
                stop_counter += 1
                if stop_counter > 100:
                    _logger.debug(
                        "Could not find a valid configuration for the current building block setup. Returning without result."
                    )
                    return None

            mem_bw = bb_setup[bb]["hw"].mem_words_per_cycle
            if "ElementWise" in bb:
                bw_in, bw_out = self.get_mem_bw_feasible(n_in=2, n_out=1)
                bw_in_1 = bw_in[0] * mem_bw
                bw_in_2 = bw_in[1] * mem_bw
                bw_out = bw_out[0] * mem_bw
                bb_setup[bb]["bw_in"] = [bw_in_1, bw_in_2]
                bb_setup[bb]["bw_out"] = [bw_out]
            else:
                bw_in, bw_out = self.get_mem_bw_feasible(n_in=1, n_out=1)
                bw_in = bw_in[0] * mem_bw
                bw_out = bw_out[0] * mem_bw
                bb_setup[bb]["bw_in"] = [bw_in]
                bb_setup[bb]["bw_out"] = [bw_out]
            if "Conv" in bb:
                layer_config = utils.generate_layer_config(bb_setup[bb]["hw"], [1, coarse_in, coarse_out])
                bb_setup[bb]["config"] = layer_config
                bb_setup[bb]["f_coarseIn"] = coarse_in
                bb_setup[bb]["interleaving_in"] = math.ceil(1 / coarse_in)
                bb_setup[bb]["f_coarseOut"] = coarse_out
                bb_setup[bb]["interleaving_out"] = math.ceil(1 / coarse_out)
            elif "Pooling" in bb:
                layer_config = utils.generate_layer_config(bb_setup[bb]["hw"], [1, coarse_inout])
                bb_setup[bb]["config"] = layer_config
                bb_setup[bb]["coarse_inout"] = coarse_inout
                bb_setup[bb]["coarse_factor"] = math.ceil(coarse_inout * channels_in_dim)
                bb_setup[bb]["interleaving_inout"] = math.ceil(1 / coarse_inout)
            elif bb in ["Activation", "GlobalAveragePool", "ElementWise"]:
                layer_config = utils.generate_layer_config(bb_setup[bb]["hw"], [coarse_inout])
                if bb == "Activation":
                    layer_config['supported_ops'] = deepcopy(activations_list)
                elif bb == "ElementWise":
                    layer_config['supported_ops'] = deepcopy(elementwise_list)
                bb_setup[bb]["config"] = layer_config
                bb_setup[bb]["coarse_inout"] = coarse_inout
                bb_setup[bb]["coarse_factor"] = math.ceil(coarse_inout * channels_in_dim)
                bb_setup[bb]["interleaving_inout"] = math.ceil(1 / coarse_inout)
            elif bb == "Gemm":
                layer_config = utils.generate_layer_config(bb_setup[bb]["hw"], [coarse_in, coarse_out])
                bb_setup[bb]["config"] = layer_config
                bb_setup[bb]["coarse_in"] = coarse_in
                bb_setup[bb]["interleaving_in"] = math.ceil(1 / coarse_in)
                bb_setup[bb]["coarse_out"] = coarse_out
                bb_setup[bb]["interleaving_out"] = math.ceil(1 / coarse_out)

            bb_setup[bb]["DSP_util"] = dsp_util
            bb_setup[bb]["BRAM_util"] = bram_util
            total_dsp += dsp_util
            total_bram += bram_util

        return bb_setup

    def get_cost_e2e(self, bblocks_config: dict, lookuptable: dict) -> float:
        if bblocks_config is None:
            return None, None, None, None, None
        cost = 0.0
        avg_BW = 0.0
        scheduling = {}
        for node in self.graph.nodes:
            bb_type = lookuptable[self.graph.nodes[node]["hw_type"]]
            hw = self.graph.nodes[node]["hw"]

            if len(hw.input_shape) < 5:
                depth_calls = 1
                height_calls = 1
                width_calls = 1
            else:
                # TODO: Search how to deal with cases where the output dimensions are also altered? Like in convolutional layers with stride > 1.
                depth_calls = math.ceil(
                    hw.input_shape[2] / bblocks_config[bb_type]["config"]["depth_in"]
                )
                height_calls = math.ceil(
                    hw.input_shape[3] / bblocks_config[bb_type]["config"]["height_in"]
                )
                width_calls = math.ceil(
                    hw.input_shape[4] / bblocks_config[bb_type]["config"]["width_in"]
                )

            if "Conv" in bb_type or "Gemm" in bb_type:
                in_calls = math.ceil(
                    hw.input_shape[1]
                    / (
                        bblocks_config[bb_type]["config"]["coarse_in_factor"]
                        * bblocks_config[bb_type]["interleaving_in"]
                    )
                )
                out_calls = math.ceil(
                    hw.output_shape[1]
                    / (
                        bblocks_config[bb_type]["config"]["coarse_out_factor"]
                        * bblocks_config[bb_type]["interleaving_out"]
                    )
                )
                if bb_type == "Gemm":
                    total_block_calls = (
                        in_calls * out_calls * depth_calls * height_calls * width_calls
                    )
                else:
                    total_block_calls = (
                        in_calls * out_calls * depth_calls * height_calls * width_calls
                    )
            else:
                inout_calls = math.ceil(
                    hw.input_shape[1]
                    / (
                        bblocks_config[bb_type]["coarse_factor"]
                        * bblocks_config[bb_type]["interleaving_inout"]
                    )
                )
                total_block_calls = (
                    inout_calls * depth_calls * height_calls * width_calls
                )

            # TODO: Update the shapes of the bblocks_config[bb_type]["hw"] to account for the overlapping regions because of feature map tilling
            if "Conv" in bb_type:
                bblock_padding = deepcopy(bblocks_config[bb_type]["hw"].padding)
                bblock_stride = deepcopy(bblocks_config[bb_type]["hw"].stride)

                bblocks_config[bb_type]["hw"].padding = hw.padding
                bblocks_config[bb_type]["hw"].stride = hw.stride
                performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                    f_fine=1,
                    f_coarseIn=bblocks_config[bb_type]["f_coarseIn"],
                    f_coarseOut=bblocks_config[bb_type]["f_coarseOut"],
                    mem_bw_in=bblocks_config[bb_type]["bw_in"][0],
                    mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
                )
                bblocks_config[bb_type]["hw"].padding = bblock_padding
                bblocks_config[bb_type]["hw"].stride = bblock_stride
            elif "Pooling" in bb_type:
                bblock_padding = deepcopy(bblocks_config[bb_type]["hw"].padding)
                bblock_stride = deepcopy(bblocks_config[bb_type]["hw"].stride)

                bblocks_config[bb_type]["hw"].padding = hw.padding
                bblocks_config[bb_type]["hw"].stride = hw.stride
                performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                    f_fine=1,
                    f_coarse_inout=bblocks_config[bb_type]["coarse_inout"],
                    mem_bw_in=bblocks_config[bb_type]["bw_in"][0],
                    mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
                )
                bblocks_config[bb_type]["hw"].padding = bblock_padding
                bblocks_config[bb_type]["hw"].stride = bblock_stride
            elif bb_type == "Activation":
                bblocks_config[bb_type]["hw"].op_type = hw.op_type
                performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                    coarse_inout=bblocks_config[bb_type]["coarse_inout"],
                    mem_bw_in=bblocks_config[bb_type]["bw_in"][0],
                    mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
                )
                bblocks_config[bb_type]["hw"].op_type = "Activation"
            elif bb_type == "GlobalAveragePool":
                performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                    coarse_inout=bblocks_config[bb_type]["coarse_inout"],
                    mem_bw_in=bblocks_config[bb_type]["bw_in"][0],
                    mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
                )
            elif bb_type == "ElementWise":
                bblocks_config[bb_type]["hw"].op_type = hw.op_type
                bblock_input_shape_2 = deepcopy(
                    bblocks_config[bb_type]["hw"].input_shape_2
                )

                layer_input_shape_2 = hw.input_shape_2
                if np.prod(layer_input_shape_2[2:]) == 1:
                    bblocks_config[bb_type]["hw"].input_shape_2[
                        2:
                    ] = layer_input_shape_2[2:]

                performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                    coarse_inout=bblocks_config[bb_type]["coarse_inout"],
                    mem_bw_in_1=bblocks_config[bb_type]["bw_in"][0],
                    mem_bw_in_2=bblocks_config[bb_type]["bw_in"][1],
                    mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
                )
                bblocks_config[bb_type]["hw"].op_type = "ElementWise"
                bblocks_config[bb_type]["hw"].input_shape_2 = bblock_input_shape_2
            elif bb_type == "Gemm":
                performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                    coarse_in=bblocks_config[bb_type]["coarse_in"],
                    coarse_out=bblocks_config[bb_type]["coarse_out"],
                    mem_bw_in=bblocks_config[bb_type]["bw_in"][0],
                    mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
                )

            # TODO: Revert back the shapes of the bblocks_config[bb_type]["hw"]

            bblocks_config[bb_type]["MemBw_util"] = float(performance_modeling["memBwUtil"])

            latency = performance_modeling["latency(S)"] * math.ceil(total_block_calls)
            avg_BW += performance_modeling["memBwUtil"]
            assert math.ceil(total_block_calls) > 0, "Zero calls aborting..."
            scheduling[node] = {
                "Block Type": bb_type,
                "Shape In": hw.input_shape,
                "Shape In 2": hw.input_shape_red if bb_type == "ElementWise" else [],
                "Shape Out": hw.output_shape,
                "Tiling Channels": in_calls
                if "Conv" in bb_type or "Gemm" in bb_type
                else inout_calls,
                "Tiling Filters": out_calls
                if "Conv" in bb_type or "Gemm" in bb_type
                else 1,
                "Tiling Depth": depth_calls,
                "Tiling Height": height_calls,
                "Tiling Width": width_calls,
                "Latency": latency,
                "Base Latency": performance_modeling["latency(S)"],
                "Read From": list(self.graph.predecessors(node)),
                "Write To": list(self.graph.successors(node)),
                "Store To Mem": (True if self.graph.out_degree(node) > 1 else False),
                "Load From Mem": (True if self.graph.in_degree(node) > 1 else False),
                "Kernel Shape": hw.kernel_shape if "Conv" in bb_type else [],
                "Stride": hw.stride if "Conv" in bb_type else [],
                "Padding": hw.padding if "Conv" in bb_type else [],
                "Broadcast": hw.broadcasting if bb_type == "ElementWise" else False,
                "Op Type": hw.op_type
                if bb_type in ["Activation", "ElementWise"]
                else "None",
            }
            cost += latency

        avg_BW = avg_BW / len(self.graph.nodes)
        final_DSP = 0
        final_BRAM = 0
        for bb in bblocks_config:
            final_DSP += bblocks_config[bb]["DSP_util"]
            final_BRAM += bblocks_config[bb]["BRAM_util"]
        return cost, scheduling, final_DSP, final_BRAM, avg_BW

    def run_optimizer_latency(self, alignedfactors: bool) -> None:

        bblocks, lookuptable = self.generate_building_blocks()
        bblocks_config = self.generate_building_blocks_config(
            bblocks, alignedfactors, lookuptable, initialization=True
        )

        cost, scheduling, dsp_util, bram_util, bw_util = self.get_cost_e2e(
            bblocks_config, lookuptable
        )

        if cost is None:
            for _ in range(100):
                bblocks, lookuptable = self.generate_building_blocks()
                bblocks_config = self.generate_building_blocks_config(
                    bblocks, alignedfactors, lookuptable, initialization=True
                )
                cost, scheduling, dsp_util, bram_util, bw_util = self.get_cost_e2e(
                    bblocks_config, lookuptable
                )
                if cost is not None:
                    break
            if cost is None:
                _logger.error("No configuration found in 100 iterations. Exiting...")
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
            if self.block_gen == 'pre_while':
                bblocks, lookuptable = self.generate_building_blocks()

            if self.enable_wandb:
                log_dict = {}
                log_dict["temperature"] = current_temp
                log_dict["latency"] = prev_cost
                log_dict["dsp_util"] = prev_dsp
                log_dict["bram_util"] = prev_bram
                log_dict["mem_bw_util"] = prev_bw
                log_dict["num_blocks"] = len(prev_state)
                wandb.log(log_dict)

            num_iterations = 0
            timeout_tmr_start = time.time()
            while num_iterations < self.iterationPerTemp and time.time() - timeout_tmr_start < 30.0:
            # for _ in range(self.iterationPerTemp):
                if self.block_gen == 'post_while':
                    bblocks, lookuptable = self.generate_building_blocks()

                if self.use_previous_config:
                    new_state = self.generate_building_blocks_config(
                        bblocks, alignedfactors, lookuptable, previous_config=prev_state
                    )
                else:
                    new_state = self.generate_building_blocks_config(
                        bblocks, alignedfactors, lookuptable, previous_config=None
                    )

                if new_state is None:
                    continue
                num_iterations += 1

                (
                    new_cost,
                    new_scheduling,
                    dsp_util,
                    bram_util,
                    bw_util,
                ) = self.get_cost_e2e(new_state, lookuptable)


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

        print(f"{current_temp:.5e}\t{prev_cost:.5e}\n")
        final_config = deepcopy(prev_state)
        final_DSP_util = 0
        final_BRAM_util = 0
        final_avg_MemBw_util = 0
        for key in final_config:
            final_config[key].pop("hw")
            final_DSP_util += final_config[key]["DSP_util"]
            final_BRAM_util += final_config[key]["BRAM_util"]
            final_avg_MemBw_util += final_config[key]["MemBw_util"]

        final_avg_MemBw_util = final_avg_MemBw_util / len(final_config)
        print(
            f"DSP Utilization: {final_DSP_util:.3f} - BRAM Utilization: {final_BRAM_util:.3f} - MemBw Utilization: {final_avg_MemBw_util:.3f}"
        )
        if self.enable_wandb:
            artifact = wandb.Artifact("config", type="json")
            with artifact.new_file("config.json") as f:
                json.dump(final_config, f, indent=2)
            wandb.log_artifact(artifact)
            artifact = wandb.Artifact("scheduling", type="json")
            with artifact.new_file("scheduling.json") as f:
                json.dump(prev_scheduling, f, indent=2)
            wandb.log_artifact(artifact)
        else:
            if not os.path.exists(
                "fpga_modeling_reports/" + self.cnn_model_name + "/latency_driven"
            ):
                os.makedirs(
                    "fpga_modeling_reports/" + self.cnn_model_name + "/latency_driven"
                )
            with open(
                "fpga_modeling_reports/" + self.cnn_model_name + "/latency_driven/config.json",
                "w",
            ) as f:
                json.dump(final_config, f, indent=2)
            with open(
                "fpga_modeling_reports/" + self.cnn_model_name + "/latency_driven/scheduling.json",
                "w",
            ) as f:
                json.dump(prev_scheduling, f, indent=2)
