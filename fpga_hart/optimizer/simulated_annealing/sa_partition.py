import copy
import math
import os
import random
import time
from copy import deepcopy

import networkx as nx
import numpy as np
from colorama import Fore, init

import wandb
from fpga_hart import _logger
from fpga_hart.layers.activation_3d import Activation3DLayer
from fpga_hart.layers.batchnorm_3d import BatchNorm3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise_3d import ElementWise3DLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap_3d import GAP3DLayer
from fpga_hart.layers.memory_interface import MemoryNode
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.layers.squeeze_excitation import SqueezeExcitationLayer
from fpga_hart.optimizer.optimizer_helper import (
    calculate_wr_factor,
    get_extra_mem_connections,
    get_minimum_resource_utilization,
    get_off_chip_mem_connections,
    get_worst_case_buffering,
)
from fpga_hart.utils import utils
from fpga_hart.utils.graph_manipulation import (
    add_off_chip_connections,
    get_input_nodes,
    get_output_nodes,
    get_split_points,
    split_graph,
    visualize_graph,
)

init(autoreset=True)


def check_partition_fitting(
    self,
    graph,
    partition_composer,
    max_BRAM_util,
    word_bytes,
    bram_type,
    brams_total,
    mem_words_per_cycle,
    result=[],
    original_graph=None,
    gap_approx=False,
    force_split=False,
):
    if original_graph is None:
        original_graph = deepcopy(graph)
    sort_order = list(nx.topological_sort(original_graph))

    _, min_bram_util = get_worst_case_buffering(
        deepcopy(graph),
        partition_composer,
        mem_words_per_cycle,
        word_bytes,
        bram_type,
        brams_total,
        gap_approx,
    )
    if min_bram_util > max_BRAM_util or force_split:
        for sp in reversed(get_split_points(graph)):
            ancestors = list(nx.ancestors(graph, sp))
            ancestors.sort(key=lambda val: sort_order.index(val))
            subgraph_nodes = deepcopy(ancestors)
            subgraph_nodes.append(sp)

            # TODO: I dont like this, but it works. It should be improved since it is a workaround
            if len(subgraph_nodes) == 1 and "Conv" not in subgraph_nodes[0]:
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

            _, min_bram_util = get_worst_case_buffering(
                deepcopy(graph_2),
                partition_composer,
                mem_words_per_cycle,
                word_bytes,
                bram_type,
                brams_total,
                gap_approx,
            )

            if min_bram_util <= max_BRAM_util:
                extra_inputs, extra_outputs = get_extra_mem_connections(
                    original_graph, subgraph_nodes
                )
                weights_reloading = calculate_wr_factor(graph_1, max_BRAM_util)
                result.append([graph_1, extra_inputs, extra_outputs, weights_reloading])

                if len(graph_2.nodes()) > 0:
                    self.check_partition_fitting(
                        graph_2,
                        partition_composer,
                        max_BRAM_util,
                        word_bytes,
                        bram_type,
                        brams_total,
                        mem_words_per_cycle,
                        result,
                        original_graph,
                        gap_approx=gap_approx,
                    )
                break
        if min_bram_util > max_BRAM_util:
            raise ValueError(
                f"Graph {graph.nodes()} does not fit in the device even after partitioning based on branch buffering"
            )
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
                            if (
                                "GlobalAveragePool" in descendant
                                or "Conv" in descendant
                            ):
                                break
                            if "Add" in descendant or "Mul" in descendant:
                                split_points = get_split_points(graph)
                                if not split_points:
                                    subgraph_nodes.append(descendant)
                                else:
                                    # TODO: Check to which split point the descendant is connected and check only that
                                    if len(split_points) > 1:
                                        raise ValueError(
                                            "More than one split point found. Not supported yet."
                                        )
                                    if not list(
                                        nx.all_simple_paths(
                                            graph, split_points[0], descendant
                                        )
                                    ):
                                        subgraph_nodes.append(descendant)
                                        break

                    extra_inputs, extra_outputs = get_extra_mem_connections(
                        original_graph, subgraph_nodes
                    )
                    graph_1 = graph.subgraph(subgraph_nodes).copy()
                    weights_reloading = calculate_wr_factor(graph_1, max_BRAM_util)
                    result.append([
                        graph_1,
                        extra_inputs,
                        extra_outputs,
                        weights_reloading,
                    ])
                else:
                    subgraph_nodes = [layer]

                    for descendant in descendants:
                        if "GlobalAveragePool" in descendant or "Conv" in descendant:
                            break
                        subgraph_nodes.append(descendant)

                    extra_inputs, extra_outputs = get_extra_mem_connections(
                        original_graph, subgraph_nodes
                    )
                    graph_1 = graph.subgraph(subgraph_nodes).copy()
                    weights_reloading = calculate_wr_factor(graph_1, max_BRAM_util)
                    result.append([
                        graph_1,
                        extra_inputs,
                        extra_outputs,
                        weights_reloading,
                    ])

                graph_2 = graph.copy()
                graph_2.remove_nodes_from(graph_1.nodes())
                if len(graph_2.nodes()) > 0:
                    self.check_partition_fitting(
                        graph_2,
                        partition_composer,
                        max_BRAM_util,
                        word_bytes,
                        bram_type,
                        brams_total,
                        mem_words_per_cycle,
                        result,
                        original_graph,
                        gap_approx=gap_approx,
                    )
                break
        if min_bram_util <= max_BRAM_util:
            extra_inputs, extra_outputs = get_extra_mem_connections(
                original_graph, graph.nodes()
            )
            weights_reloading = calculate_wr_factor(graph, max_BRAM_util)
            result.append([graph, extra_inputs, extra_outputs, weights_reloading])

    result_split = []
    for i, r in enumerate(result):
        curr_graph, extra_inputs, extra_outputs, weights_reloading = r
        curr_graph = deepcopy(curr_graph)
        nodes_in, nodes_out = get_off_chip_mem_connections(curr_graph)
        read_points, write_points = add_off_chip_connections(
            curr_graph, nodes_in, nodes_out, gap_approx=gap_approx
        )

        (
            config,
            _,
            _,
            _,
            _,
        ) = self.initialize_optimizer_partition(
            graph=curr_graph,
            read_points=read_points,
            write_points=write_points,
            wr_factor=weights_reloading,
            init_time_limit=15,
        )

        if config is None:
            _logger.warning(
                "Initial design point can not be found, splitting the graph further"
            )
            result_split.append(i)

    for i in result_split:
        curr_graph, _, _, _ = result[i]
        del result[i]
        self.check_partition_fitting(
            curr_graph,
            partition_composer,
            max_BRAM_util,
            word_bytes,
            bram_type,
            brams_total,
            mem_words_per_cycle,
            result,
            original_graph,
            gap_approx=gap_approx,
            force_split=True,
        )

    return result


def initialize_optimizer_partition(
    self, graph, read_points, write_points, wr_factor, init_time_limit=90.0
):
    self.freeze_param = True

    config, mem_bw, _, _ = self.generate_random_config_partition(target_graph=graph)
    cost, dp_info = self.get_cost_partition(
        config,
        mem_bw,
        read_points,
        write_points,
        target_graph=graph,
        wr_factor=wr_factor,
    )
    slowest_nodes = None

    if cost is None:
        start_time = time.time()
        while time.time() - start_time < init_time_limit:
            x = float(time.time() - start_time)
            perc = 1 / (1 + math.exp(-0.1 * (x - 45)))
            config, mem_bw, _, _ = self.generate_random_config_partition(
                target_graph=graph, keep_percentage=perc
            )
            cost, dp_info = self.get_cost_partition(
                config,
                mem_bw,
                read_points,
                write_points,
                target_graph=graph,
                wr_factor=wr_factor,
            )

            if cost is not None:
                slowest_nodes = dp_info["slowestNodes"]
                break
        if cost is None:
            print(
                f"No configuration found after {init_time_limit} seconds. Aborting..."
            )
            return None, None, None, None, None

    prev_state = config
    prev_cost = cost
    solution_dp = dp_info
    solution_mem = mem_bw

    return prev_state, prev_cost, solution_dp, solution_mem, slowest_nodes


def run_optimizer_partition(self):
    # TODO: Searching for partition fitting or not to the device we assume a lower bram utilization than the provided one from the user (initial_max_bram_util) by 10 %.
    sub_partitions = self.check_partition_fitting(
        self.graph,
        self.partition_composer,
        self.config.initial_max_bram_util - 10,
        self.platform.word_bytes,
        self.platform.bram_Kbytes,
        self.platform.bram,
        self.platform.mem_words_per_cycle,
        [],
        gap_approx=self.gap_approx,
    )
    extra_reconfigurations = len(sub_partitions) - 1
    if extra_reconfigurations > 0:
        _logger.warning(
            f"Splitting original partition into {len(sub_partitions)} sub-partitions. A number of {extra_reconfigurations} extra reconfiguration(s) will be added."
        )

    mem_bw_list = []
    dp_info_list = []
    wr_list = []
    for i, sp in enumerate(sub_partitions):
        graph = sp[0]
        mem_in = sp[1]
        mem_out = sp[2]
        weights_reloading = sp[3]

        nodes_in, nodes_out = get_off_chip_mem_connections(graph)
        read_points, write_points = add_off_chip_connections(
            graph, nodes_in, nodes_out, gap_approx=self.gap_approx
        )
        visualize_graph(
            graph,
            os.getcwd()
            + "/fpga_modeling_reports/"
            + self.cnn_model_name
            + "/partition_graphs/"
            + self.part_name
            + "_split"
            + str(i),
            self.enable_wandb,
            f"{self.part_name}_split_{i}",
        )

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
            ) = self.initialize_optimizer_partition(
                graph=graph,
                read_points=read_points,
                write_points=write_points,
                wr_factor=weights_reloading,
            )

            if config is None:
                return None, None, None, None, None

            prev_state = config
            prev_cost = cost
            solution_dp = dp_info
            solution_mem = mem_bw

            current_temp = self.t_max
            # first_restart, second_restart, third_restart = True, True, True
            count = 0
            print(
                f"{Fore.LIGHTGREEN_EX}{'Temperature':<12} | {'Latency':<12} | {'Count':<8} | {'Best Latency':<12}"
            )
            while current_temp > self.t_min:
                # if self.enable_wandb:
                #     log_dict = {}
                #     log_dict["temperature"] = current_temp
                #     log_dict["latency"] = prev_cost
                #     wandb.log(log_dict)

                num_iterations = 0
                timeout_tmr_start = time.time()
                while (
                    num_iterations < self.iterationPerTemp
                    and time.time() - timeout_tmr_start < 2.5
                ):
                    # for _ in range(self.iterationPerTemp):
                    (
                        new_state,
                        new_mem_bw,
                        _,
                        _,
                    ) = self.generate_random_config_partition(
                        neighbours=True,
                        prev_state=prev_state,
                        slowest_nodes=slowest_nodes,
                        target_graph=graph,
                    )
                    new_cost, new_dp_info = self.get_cost_partition(
                        new_state,
                        new_mem_bw,
                        read_points,
                        write_points,
                        target_graph=graph,
                        wr_factor=weights_reloading,
                    )
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
                        if random.uniform(0, 1) < math.exp(
                            cost_diff / (current_temp * self.k)
                        ):
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
                if prev_cost < best_latency:
                    best_latency = prev_cost
                    best_solution_mem = solution_mem
                    best_solution_dp = solution_dp

                print(
                    f"{current_temp:<12.5e}   {prev_cost:<12.5e}   {count:<8d}  {Fore.LIGHTBLUE_EX}{best_latency:12.5e}",
                    end="\r",
                )

            if prev_cost < best_latency:
                best_latency = prev_cost
                best_solution_mem = solution_mem
                best_solution_dp = solution_dp

            print(
                f"{current_temp:<12.5e}   {prev_cost:<12.5e}   {count:<8d}  {Fore.LIGHTBLUE_EX}{best_latency:12.5e}",
                end="\r",
            )

        mem_bw_list.append(best_solution_mem)
        dp_info_list.append(best_solution_dp)
        wr_list.append(weights_reloading)

        print(
            f"\n\n{Fore.LIGHTBLUE_EX}Latency: {best_latency}{Fore.WHITE}\nFinal Memory IN {list(np.array(best_solution_mem[0]) * self.platform.mem_words_per_cycle)}, Memory OUT {list(np.array(best_solution_mem[1]) * self.platform.mem_words_per_cycle)}"
        )
        print(
            "Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={}({:.2f}), BRAM(%)={}({:.2f}), rateIn={}, RateOut={}, Depth={}({}), Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(  # \nPartition Configuration: {}
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
                # best_solution_dp["config"],
            )
        )
        print("*" * 60)
    return (
        self.platform.mem_words_per_cycle,
        mem_bw_list,
        dp_info_list,
        extra_reconfigurations,
        wr_list,
    )


def get_cost_partition(
    self,
    config,
    mem_bw,
    read_points,
    write_points,
    target_graph=None,
    branch_mem_update=None,
    wr_factor=1,
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
        wr_factor=wr_factor,
    )
    if dp_info["config"]:
        return dp_info["latency(S)"], dp_info
        # return -dp_info['GOP/s'], dp_info
    return None, None


def generate_random_config_partition(
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

    # keep_percentage = 0.95
    for node in config_nodes:
        if slowest_nodes is not None and node not in slowest_nodes:
            continue

        op_type = graph.nodes[node]["type"]
        hw = graph.nodes[node]["hw"]
        if isinstance(hw, GAP3DLayer):
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
            fine_feasible = utils.get_fine_feasible(
                kernel_size, keep_percentage=keep_percentage
            )
            coarse_in_factor = random.choice(coarse_in_feasible) / channels
            coarse_out_factor = random.choice(coarse_out_feasible) / filters
            fine_factor = random.choice(fine_feasible) / np.prod(np.array(kernel_size))
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
            fine_feasible = utils.get_fine_feasible(
                kernel_size, keep_percentage=keep_percentage
            )
            coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
            fine_factor = random.choice(fine_feasible) / np.prod(np.array(kernel_size))
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
        elif isinstance(hw, Activation3DLayer):
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
        elif isinstance(hw, ElementWise3DLayer):
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


def run_optimizer_partition_double_graph(self):
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
        self.platform.word_bytes,
        self.platform.bram_Kbytes,
        self.platform.bram,
        self.platform.max_BRAM_util,
    )

    nIN1 = len(mem_in_1)
    nOUT1 = len(mem_out_1)
    assert nIN1 > 0 and nOUT1 > 0, "No memory in/out nodes found"
    config_1, mem_bw_1 = self.generate_random_config_partition(
        target_graph=graph_1, n_in=nIN1, n_out=nOUT1
    )
    cost_1, dp_info_1 = self.get_cost_partition(
        config_1,
        mem_bw_1,
        len(mem_in_1),
        len(mem_out_1),
        target_graph=graph_1,
        branch_mem_update=bb1,
    )

    nIN2 = len(mem_in_2)
    nOUT2 = len(mem_out_2)
    assert nIN2 > 0 and nOUT2 > 0, "No memory in/out nodes found"
    config_2, mem_bw_2 = self.generate_random_config_partition(
        target_graph=graph_2, n_in=nIN2, n_out=nOUT2
    )
    cost_2, dp_info_2 = self.get_cost_partition(
        config_2,
        mem_bw_2,
        len(mem_in_2),
        len(mem_out_2),
        target_graph=graph_2,
        branch_mem_update=bb2,
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
                self.platform.word_bytes,
                self.platform.bram_Kbytes,
                self.platform.bram,
                self.config.max_bram_util,
            )

            nIN1 = len(mem_in_1)
            nOUT1 = len(mem_out_1)
            assert nIN1 > 0 and nOUT1 > 0, "No memory in/out nodes found"
            config_1, mem_bw_1 = self.generate_random_config_partition(
                target_graph=graph_1, n_in=nIN1, n_out=nOUT1
            )
            cost_1, dp_info_1 = self.get_cost_partition(
                config_1,
                mem_bw_1,
                len(mem_in_1),
                len(mem_out_1),
                target_graph=graph_1,
                branch_mem_update=bb1,
            )

            nIN2 = len(mem_in_2)
            nOUT2 = len(mem_out_2)
            assert nIN2 > 0 and nOUT2 > 0, "No memory in/out nodes found"
            config_2, mem_bw_2 = self.generate_random_config_partition(
                target_graph=graph_2, n_in=nIN2, n_out=nOUT2
            )
            cost_2, dp_info_2 = self.get_cost_partition(
                config_2,
                mem_bw_2,
                len(mem_in_2),
                len(mem_out_2),
                target_graph=graph_2,
                branch_mem_update=bb2,
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

    print("Temperature  |  Latency")
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
                    self.platform.word_bytes,
                    self.platform.bram_Kbytes,
                    self.platform.bram,
                    self.config.max_bram_util,
                )

            prev_state_1 = self.fix_inconsistent_config(prev_state_1, graph_1)
            prev_state_2 = self.fix_inconsistent_config(prev_state_2, graph_2)

            nIN1 = len(mem_in_1)
            nOUT1 = len(mem_out_1)
            assert nIN1 > 0 and nOUT1 > 0, "No memory in/out nodes found"
            new_state_1, new_mem_bw_1 = self.generate_random_config_partition(
                target_graph=graph_1,
                neighbours=True,
                prev_state=prev_state_1,
                n_in=nIN1,
                n_out=nOUT1,
            )
            new_cost_1, new_dp_info_1 = self.get_cost_partition(
                new_state_1,
                new_mem_bw_1,
                len(mem_in_1),
                len(mem_out_1),
                target_graph=graph_1,
                branch_mem_update=bb1,
            )

            nIN2 = len(mem_in_2)
            nOUT2 = len(mem_out_2)
            assert nIN2 > 0 and nOUT2 > 0, "No memory in/out nodes found"
            new_state_2, new_mem_bw_2 = self.generate_random_config_partition(
                target_graph=graph_2,
                neighbours=True,
                prev_state=prev_state_2,
                n_in=nIN2,
                n_out=nOUT2,
            )
            new_cost_2, new_dp_info_2 = self.get_cost_partition(
                new_state_2,
                new_mem_bw_2,
                len(mem_in_2),
                len(mem_out_2),
                target_graph=graph_2,
                branch_mem_update=bb2,
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
        f"\n\nLatency: {prev_cost}. volumes/s: {1 / prev_cost}.\nSolution 1: Memory IN {list(np.array(solution_mem_1[0]) * self.platform.mem_words_per_cycle)}, Memory OUT {list(np.array(solution_mem_1[1]) * self.platform.mem_words_per_cycle)}\nSolution 2: Memory IN {list(np.array(solution_mem_2[0]) * self.platform.mem_words_per_cycle)}, Memory OUT {list(np.array(solution_mem_2[1]) * self.platform.mem_words_per_cycle)}."
    )
    print(
        "Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(  # \nPartition Configuration: {}
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
            # solution_dp_1["config"],
        )
    )
    print(
        "Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(  # \nPartition Configuration: {}
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
            # solution_dp_2["config"],
        )
    )
    print("*" * 60)
    return (
        self.platform.mem_words_per_cycle,
        [solution_mem_1, solution_mem_2],
        [solution_dp_1, solution_dp_2],
        0,
    )
