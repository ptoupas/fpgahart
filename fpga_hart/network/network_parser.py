import json
import os
import random
import shutil
from copy import copy, deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scienceplots
from dotmap import DotMap

import wandb
from fpga_hart import _logger
from fpga_hart.layers.activation_3d import Activation3DLayer
from fpga_hart.layers.batchnorm_3d import BatchNorm3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise_3d import ElementWise3DLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap_3d import GAP3DLayer
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.optimizer.optimizer_helper import (
    add_off_chip_connections,
    calculate_wr_factor,
    get_minimum_resource_utilization,
    get_off_chip_mem_connections,
    get_worst_case_buffering,
)
from fpga_hart.parser.model_descriptor import ModelLayerDescriptor
from fpga_hart.partitions.partition_compose import PartitionComposer
from fpga_hart.partitions.partition_parser import PartitionParser
from fpga_hart.platform.platform import Platform
from fpga_hart.utils.graph_manipulation import visualize_graph
from fpga_hart.utils.utils import get_conv_type, get_pool_type, num_sort

plt.style.use(["science", "ieee", "grid"])


@dataclass
class NetworkParser(ModelLayerDescriptor):
    batch_size: int
    num_reconfig_points: int
    allowed_reconfig_layers: list
    min_partition_layers: int
    max_partition_layers: int
    gap_approx: bool
    platform: Platform
    config: DotMap
    enable_wandb: bool

    def __post_init__(self) -> None:
        ModelLayerDescriptor.__post_init__(self)  # Initialize the parent class
        # _logger.setLevel(level=logging.DEBUG)

        self.partition_composer = PartitionComposer(
            max_DSP_util=self.config.max_dsp_util,
            max_BRAM_util=self.config.max_bram_util,
            platform=self.platform,
        )
        self.partition_parser = PartitionParser(
            model_name=self.model_name,
            se_block=False,
            gap_approx=self.gap_approx,
            singlethreaded=False,
            per_layer_plot=False,
            platform=self.platform,
            config=self.config,
            enable_wandb=self.enable_wandb,
        )

    def get_reconfig_points(self):
        available_reconfig_points = [
            layer
            for layer in self.layers
            if self.layers[layer]["operation"] in self.allowed_reconfig_layers
        ]

        # Remove the network input and output layers from the available reconfig points
        remove_io_nodes = []
        for rp in available_reconfig_points:
            if (
                self.layers[rp]["node_in"][0] in self.initial_model_inputs
                or self.layers[rp]["node_out"] in self.initial_model_outputs
            ):
                remove_io_nodes.append(rp)
        for rp in remove_io_nodes:
            available_reconfig_points.remove(rp)

        rp_idx = 0
        while rp_idx <= self.num_reconfig_points:
            reconfig_points = random.sample(
                available_reconfig_points, self.num_reconfig_points
            )
            reconfig_points.sort(key=num_sort)

            rp_idx = 0
            layers_counter = 0
            for layer in self.layers:
                if (
                    rp_idx < self.num_reconfig_points
                    and layer == reconfig_points[rp_idx]
                ):
                    if (
                        layers_counter < self.min_partition_layers
                        or layers_counter > self.max_partition_layers
                    ):
                        _logger.debug(
                            f"Reconfig point {reconfig_points[rp_idx]} number of layers {layers_counter} is not within the allowed range {self.min_partition_layers} <= num layers <= {self.max_partition_layers}. Reconfiguring..."
                        )
                        break
                    else:
                        layers_counter = 0
                        rp_idx += 1
                layers_counter += 1
            if (
                layers_counter < self.min_partition_layers
                or layers_counter > self.max_partition_layers
            ):
                _logger.debug(
                    f"Final reconfig point number of layers {layers_counter} is not within the allowed range {self.min_partition_layers} <= num layers <= {self.max_partition_layers}. Reconfiguring..."
                )
            else:
                rp_idx += 1
        return reconfig_points

    def get_partitions(self):
        reconfig_points = self.get_reconfig_points()
        model_layers = list(self.layers.keys())

        partitions = dict()
        for i, rp in enumerate(reconfig_points):
            partition_specs = dict()
            current_partition = []
            for layer in model_layers:
                if layer == rp:
                    break
                current_partition.append(layer)

            partition_name = f"part_{i}"
            partition_specs["layers"] = current_partition

            part_graph = self.create_graph(current_partition)
            partition_specs["graph"] = part_graph

            bram_util, dsp_util, layers_bram, branch_bram = (
                self.get_partition_utilization(part_graph)
            )

            _logger.debug(
                f"Partition {partition_name} has {len(current_partition)} layers, BRAM utilization = {bram_util:.2f} (layers BRAM = {layers_bram:.2f}, branch BRAM = {branch_bram:.2f}), DSP utilization = {dsp_util:.2f}"
            )
            partition_specs["valid"] = (
                True
                if bram_util <= self.config.initial_max_bram_util
                and dsp_util <= self.config.max_dsp_util
                else False
            )
            partition_specs["weights_reloading"] = 1

            partition_specs["total_bram"] = bram_util
            partition_specs["total_dsp"] = dsp_util
            partition_specs["layers_bram"] = layers_bram
            partition_specs["branch_bram"] = branch_bram

            partitions[partition_name] = partition_specs
            model_layers = model_layers[model_layers.index(rp) :]

        partition_specs = dict()
        if len(reconfig_points) == 0:
            partition_name = "part_0"
        else:
            partition_name = f"part_{i + 1}"
        partition_specs["layers"] = model_layers

        part_graph = self.create_graph(model_layers)
        partition_specs["graph"] = part_graph
        bram_util, dsp_util, layers_bram, branch_bram = self.get_partition_utilization(
            part_graph
        )

        _logger.debug(
            f"Partition {partition_name} has {len(current_partition)} layers, BRAM utilization = {bram_util:.2f} (layers BRAM = {layers_bram:.2f}, branch BRAM = {branch_bram:.2f}), DSP utilization = {dsp_util:.2f}"
        )
        partition_specs["valid"] = (
            True
            if bram_util <= self.config.initial_max_bram_util
            and dsp_util <= self.config.max_dsp_util
            else False
        )
        partition_specs["weights_reloading"] = 1

        partition_specs["total_bram"] = bram_util
        partition_specs["total_dsp"] = dsp_util
        partition_specs["layers_bram"] = layers_bram
        partition_specs["branch_bram"] = branch_bram

        partitions[partition_name] = partition_specs

        return partitions

    def visualize_partitions(self, partitions):
        if not os.path.exists(
            os.getcwd()
            + "/fpga_modeling_reports/"
            + self.model_name
            + "/throughput/model_graphs/"
        ):
            os.makedirs(
                os.getcwd()
                + "/fpga_modeling_reports/"
                + self.model_name
                + "/throughput/model_graphs/"
            )

        for fp in os.listdir(
            os.getcwd()
            + "/fpga_modeling_reports/"
            + self.model_name
            + "/throughput/model_graphs/"
        ):
            os.remove(
                os.getcwd()
                + "/fpga_modeling_reports/"
                + self.model_name
                + "/throughput/model_graphs/"
                + fp
            )

        for part, specs in partitions.items():
            graph = deepcopy(specs["graph"])
            nodes_in, nodes_out = get_off_chip_mem_connections(graph)
            _, _ = add_off_chip_connections(graph, nodes_in, nodes_out, self.gap_approx)
            visualize_graph(
                graph,
                os.getcwd()
                + "/fpga_modeling_reports/"
                + self.model_name
                + "/throughput/model_graphs/"
                + part,
                self.enable_wandb,
                part,
                specs["valid"],
            )

    def get_partition_utilization(self, graph, wr_factor=1):
        total_bram_util = 0
        total_dsp_util = 0
        for layer in nx.topological_sort(graph):
            # TODO: This has to change into get_resource_utilization with a config file of the hardware nodes so it can be used during optimization
            bram_util, dsp_util, pipeline_depth, _ = get_minimum_resource_utilization(
                graph.nodes[layer]["hw"], gap_approx=self.gap_approx
            )
            _logger.debug(
                f"Layer {layer}: BRAM utilization = {bram_util:.2f}, DSP utilization = {dsp_util:.2f}, Pipeline depth = {pipeline_depth}"
            )
            total_bram_util += bram_util
            total_dsp_util += dsp_util
            # if total_bram_util > self.config.initial_max_bram_util or total_dsp_util > self.config.max_dsp_util:
            #     _logger.warning(f"Partition BRAM utilization = {total_bram_util:.2f}, DSP utilization = {total_dsp_util:.2f}")

        layers_bram_util = deepcopy(total_bram_util)
        _, min_bram_util = get_worst_case_buffering(
            deepcopy(graph),
            self.partition_composer,
            self.platform.mem_words_per_cycle,
            self.platform.word_bytes,
            self.platform.bram_Kbytes,
            self.platform.bram,
            self.gap_approx,
            wr_factor=wr_factor,
        )
        branch_buffering_bram_util = deepcopy(min_bram_util)
        total_bram_util += min_bram_util

        return (
            total_bram_util,
            total_dsp_util,
            layers_bram_util,
            branch_buffering_bram_util,
        )

    def update_partitions(self, partitions):
        for part in partitions.values():
            bram_util, dsp_util, layers_bram, branch_bram = (
                self.get_partition_utilization(part["graph"])
            )

            prev_part_valid = part["valid"]
            part["valid"] = (
                True
                if bram_util <= self.config.initial_max_bram_util
                and dsp_util <= self.config.max_dsp_util
                else False
            )
            assert prev_part_valid == part["valid"], "Partition validity changed"

            prev_total_bram = part["total_bram"]
            prev_total_dsp = part["total_dsp"]
            prev_layers_bram = part["layers_bram"]
            prev_branch_bram = part["branch_bram"]
            part["total_bram"] = bram_util
            part["total_dsp"] = dsp_util
            part["layers_bram"] = layers_bram
            part["branch_bram"] = branch_bram

            assert prev_total_bram == part["total_bram"], (
                "Total BRAM utilization changed"
            )
            assert prev_total_dsp == part["total_dsp"], "Total DSP utilization changed"
            assert prev_layers_bram == part["layers_bram"], (
                "Layers BRAM utilization changed"
            )
            assert prev_branch_bram == part["branch_bram"], (
                "Branch BRAM utilization changed"
            )

        return partitions

    def validate_partitions(self, partitions):
        partitions = self.update_partitions(partitions)

        # Check if all layers are assigned to a partition
        for origin_layer in self.layers:
            layer_found = False
            for part in partitions.values():
                if origin_layer in part["layers"]:
                    layer_found = True
                    break
            if not layer_found:
                raise ValueError(f"Layer {origin_layer} not found in any partition")

        # Check if any layer is assigned to more than one partition
        model_layers = copy(list(self.layers.keys()))
        for part in partitions.values():
            for layer in part["layers"]:
                if layer in model_layers:
                    model_layers.remove(layer)
                else:
                    raise ValueError(
                        f"Layer {layer} is assigned to more than one partition"
                    )
        assert len(model_layers) == 0, (
            f"Layers {model_layers} are not assigned to any partition"
        )

        invalid_partitions = []
        for part_name, part in partitions.items():
            if not part["valid"]:
                _logger.warning(f"Partition {part_name} is invalid")
                invalid_partitions.append(part_name)
            if len(part["layers"]) < 1:
                _logger.warning(f"Partition {part_name} has no layers")
                invalid_partitions.append(part_name)
            if part["weights_reloading"] == -1:
                _logger.warning(f"Partition {part_name} has WR factor of -1")
                invalid_partitions.append(part_name)

        return invalid_partitions

    def add_key_and_shift_dict(self, my_dict, key_to_add, key_number):
        num_keys = len(my_dict.keys())
        my_dict["part_" + str(num_keys)] = dict()

        for i in range(num_keys, key_number, -1):
            old_key = "part_" + str(i - 1)
            new_key = "part_" + str(i)
            my_dict[new_key] = my_dict[old_key]

        my_dict[key_to_add] = dict()
        my_dict["part_" + str(key_number - 1)] = dict()

        return my_dict

    def split_partition(self, partitions, name):
        part_number = int(name.split("_")[-1])
        part = partitions[name]
        layers = part["layers"]
        num_layers = len(layers)

        assert len(layers) > 1, "Trying to split a partition with only one layer"

        # Add a new partition to the dictionary
        partitions = self.add_key_and_shift_dict(
            partitions, "part_" + str(part_number + 1), part_number + 1
        )

        # Split the partition in two
        sub_partition_1 = layers[: num_layers // 2]
        part_1_valid = self.update_single_partition(partitions, name, sub_partition_1)
        sub_partition_2 = layers[num_layers // 2 :]
        part_2_valid = self.update_single_partition(
            partitions, "part_" + str(part_number + 1), sub_partition_2
        )

        if not part_1_valid:
            partitions = self.split_partition(partitions, name)
        if not part_2_valid:
            partitions = self.split_partition(
                partitions, "part_" + str(part_number + 1)
            )

        return partitions

    def remove_key_and_shift_dict(self, my_dict, key_to_remove, key_number):
        num_keys = len(my_dict.keys())

        del my_dict[key_to_remove]

        for i in range(key_number + 1, num_keys):
            old_key = "part_" + str(i)
            new_key = "part_" + str(i - 1)
            my_dict[new_key] = my_dict.pop(old_key)

        return my_dict

    def merge_partition(self, partitions, name, direction):
        part_number = int(name.split("_")[-1])

        if direction == "prev":
            merge_part_number = part_number - 1
            merge_part = f"part_{merge_part_number}"
            if partitions[merge_part].pop("graph", None) is None:
                raise ValueError(f"Cannot find graph for partition {merge_part}")
            valid_part = self.update_single_partition(
                partitions,
                merge_part,
                partitions[merge_part]["layers"] + partitions[name]["layers"],
            )
            self.remove_key_and_shift_dict(partitions, name, part_number)
        elif direction == "next":
            merge_part_number = part_number + 1
            merge_part = f"part_{merge_part_number}"
            if partitions[name].pop("graph", None) is None:
                raise ValueError(f"Cannot find graph for partition {name}")
            valid_part = self.update_single_partition(
                partitions,
                name,
                partitions[name]["layers"] + partitions[merge_part]["layers"],
            )
            self.remove_key_and_shift_dict(partitions, merge_part, merge_part_number)

        # assert valid_part, f"Invalid partition after merging part {name} with part {merge_part}"
        if not valid_part:
            _logger.warning(
                f"Invalid partition after merging part '{name}' with part '{merge_part}'"
            )

        return partitions, valid_part

    def update_single_partition(self, partitions, name, layers, wr_factor=None):
        if wr_factor is None:
            graph_new = self.create_graph(layers)
            wr_factor = calculate_wr_factor(
                graph_new, self.config.initial_max_bram_util
            )
        else:
            graph_new = partitions[name]["graph"]
        if wr_factor == -1:
            partitions[name]["layers"] = layers
            partitions[name]["graph"] = graph_new
            partitions[name]["valid"] = False
            partitions[name]["weights_reloading"] = -1
            return False

        bram_util, dsp_util, layers_bram, branch_bram = self.get_partition_utilization(
            graph_new, wr_factor=wr_factor
        )
        part_validity = (
            True
            if bram_util <= self.config.initial_max_bram_util
            and dsp_util <= self.config.max_dsp_util
            else False
        )

        partitions[name]["layers"] = layers
        partitions[name]["graph"] = graph_new
        partitions[name]["valid"] = part_validity
        partitions[name]["weights_reloading"] = wr_factor
        partitions[name]["total_bram"] = bram_util
        partitions[name]["total_dsp"] = dsp_util
        partitions[name]["layers_bram"] = layers_bram
        partitions[name]["branch_bram"] = branch_bram

        return part_validity

    def get_candidate_merge_partition(self, partitions, blacklisted_partitions):
        min_bram_util = float("inf")
        min_bram_part_name = None
        for name, specs in partitions.items():
            if specs["total_bram"] < min_bram_util:
                min_bram_util = specs["total_bram"]
                min_bram_part_name = name

        assert min_bram_part_name is not None, (
            "No partition found with minimum BRAM utilization"
        )

        min_bram_part = partitions[min_bram_part_name]
        if (
            min_bram_part["total_bram"] < self.config.merge_bram_threshold
            and min_bram_part_name not in blacklisted_partitions
        ):
            return min_bram_part_name

        return None

    def get_partition_to_merge(self, partitions, name):
        part_number = int(name.split("_")[-1])
        part_bram_util = partitions[name]["total_bram"]
        part_wr_factor = partitions[name]["weights_reloading"]
        direction = None

        if part_number == 0:
            next_part_name = f"part_{part_number + 1}"
            next_part_bram_util = partitions[next_part_name]["total_bram"]
            next_part_wr_factor = partitions[next_part_name]["weights_reloading"]

            if part_bram_util + next_part_bram_util < self.config.initial_max_bram_util:
                direction = "next"
        elif part_number == len(partitions.keys()) - 1:
            prev_part_name = f"part_{part_number - 1}"
            prev_part_bram_util = partitions[prev_part_name]["total_bram"]
            prev_part_wr_factor = partitions[prev_part_name]["weights_reloading"]

            if part_bram_util + prev_part_bram_util < self.config.initial_max_bram_util:
                direction = "prev"
        else:
            prev_part_name = f"part_{part_number - 1}"
            prev_part_bram_util = partitions[prev_part_name]["total_bram"]
            prev_part_wr_factor = partitions[prev_part_name]["weights_reloading"]

            next_part_name = f"part_{part_number + 1}"
            next_part_bram_util = partitions[next_part_name]["total_bram"]
            next_part_wr_factor = partitions[next_part_name]["weights_reloading"]

            if (
                part_bram_util + prev_part_bram_util
                <= part_bram_util + next_part_bram_util
            ) and (
                part_bram_util + prev_part_bram_util < self.config.initial_max_bram_util
            ):
                direction = "prev"
            elif (
                part_bram_util + next_part_bram_util
                < part_bram_util + prev_part_bram_util
            ) and (
                part_bram_util + next_part_bram_util < self.config.initial_max_bram_util
            ):
                direction = "next"

        return direction

    def refine_partitions(self, partitions, invalid_partitions):
        for part_name in invalid_partitions:
            _logger.info(f"Refining partition {part_name}")
            partition = partitions[part_name]
            assert len(partition["layers"]) > 0, (
                f"Partition {part_name} has no layers assigned to it."
            )

            wr_factor = calculate_wr_factor(
                partition["graph"], self.config.initial_max_bram_util
            )
            if wr_factor > 1:
                part_valid = self.update_single_partition(
                    partitions, part_name, partition["layers"], wr_factor=wr_factor
                )
                _logger.debug(
                    f"WR factor for partition {part_name} is {wr_factor}. Validity is {part_valid}."
                )
                if not part_valid:
                    return self.split_partition(partitions, part_name)
            else:
                _logger.warning(
                    f"Partition {part_name} cannot fit in the FPGA even after WR. Splitting it into two partitions."
                )
                return self.split_partition(partitions, part_name)

        return partitions

    def parse(self):
        network_partitions = self.get_partitions()
        while invalid_partitions := self.validate_partitions(network_partitions):
            network_partitions = self.refine_partitions(
                network_partitions, invalid_partitions
            )

        if self.config.enable_partition_merging:
            blacklisted_parts = []
            cm_part = self.get_candidate_merge_partition(
                network_partitions, blacklisted_parts
            )
            while cm_part is not None:
                direction = self.get_partition_to_merge(network_partitions, cm_part)

                if direction is not None:
                    network_partitions_old = deepcopy(network_partitions)
                    network_partitions, valid = self.merge_partition(
                        network_partitions,
                        cm_part,
                        direction,
                    )
                    if not valid:
                        _logger.warning(
                            f"Invalid partition after merging {cm_part} with {direction} partition"
                        )
                        blacklisted_parts.append(cm_part)
                        network_partitions = network_partitions_old
                    else:
                        blacklisted_parts.clear()
                else:
                    blacklisted_parts.append(cm_part)
                cm_part = self.get_candidate_merge_partition(
                    network_partitions, blacklisted_parts
                )

        assert self.validate_partitions(network_partitions) == [], (
            "Partitions are not valid after merging."
        )
        # self.visualize_partitions(network_partitions)

        num_dev_reconfig = len(network_partitions) - 1
        _logger.info(f"Initial number of device reconfigurations: {num_dev_reconfig}.")

        partition_graphs_path = os.path.join(
            os.getcwd(),
            "fpga_modeling_reports",
            self.model_name,
            "partition_graphs",
        )
        if os.path.exists(partition_graphs_path):
            for file in os.listdir(partition_graphs_path):
                os.unlink(os.path.join(partition_graphs_path, file))
        for part_name, specs in network_partitions.items():
            num_dev_reconfig += self.partition_parser.model_partition(
                specs["layers"], name=part_name
            )

        assert self.validate_partitions(network_partitions) == [], (
            "Final optimized partitions are not valid."
        )

        _logger.info(f"Final number of device reconfigurations: {num_dev_reconfig}.")

        for key in self.partition_parser.model_avg_metrics:
            self.partition_parser.model_avg_metrics[key] = (
                self.partition_parser.df[key]
                .repeat(self.partition_parser.df["Times Repeated"].to_list())
                .mean()
            )
        self.partition_parser.model_avg_metrics["latency(C) Sum"] = int(
            (
                self.partition_parser.df["latency(C)"]
                * self.partition_parser.df["Times Repeated"]
            ).sum()
        )
        self.partition_parser.model_avg_metrics["latency(S) Sum"] = (
            self.partition_parser.df["latency(S)"]
            * self.partition_parser.df["Times Repeated"]
        ).sum()
        self.partition_parser.model_avg_metrics["GOPs Sum"] = (
            self.partition_parser.df["GOPs"]
            * self.partition_parser.df["Times Repeated"]
        ).sum()
        self.partition_parser.model_avg_metrics["depth Sum"] = int(
            (
                self.partition_parser.df["depth"]
                * self.partition_parser.df["Times Repeated"]
            ).sum()
        )

        if not self.enable_wandb:
            log_results_path = os.path.join(
                os.getcwd(),
                "fpga_modeling_reports",
                self.model_name,
                "partition_results",
            )
            if not os.path.exists(log_results_path):
                os.makedirs(log_results_path)
            else:
                for file in os.listdir(log_results_path):
                    os.unlink(os.path.join(log_results_path, file))

        batch_size = np.arange(1, 500, 1)
        lat_sec = np.sum(
            np.array([
                (
                    (
                        self.partition_parser.df["latency(C)"][idx]
                        - self.partition_parser.df["depth"][idx]
                    )
                    * batch_size
                    + self.partition_parser.df["depth"][idx]
                )
                / (self.platform.clock_freq * 1e6)
                for idx in range(self.partition_parser.df["latency(C)"].size)
            ]),
            axis=0,
        ) + (self.platform.reconfiguration_time * num_dev_reconfig)
        plt.plot(batch_size, lat_sec)
        plt.xlabel("Batch Size")
        plt.ylabel("Seconds")
        plt.title("Latency vs Batch Size")
        if self.enable_wandb:
            wandb.log({"Latency vs Batch Size": plt})
        else:
            plt.savefig(os.path.join(log_results_path, "latency_vs_batch_size.png"))
        through_gops_sec = np.sum(
            np.array([
                self.partition_parser.df["GOPs"][idx] * batch_size
                for idx in range(self.partition_parser.df["GOPs"].size)
            ]),
            axis=0,
        )
        through_gops_sec = through_gops_sec / lat_sec
        plt.cla()
        plt.clf()
        plt.plot(batch_size, through_gops_sec)
        plt.xlabel("Batch Size")
        plt.ylabel("GOPs/s")
        plt.title("Throughput (GOPs/s) vs Batch Size")
        if self.enable_wandb:
            wandb.log({"Throughput (GOPs/s) vs Batch Size": plt})
        else:
            plt.savefig(
                os.path.join(log_results_path, "throughput_gops_vs_batch_size.png")
            )
        through_vols_sec = batch_size / lat_sec
        plt.cla()
        plt.clf()
        plt.plot(batch_size, through_vols_sec)
        plt.xlabel("Batch Size")
        plt.ylabel("Volumes/s")
        plt.title("Throughput (Volumes/s) vs Batch Size")
        if self.enable_wandb:
            wandb.log({"Throughput (Volumes/s) vs Batch Size": plt})
        else:
            plt.savefig(
                os.path.join(log_results_path, "throughput_vols_vs_batch_size.png")
            )
        gops_sec_dsp = through_gops_sec / self.platform.dsp
        plt.cla()
        plt.clf()
        plt.plot(batch_size, gops_sec_dsp)
        plt.xlabel("Batch Size")
        plt.ylabel("GOPs/s/DSP")
        plt.title("Throughput (GOPs/s/DSP) vs Batch Size")
        if self.enable_wandb:
            wandb.log({"Throughput (GOPs/s/DSP) vs Batch Size": plt})
        else:
            plt.savefig(
                os.path.join(log_results_path, "throughput_gops_dsp_vs_batch_size.png")
            )
        gops_sec_dsp_cycle = (gops_sec_dsp / self.platform.clock_freq) * 1e3
        plt.cla()
        plt.clf()
        plt.plot(batch_size, gops_sec_dsp_cycle)
        plt.xlabel("Batch Size")
        plt.ylabel("GOPs/s/DSP/Cycle")
        plt.title("Throughput (GOPs/s/DSP/Cycle) vs Batch Size")
        if self.enable_wandb:
            wandb.log({"Throughput (GOPs/s/DSP/Cycle) vs Batch Size": plt})
        else:
            plt.savefig(
                os.path.join(
                    log_results_path, "throughput_gops_dsp_cycle_vs_batch_size.png"
                )
            )

        self.partition_parser.model_avg_metrics["latency(S)-reconfig"] = {
            "Batch 1": lat_sec[0],
            "Batch 30": lat_sec[29],
            "Batch 100": lat_sec[99],
            "Batch 250": lat_sec[249],
        }

        self.partition_parser.model_avg_metrics["GOPs/s"] = {
            "Batch 1": through_gops_sec[0],
            "Batch 30": through_gops_sec[29],
            "Batch 100": through_gops_sec[99],
            "Batch 250": through_gops_sec[249],
        }
        self.partition_parser.model_avg_metrics["Volumes/s"] = {
            "Batch 1": through_vols_sec[0],
            "Batch 30": through_vols_sec[29],
            "Batch 100": through_vols_sec[99],
            "Batch 250": through_vols_sec[249],
        }
        self.partition_parser.model_avg_metrics["GOPs/s/DSP"] = {
            "Batch 1": gops_sec_dsp[0],
            "Batch 30": gops_sec_dsp[29],
            "Batch 100": gops_sec_dsp[99],
            "Batch 250": gops_sec_dsp[249],
        }
        self.partition_parser.model_avg_metrics["GOPs/s/DSP/cycle"] = {
            "Batch 1": gops_sec_dsp_cycle[0],
            "Batch 30": gops_sec_dsp_cycle[29],
            "Batch 100": gops_sec_dsp_cycle[99],
            "Batch 250": gops_sec_dsp_cycle[249],
        }

        del self.partition_parser.model_avg_metrics["latency(C)"]
        del self.partition_parser.model_avg_metrics["latency(S)"]
        del self.partition_parser.model_avg_metrics["GOPs"]
        del self.partition_parser.model_avg_metrics["depth"]

        if self.enable_wandb:
            wandb.log(self.partition_parser.model_avg_metrics)
            wandb.log({
                "Partition Results": wandb.Table(dataframe=self.partition_parser.df)
            })
        else:
            with open(self.partition_parser.partition_model_file, "r") as fp:
                dictObj = json.load(fp)

            dictObj["metrics"] = self.partition_parser.model_avg_metrics

            with open(self.partition_parser.partition_model_file, "w") as json_file:
                json.dump(dictObj, json_file, indent=2)

        # for part, specs in network_partitions.items():
        #     print(f"Partition {part} has {len(specs['layers'])} layers and BRAM utilization {specs['total_bram']:.2f}, wr factor of {specs['weights_reloading']}")
        #     for node in specs["graph"].nodes:
        #         hw = specs["graph"].nodes[node]["hw"]
        #         print(f"Layer {node} has input shape {hw.input_shape}, output shape {hw.output_shape}")

        # TODO: SOS! During the WR calculation the shapes of the layers inside the partition are being changed. This has to be taken into consideration when merging spliting or moving layers between partitions where the WR factor is greater than 1. WR factor is probably INCORRECT for the partitions that have been split or merged.
        # TODO: Instead of generating completely new partitions we can have a new transform that alters a bit the existing partitions by adding or removing layers from previous or next partitions.
        # TODO: I dont like the thing that layers partitions and network are not being connected somehow. It would be nice to have a way to connect them and build the network from the partitions and the partitions from the layers.

    def create_graph(self, partition: list) -> nx.DiGraph:
        graph = nx.DiGraph()
        # _logger.info("*" * 40)
        for layer in partition:
            # _logger.info("Adding {} layer to graph...".format(layer))
            if self.layers[layer]["operation"] == "GlobalAveragePool":
                hw_layer = GAP3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                    self.platform,
                )
                layer_type = self.layers[layer]["operation"]
            elif self.layers[layer]["operation"] == "Conv":
                hw_layer = Convolutional3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                    self.platform,
                )
                layer_type = self.layers[layer]["operation"]
            elif (
                self.layers[layer]["operation"] == "MaxPool"
                or self.layers[layer]["operation"] == "AveragePool"
            ):
                hw_layer = Pooling3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                    self.platform,
                )
                layer_type = "Pooling"
            elif (
                self.layers[layer]["operation"] == "Relu"
                or self.layers[layer]["operation"] == "Sigmoid"
                or self.layers[layer]["operation"] == "Swish"
            ):
                hw_layer = Activation3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                    self.platform,
                )
                layer_type = "Activation"
            elif (
                self.layers[layer]["operation"] == "Mul"
                or self.layers[layer]["operation"] == "Add"
            ):
                hw_layer = ElementWise3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                    self.platform,
                )
                layer_type = "ElementWise"
            elif (
                self.layers[layer]["operation"] == "Gemm"
                or self.layers[layer]["operation"] == "MatMul"
            ):
                layer_type = self.layers[layer]["operation"]
                hw_layer = FCLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                    self.platform,
                )
            elif self.layers[layer]["operation"] == "BatchNormalization":
                layer_type = self.layers[layer]["operation"]
                hw_layer = BatchNorm3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                    self.platform,
                )
            else:
                assert False, "{} operation in layer {} is not supported".format(
                    self.layers[layer]["operation"], layer
                )
            if self.layers[layer]["operation"] == "Conv":
                hw_type = get_conv_type(
                    layer=self.layers[layer],
                    discriminate_kernel_size=True,
                    discriminate_stide=False,
                    discriminate_padding=False,
                )
            elif self.layers[layer]["operation"] in ["MaxPool", "AveragePool"]:
                hw_type = get_pool_type(
                    layer=self.layers[layer],
                    discriminate_kernel_size=True,
                    discriminate_stide=False,
                    discriminate_padding=False,
                )
            else:
                hw_type = layer_type

            if self.layers[layer]["branching"]:
                layer_mode = "split"
            elif layer_type == "ElementWise":
                layer_mode = "merge"
            else:
                layer_mode = "sequential"

            graph.add_node(
                layer,
                type=layer_type,
                hw=hw_layer,
                hw_type=hw_type,
                layer_mode=layer_mode,
            )
        # _logger.info("*" * 40)

        edges = []
        for name in graph.nodes():
            inputs = self.layers[name]["node_in"]
            outputs = self.layers[name]["node_out"]

            for conn_node in self.connected_nodes(partition, outputs):
                edges.append((name, conn_node))

        for edge in edges:
            graph.add_edge(*edge)

        return graph

    def connected_nodes(self, partition, node_id):
        nodes = []
        for layer in partition:
            if node_id in self.layers[layer]["node_in"]:
                nodes.append(layer)
        return nodes
