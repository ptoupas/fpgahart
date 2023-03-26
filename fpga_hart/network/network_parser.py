import os
import random
from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import scienceplots

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
    add_off_chip_connections, calculate_wr_factor,
    get_minimum_resource_utilization, get_off_chip_mem_connections,
    get_worst_case_buffering)
from fpga_hart.parser.model_descriptor import ModelLayerDescriptor
from fpga_hart.partitions.partition_compose import PartitionComposer
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
    config: wandb.Config
    enable_wandb: bool

    def __post_init__(self) -> None:
        ModelLayerDescriptor.__post_init__(self)  # Initialize the parent class
        # _logger.setLevel(level=logging.DEBUG)

        self.partition_composer = PartitionComposer(
            max_DSP_util=self.config.max_dsp_util, max_BRAM_util=self.config.max_bram_util, platform=self.platform
        )

    def get_reconfig_points(self):
        available_reconfig_points = [layer for layer in self.layers if self.layers[layer]['operation'] in self.allowed_reconfig_layers]

        # Remove the network input and output layers from the available reconfig points
        remove_io_nodes = []
        for rp in available_reconfig_points:
            if self.layers[rp]['node_in'][0] in self.initial_model_inputs or self.layers[rp]['node_out'] in self.initial_model_outputs:
                remove_io_nodes.append(rp)
        for rp in remove_io_nodes:
            available_reconfig_points.remove(rp)

        rp_idx = 0
        while rp_idx <= self.num_reconfig_points:
            reconfig_points = random.sample(available_reconfig_points, self.num_reconfig_points)
            reconfig_points.sort(key=num_sort)

            rp_idx = 0
            layers_couter = 0
            for layer in self.layers:
                if rp_idx < self.num_reconfig_points and layer == reconfig_points[rp_idx]:
                    if layers_couter < self.min_partition_layers or layers_couter > self.max_partition_layers:
                        _logger.debug(f"Reconfig point {reconfig_points[rp_idx]} number of layers {layers_couter} is not within the allowed range {self.min_partition_layers} <= num layers <= {self.max_partition_layers}. Reconfiguring...")
                        break
                    else:
                        layers_couter = 0
                        rp_idx += 1
                layers_couter += 1
            if layers_couter < self.min_partition_layers or layers_couter > self.max_partition_layers:
                _logger.debug(f"Final reconfig point number of layers {layers_couter} is not within the allowed range {self.min_partition_layers} <= num layers <= {self.max_partition_layers}. Reconfiguring...")
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
            bram_util, dsp_util, layers_bram, branch_bram = self.get_partition_utilization(part_graph)

            partition_specs["valid"] = True if bram_util <= self.config.initial_max_bram_util and dsp_util <= self.config.max_dsp_util else False
            partition_specs["weights_reloading"] = 1

            partition_specs["total_bram"] = bram_util
            partition_specs["total_dsp"] = dsp_util
            partition_specs["layers_bram"] = layers_bram
            partition_specs["branch_bram"] = branch_bram

            partitions[partition_name] = partition_specs
            model_layers = model_layers[model_layers.index(rp):]

        partition_specs = dict()
        if len(reconfig_points) == 0:
            partition_name = "part_0"
        else:
            partition_name = f"part_{i+1}"
        partition_specs["layers"] = model_layers

        part_graph = self.create_graph(model_layers)
        partition_specs["graph"] = part_graph
        bram_util, dsp_util, layers_bram, branch_bram = self.get_partition_utilization(part_graph)

        partition_specs["valid"] = True if bram_util <= self.config.initial_max_bram_util and dsp_util <= self.config.max_dsp_util else False
        partition_specs["weights_reloading"] = 1

        partition_specs["total_bram"] = bram_util
        partition_specs["total_dsp"] = dsp_util
        partition_specs["layers_bram"] = layers_bram
        partition_specs["branch_bram"] = branch_bram

        partitions[partition_name] = partition_specs

        return partitions

    def visualize_partitions(self, partitions):
        if not os.path.exists(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/"):
            os.makedirs(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/")

        for fp in os.listdir(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/"):
            os.remove(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/" + fp)

        for part, specs in partitions.items():
            graph = deepcopy(specs["graph"])
            nodes_in, nodes_out = get_off_chip_mem_connections(graph)
            _, _ = add_off_chip_connections(
                graph, nodes_in, nodes_out, self.gap_approx)
            visualize_graph(
                graph,
                os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/" + part,
                self.enable_wandb,
                part,
                specs["valid"]
            )

    def get_partition_utilization(self, graph, wr_factor=1):
        total_bram_util = 0
        total_dsp_util = 0
        for layer in nx.topological_sort(graph):
            # TODO: This has to change into get_resource_utilization with a config file of the hardware nodes so it can be used during optimization
            bram_util, dsp_util, pipeline_depth, _ = get_minimum_resource_utilization(graph.nodes[layer]["hw"], gap_approx=self.gap_approx)
            _logger.debug(f"Layer {layer}: BRAM utilization = {bram_util:.2f}, DSP utilization = {dsp_util:.2f}, Pipeline depth = {pipeline_depth}")
            total_bram_util += bram_util
            total_dsp_util += dsp_util
            if total_bram_util > self.config.initial_max_bram_util or total_dsp_util > self.config.max_dsp_util:
                _logger.warning(f"Partition BRAM utilization = {total_bram_util:.2f}, DSP utilization = {total_dsp_util:.2f}")

        layers_bram_util = deepcopy(total_bram_util)
        _, min_bram_util = get_worst_case_buffering(deepcopy(graph), self.partition_composer, self.platform.mem_words_per_cycle, self.platform.word_bytes, self.platform.bram_Kbytes, self.platform.bram, self.gap_approx, wr_factor=wr_factor)
        branch_buffering_bram_util = deepcopy(min_bram_util)
        total_bram_util += min_bram_util

        return total_bram_util, total_dsp_util, layers_bram_util, branch_buffering_bram_util

    def validate_partitions(self, partitions):
        for origin_layer in self.layers:
            layer_found = False
            for part in partitions.values():
                if origin_layer in part["layers"]:
                    layer_found = True
                    break
            if not layer_found:
                raise ValueError(f"Layer {origin_layer} not found in any partition")

        for part in partitions.values():
            if not part["valid"] or len(part["layers"]) < 1 or part["weights_reloading"] == -1:
                return False

        return True

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
        partitions = self.add_key_and_shift_dict(partitions, "part_" + str(part_number + 1), part_number + 1)

        # Split the partition in two
        sub_partition_1 = layers[:num_layers // 2]
        part_1_valid = self.update_single_partition(partitions, name, sub_partition_1)
        sub_partition_2 = layers[num_layers // 2:]
        part_2_valid = self.update_single_partition(partitions, "part_" + str(part_number + 1), sub_partition_2)

        if not part_1_valid:
            partitions = self.split_partition(partitions, name)
        if not part_2_valid:
            partitions = self.split_partition(partitions, "part_" + str(part_number + 1))

        return partitions

    def remove_key_and_shift_dict(self, my_dict, key_to_remove, key_number):
        num_keys = len(my_dict.keys())

        del my_dict[key_to_remove]

        for i in range(key_number + 1, num_keys):
            old_key = "part_" + str(i)
            new_key = "part_" + str(i - 1)
            my_dict[new_key] = my_dict.pop(old_key)

        return my_dict

    def merge_partition(self, partitions, name, direction, extra_bram_allowed=0):
        part_number = int(name.split("_")[-1])

        if direction == "prev":
            merge_part_number = part_number - 1
            merge_part = f"part_{merge_part_number}"
            if partitions[merge_part].pop("graph", None) is None:
                raise ValueError(f"Cannot find graph for partition {merge_part}")
            valid_part = self.update_single_partition(partitions, merge_part, partitions[merge_part]["layers"] + partitions[name]["layers"], extra_bram_allowed=extra_bram_allowed)
            self.remove_key_and_shift_dict(partitions, name, part_number)
        elif direction == "next":
            merge_part_number = part_number + 1
            merge_part = f"part_{merge_part_number}"
            if partitions[name].pop("graph", None) is None:
                raise ValueError(f"Cannot find graph for partition {name}")
            valid_part = self.update_single_partition(partitions, name, partitions[name]["layers"] + partitions[merge_part]["layers"], extra_bram_allowed=extra_bram_allowed)
            self.remove_key_and_shift_dict(partitions, merge_part, merge_part_number)

        assert valid_part, f"Invalid partition after merging part {name} with part {merge_part}"

        return partitions

    def update_single_partition(self, partitions, name, layers, wr_factor=None, extra_bram_allowed=0):

        if wr_factor is None:
            graph_new = self.create_graph(layers)
            wr_factor = calculate_wr_factor(graph_new, self.config.initial_max_bram_util + extra_bram_allowed)
        else:
            graph_new = partitions[name]["graph"]
        if wr_factor == -1:
            partitions[name]["layers"] = layers
            partitions[name]["graph"] = graph_new
            partitions[name]["valid"] = False
            partitions[name]["weights_reloading"] = -1
            return False
        bram_util, dsp_util, layers_bram, branch_bram = self.get_partition_utilization(graph_new, wr_factor=wr_factor)
        part_validity = True if bram_util <= self.config.initial_max_bram_util + extra_bram_allowed and dsp_util <= self.config.max_dsp_util else False

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
        for name, specs in partitions.items():
            if specs["total_bram"] < self.config.merge_bram_threshold and name not in blacklisted_partitions:
                return name
        return None

    def get_partition_to_merge(self, partitions, name, extra_bram_allowed):
        part_number = int(name.split("_")[-1])
        part_bram_util = partitions[name]["total_bram"]
        part_wr_factor = partitions[name]["weights_reloading"]
        direction = None

        if part_number == 0:
            next_part_name = f"part_{part_number + 1}"
            next_part_bram_util = partitions[next_part_name]["total_bram"]
            next_part_wr_factor = partitions[next_part_name]["weights_reloading"]

            if part_bram_util + next_part_bram_util < self.config.initial_max_bram_util + extra_bram_allowed:
                direction = "next"
        elif part_number == len(partitions.keys()) - 1:
            prev_part_name = f"part_{part_number - 1}"
            prev_part_bram_util = partitions[prev_part_name]["total_bram"]
            prev_part_wr_factor = partitions[prev_part_name]["weights_reloading"]

            if part_bram_util + prev_part_bram_util < self.config.initial_max_bram_util + extra_bram_allowed:
                direction = "prev"
        else:
            prev_part_name = f"part_{part_number - 1}"
            prev_part_bram_util = partitions[prev_part_name]["total_bram"]
            prev_part_wr_factor = partitions[prev_part_name]["weights_reloading"]

            next_part_name = f"part_{part_number + 1}"
            next_part_bram_util = partitions[next_part_name]["total_bram"]
            next_part_wr_factor = partitions[next_part_name]["weights_reloading"]

            if (part_bram_util + prev_part_bram_util <= part_bram_util + next_part_bram_util) and (part_bram_util + prev_part_bram_util < self.config.initial_max_bram_util + extra_bram_allowed):
                direction = "prev"
            elif (part_bram_util + next_part_bram_util < part_bram_util + prev_part_bram_util) and (part_bram_util + next_part_bram_util < self.config.initial_max_bram_util + extra_bram_allowed):
                direction = "next"

        return direction

    # def count_layer_types(self, graph):

    #     layers_count = dict()
    #     num_layers = 0
    #     for layer in nx.topological_sort(graph):
    #         layer_type = graph.nodes[layer]["type"]
    #         if layer_type not in ["mem_in", "mem_out"]:
    #             num_layers += 1
    #         if layer_type not in layers_count:
    #             layers_count[layer_type] = 1
    #         else:
    #             layers_count[layer_type] += 1

    #     return layers_count, num_layers

    # def update_partitions(self, partitions, part_name, mode="move", direction="prev"):
    #     part_number = int(part_name.split("_")[-1])
    #     prev_part_name = f"part_{part_number - 1}"
    #     next_part_name = f"part_{part_number + 1}"

    #     match mode:
    #         case "move":
    #             num_layers_to_move = 2
    #             if direction == "prev":
    #                 print(f"Moving layers from {part_name} to previous part {prev_part_name}")
    #                 prev_part_layers = partitions[prev_part_name]["layers"]
    #                 curr_part_layers = partitions[part_name]["layers"]
    #                 self.update_single_partition(partitions, prev_part_name, prev_part_layers + curr_part_layers[:num_layers_to_move])
    #                 self.update_single_partition(partitions, part_name, curr_part_layers[num_layers_to_move:])
    #             elif direction == "next":
    #                 print(f"Moving layers from {part_name} to next part {next_part_name}")
    #                 curr_part_layers = partitions[part_name]["layers"]
    #                 next_part_layers = partitions[next_part_name]["layers"]
    #                 self.update_single_partition(partitions, part_name, curr_part_layers[:-num_layers_to_move])
    #                 self.update_single_partition(partitions, next_part_name, curr_part_layers[-num_layers_to_move:] + next_part_layers)
    #             else:
    #                 raise ValueError(f"Invalid direction {direction} for moving layers between partitions.")
    #         case "merge":
    #             if direction == "prev":
    #                 print(f"Merging {part_name} with previous part {prev_part_name}")
    #                 self.update_single_partition(partitions, prev_part_name, partitions[prev_part_name]["layers"] + partitions[part_name]["layers"])
    #                 self.remove_key_and_shift_dict(partitions, part_name, part_number)
    #             elif direction == "next":
    #                 print(f"Merging {part_name} with next part {next_part_name}")
    #                 self.update_single_partition(partitions, part_name, partitions[part_name]["layers"] + partitions[next_part_name]["layers"])
    #                 self.remove_key_and_shift_dict(partitions, next_part_name, part_number + 1)
    #             else:
    #                 raise ValueError(f"Invalid direction {direction} for merging partitions.")
    #         case "split":
    #             pass
    #         case _:
    #             raise ValueError(f"Invalid mode {mode} for updating partitions.")

    #     return partitions

    def refine_partitions(self, partitions):
        for part, specs in partitions.items():
            assert len(specs["layers"]) > 0, f"Partition {part} has no layers assigned to it."
            if not specs["valid"]:
                if "total_bram" in specs:
                    bram_util = specs["total_bram"]
                else:
                    bram_util = "unknown"
                _logger.info(f"Refining partition {part} with {len(specs['layers'])} layers and BRAM utilization {bram_util}")

                wr_factor = calculate_wr_factor(specs["graph"], self.config.initial_max_bram_util)
                if wr_factor >= 1:
                    part_valid = self.update_single_partition(partitions, part, specs["layers"], wr_factor=wr_factor)
                    _logger.info(f"WR factor for partition {part} is {wr_factor}. Validity is {part_valid}.")
                else:
                    _logger.warning(f"Partition {part} cannot fit in the FPGA even after WR. Splitting it into two partitions.")
                    return self.split_partition(partitions, part)

        return partitions

    def parse(self):
        network_partitions = self.get_partitions()

        while not self.validate_partitions(network_partitions):
            network_partitions = self.refine_partitions(network_partitions)

        # for part, specs in network_partitions.items():
        #     print(f"Partition {part} has {len(specs['layers'])} layers and BRAM utilization {specs['total_bram']:.2f}, wr factor of {specs['weights_reloading']}")

        blacklisted_parts = []
        merge_extra_bram_allowed = 10
        cm_part = self.get_candidate_merge_partition(network_partitions, blacklisted_parts)
        while cm_part is not None:
            direction = self.get_partition_to_merge(network_partitions, cm_part, merge_extra_bram_allowed)
            _logger.error(f"Merging {cm_part} with {len(network_partitions[cm_part]['layers'])} number of layers into {direction} partition")
            if direction is not None:
                network_partitions = self.merge_partition(network_partitions, cm_part, direction, extra_bram_allowed=merge_extra_bram_allowed)
                blacklisted_parts.clear()
                # for part, specs in network_partitions.items():
                #     print(f"(MERGE) Partition {part} has {len(specs['layers'])} layers and BRAM utilization {specs['total_bram']:.2f}, wr factor of {specs['weights_reloading']}")
            else:
                blacklisted_parts.append(cm_part)
            cm_part = self.get_candidate_merge_partition(network_partitions, blacklisted_parts)

        assert self.validate_partitions(network_partitions), "Partitions are not valid after merging."
        self.visualize_partitions(network_partitions)

        for part, specs in network_partitions.items():
            print(f"Partition {part} has {len(specs['layers'])} layers and BRAM utilization {specs['total_bram']:.2f}, wr factor of {specs['weights_reloading']}")
            for node in specs["graph"].nodes:
                hw = specs["graph"].nodes[node]["hw"]
                print(f"Layer {node} has input shape {hw.input_shape}, output shape {hw.output_shape}")

        # TODO: SOS! During the WR calculation the shapes of the layers inside the partition are being changed. This has to be taken into consideration when merging spliting or moving layers between partitions where the WR factor is greater than 1. WR factor is probably INCORRECT for the partitions that have been split or merged.
        # TODO: Instead of generating completely new partitions we can have a new transform that alters a bit the existing partitions by adding or removing layers from previous or next partitions.
        # TODO: I dont like the thing that layers partitions and network are not being connected somehow. It would be nice to have a way to connect them and build the network from the partitions and the partitions from the layers.

    def create_graph(self, partition: list) -> nx.DiGraph:
        graph = nx.DiGraph()
        _logger.info("*" * 40)
        for layer in partition:
            _logger.info("Adding {} layer to graph...".format(layer))
            if self.layers[layer]["operation"] == "GlobalAveragePool":
                hw_layer = GAP3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                    self.platform
                )
                layer_type = self.layers[layer]["operation"]
            elif self.layers[layer]["operation"] == "Conv":
                hw_layer = Convolutional3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                    self.platform
                )
                layer_type = self.layers[layer]["operation"]
            elif self.layers[layer]["operation"] == "MaxPool" or self.layers[layer]["operation"] == "AveragePool":
                hw_layer = Pooling3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                    self.platform
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
                    self.platform
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
                    self.platform
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
                    self.platform
                )
            elif self.layers[layer]["operation"] == "BatchNormalization":
                layer_type = self.layers[layer]["operation"]
                hw_layer = BatchNorm3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                    self.platform
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

            graph.add_node(layer, type=layer_type, hw=hw_layer, hw_type=hw_type, layer_mode=layer_mode)
        _logger.info("*" * 40)

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