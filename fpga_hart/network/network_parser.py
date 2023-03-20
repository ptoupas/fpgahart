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
    get_minimum_resource_utilization, get_worst_case_buffering)
from fpga_hart.parser.model_descriptor import ModelLayerDescriptor
from fpga_hart.partitions.partition_compose import PartitionComposer
from fpga_hart.platform.platform import Platform
from fpga_hart.utils.graph_manipulation import (add_off_chip_connections,
                                                visualize_graph)
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
    config: wandb.Config
    enable_wandb: bool

    def __post_init__(self) -> None:
        ModelLayerDescriptor.__post_init__(self)  # Initialize the parent class
        # _logger.setLevel(level=logging.DEBUG)

        self.platform = Platform()

        self.partition_composer = PartitionComposer(
            max_DSP_util=self.config.max_dsp_util, max_BRAM_util=self.config.max_bram_util
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

    def get_partitions(self, plot_graph=False):
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
            for n in nx.topological_sort(part_graph):
                print(n, part_graph.nodes[n])
            bram_util, dsp_util, layers_bram, branch_bram = self.get_partition_utilization(part_graph)
            if plot_graph:
                if not os.path.exists(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/"):
                        os.makedirs(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/")
                visualize_graph(
                    part_graph,
                    os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/" + partition_name,
                    self.enable_wandb,
                    partition_name,
                )
            partition_specs["total_bram"] = bram_util
            partition_specs["total_dsp"] = dsp_util
            partition_specs["layers_bram"] = layers_bram
            partition_specs["branch_bram"] = branch_bram

            partitions[partition_name] = partition_specs
            model_layers = model_layers[model_layers.index(rp):]

        partition_specs = dict()
        partition_name = f"part_{i}"
        partition_specs["layers"] = model_layers

        part_graph = self.create_graph(model_layers)
        bram_util, dsp_util, layers_bram, branch_bram = self.get_partition_utilization(part_graph)
        if plot_graph:
            if not os.path.exists(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/"):
                    os.makedirs(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/")
            visualize_graph(
                part_graph,
                os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/" + partition_name,
                self.enable_wandb,
                partition_name,
            )
        partition_specs["total_bram"] = bram_util
        partition_specs["total_dsp"] = dsp_util
        partition_specs["layers_bram"] = layers_bram
        partition_specs["branch_bram"] = branch_bram

        partitions[partition_name] = partition_specs

        return partitions

    def get_partition_utilization(self, graph):
        total_bram_util = 0
        total_dsp_util = 0
        for layer in nx.topological_sort(graph):
            bram_util, dsp_util, pipeline_depth, _ = get_minimum_resource_utilization(graph.nodes[layer]["hw"], gap_approx=self.gap_approx)
            _logger.debug(f"Layer {layer}: BRAM utilization = {bram_util:.2f}, DSP utilization = {dsp_util:.2f}, Pipeline depth = {pipeline_depth}")
            total_bram_util += bram_util
            total_dsp_util += dsp_util
            if total_bram_util > self.config.max_bram_util or total_dsp_util > self.config.max_dsp_util:
                _logger.warning(f"Partition BRAM utilization = {total_bram_util:.2f}, DSP utilization = {total_dsp_util:.2f}")

        layers_bram_util = deepcopy(total_bram_util)
        # TODO: Do we want to use a copy of a graph here or not? In case we remove that we will alter the original graph by adding the off-chip memory connections
        _, min_bram_util = get_worst_case_buffering(graph, self.partition_composer, self.platform.mem_words_per_cycle, self.platform.word_bytes, self.platform.bram_Kbytes, self.platform.bram, self.gap_approx)
        branch_buffering_bram_util = deepcopy(min_bram_util)
        total_bram_util += min_bram_util

        return total_bram_util, total_dsp_util, layers_bram_util, branch_buffering_bram_util

    def validate_partitions(self, partitions):
        for part_name, part in partitions.items():

            # part_graph = self.create_graph(part["layers"])

            # bram_util, dsp_util, layers_bram, branch_bram = self.get_partition_utilization(part_graph)
            # part["total_bram"] = bram_util
            # part["total_dsp"] = dsp_util
            # part["layers_bram"] = layers_bram
            # part["branch_bram"] = branch_bram
            if part["total_bram"] > self.config.max_bram_util or part["total_dsp"] > self.config.max_dsp_util:
                return False

        return True

    def rearrange_partitions(self, partitions):
        for part, specs in partitions.items():
            print(f"{part}: {specs}")


    def parse(self):
        initial_partitions = self.get_partitions(plot_graph=True)
        if not self.validate_partitions(initial_partitions):
            _logger.error("Invalid partitions. Exiting...")
            self.rearrange_partitions(initial_partitions)
            exit()

        # TODO: Instead of generating completely new partitions we can have a new transform that alters a bit the existing partitions by adding or removing layers from previous or next partitions.
        # TODO: We should always have a check that validates the partition and checks whether the partition weights are within the BRAM limits. Otherwise, we are going to need the wieghts reloaded from the DRAM.
        # TODO: When spliting we should also check and add the extra inputs that may be needed because of either spliting on branches or because of the ElementWise layers.
        # TODO: Instead of setting a fixed number of partitions we can start with a valid model partitioning and then we can have a transform that merges or splits partitions based on the available resources and performance.
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
                )
                layer_type = self.layers[layer]["operation"]
            elif self.layers[layer]["operation"] == "Conv":
                hw_layer = Convolutional3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                )
                layer_type = self.layers[layer]["operation"]
            elif self.layers[layer]["operation"] == "MaxPool" or self.layers[layer]["operation"] == "AveragePool":
                hw_layer = Pooling3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
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
                )
            elif self.layers[layer]["operation"] == "BatchNormalization":
                layer_type = self.layers[layer]["operation"]
                hw_layer = BatchNorm3DLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
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

        if not os.path.exists(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/"):
            os.makedirs(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/")
        visualize_graph(
                graph,
                os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/throughput/model_graphs/" + "test",
                self.enable_wandb,
                "test",
            )

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