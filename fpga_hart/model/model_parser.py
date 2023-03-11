import os
import random
from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import wandb
from fpga_hart import _logger
from fpga_hart.layers.activation import ActivationLayer
from fpga_hart.layers.batchnorm_3d import BatchNorm3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise import ElementWiseLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap import GAPLayer
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.network_representation.model_descriptor import \
    ModelLayerDescriptor
from fpga_hart.utils.graph_manipulation import visualize_graph
from fpga_hart.utils.utils import get_conv_type, get_pool_type, num_sort

sns.set(rc={"figure.figsize": (15, 8)})
sns.set_style("whitegrid")

def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results


@dataclass
class ModelParser(ModelLayerDescriptor):
    batch_size: int
    num_reconfig_points: int
    allowed_reconfig_layers: list
    min_partition_layers: int
    max_partition_layers: int
    config: wandb.Config
    enable_wandb: bool

    def __post_init__(self) -> None:
        ModelLayerDescriptor.__post_init__(self)  # Initialize the parent class
        # _logger.setLevel(level=logging.DEBUG)

    def get_reconfig_points(self):
        available_reconfig_points = [layer for layer in self.layers if self.layers[layer]['operation'] in self.allowed_reconfig_layers]

        # Remove the input and output layers from the available reconfig points
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

        partitions = []
        for rp in reconfig_points:
            current_partition = []
            for layer in model_layers:
                if layer == rp:
                    break
                current_partition.append(layer)
            partitions.append(current_partition)
            model_layers = model_layers[model_layers.index(rp):]
        partitions.append(model_layers)

        return partitions

    def parse(self):
        initial_partitions = self.get_partitions()

        for i, partition in enumerate(initial_partitions):
            _logger.warning("Partition {} with {} number of layers:".format(i, len(partition)))
            graph_partition = self.create_graph(partition)
            name = "partition_{}".format(i)

            if not os.path.exists(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/model_graphs/"):
                os.makedirs(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/model_graphs/")
            visualize_graph(
                graph_partition,
                os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/model_graphs/" + name,
                self.enable_wandb,
                name,
            )
        # TODO: Instead of generating completely new partitions we can have a new transform that alters a bit the existing partitions by adding or removing layers from previous or next partitions.
        # TODO: We should always have a check that validates the partition and checks whether the partition weights are within the BRAM limits. Otherwise, we are going to need the wieghts reloaded from the DRAM.
        # TODO: When spliting we should also check and add the extra inputs that may be needed because of either spliting on branches or because of the ElementWise layers.

    def create_graph(self, partition: list) -> nx.DiGraph:
        graph = nx.DiGraph()
        _logger.info("*" * 40)
        for layer in partition:
            _logger.info("Adding {} layer to graph...".format(layer))
            if self.layers[layer]["operation"] == "GlobalAveragePool":
                hw_layer = GAPLayer(
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
                hw_layer = ActivationLayer(
                    self.config.max_dsp_util,
                    self.config.max_bram_util,
                    self.layers[layer],
                )
                layer_type = "Activation"
            elif (
                self.layers[layer]["operation"] == "Mul"
                or self.layers[layer]["operation"] == "Add"
            ):
                hw_layer = ElementWiseLayer(
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
            graph.add_node(layer, type=layer_type, hw=hw_layer, hw_type=hw_type)
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