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
from fpga_hart.network_representation.model_descriptor import \
    ModelLayerDescriptor
from fpga_hart.utils.graph_manipulation import visualize_graph
from fpga_hart.utils.utils import num_sort

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

        reconfig_points = random.sample(available_reconfig_points, self.num_reconfig_points)
        reconfig_points.sort(key=num_sort)

        return reconfig_points

    def get_partitions(self):
        reconfig_points = self.get_reconfig_points()
        _logger.warning(f"Reconfig points: {reconfig_points}")
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

    def parse(self):
        self.get_partitions()
