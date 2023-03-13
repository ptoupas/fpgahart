import os
from collections import Counter, deque

import networkx as nx
from matplotlib import pyplot as plt

from fpga_hart import _logger
from fpga_hart.optimizer.simulated_annealing import SimulatedAnnealing
from fpga_hart.utils import utils
from fpga_hart.utils.graph_manipulation import visualize_graph

def create_partitions(self, layers: dict) -> list:
    final_layers = []

    if self.model_name == "x3d_m":
        if not self.se_block:
            layer_type_1 = [
                "Relu",
                "Conv",
                "Relu",
                "Conv",
                "GlobalAveragePool",
                "Conv",
                "Relu",
                "Conv",
                "Sigmoid",
                "Mul",
                "Swish",
                "Conv",
                "Conv",
                "Add",
            ]
            layer_type_2 = [
                "Relu",
                "Conv",
                "Relu",
                "Conv",
                "GlobalAveragePool",
                "Conv",
                "Relu",
                "Conv",
                "Sigmoid",
                "Mul",
                "Swish",
                "Conv",
                "Add",
            ]
            layer_type_3 = ["Relu", "Conv", "Relu", "Conv", "Swish", "Conv", "Add"]
            layer_type_4 = [
                "Conv",
                "Conv",
                "Relu",
                "Conv",
                "Relu",
                "Conv",
                "GlobalAveragePool",
                "Conv",
                "Relu",
                "Conv",
                "Sigmoid",
                "Mul",
                "Swish",
                "Conv",
                "Conv",
                "Add",
            ]
            layer_type_5 = [
                "Relu",
                "Conv",
                "Relu",
                "GlobalAveragePool",
                "Gemm",
                "Relu",
                "Gemm",
            ]
            layer_queue = deque(maxlen=16)
            layer_queue_operations = deque(maxlen=16)
            for k in layers.keys():
                layer_queue_operations.append(layers[k]["operation"])
                layer_queue.append(k)
                if list(layer_queue_operations) == layer_type_4:
                    final_layers.append(list(layer_queue))
                elif list(layer_queue_operations)[2:] == layer_type_1:
                    final_layers.append(list(layer_queue)[2:])
                elif list(layer_queue_operations)[3:] == layer_type_2:
                    final_layers.append(list(layer_queue)[3:])
                elif list(layer_queue_operations)[9:] == layer_type_3:
                    final_layers.append(list(layer_queue)[9:])
                elif list(layer_queue_operations)[9:] == layer_type_5:
                    final_layers.append(list(layer_queue)[9:])
        else:
            layer_type_1 = [
                "Relu",
                "Conv",
                "Relu",
                "Conv",
                "SqueezeExcitation",
                "Swish",
                "Conv",
                "Conv",
                "Add",
            ]
            layer_type_2 = [
                "Relu",
                "Conv",
                "Relu",
                "Conv",
                "SqueezeExcitation",
                "Swish",
                "Conv",
                "Add",
            ]
            layer_type_3 = ["Relu", "Conv", "Relu", "Conv", "Swish", "Conv", "Add"]
            layer_type_4 = ["Conv", "Conv"]
            layer_type_5 = [
                "Relu",
                "Conv",
                "Relu",
                "GlobalAveragePool",
                "Gemm",
                "Relu",
                "Gemm",
            ]
            layer_queue = deque(maxlen=9)
            layer_queue_operations = deque(maxlen=9)
            for k in layers.keys():
                layer_queue_operations.append(layers[k]["operation"])
                layer_queue.append(k)
                if list(layer_queue_operations) == layer_type_1:
                    final_layers.append(list(layer_queue))
                if list(layer_queue_operations)[:-1] == layer_type_2:
                    final_layers.append(list(layer_queue)[:-1])
                if list(layer_queue_operations)[:-2] == layer_type_3:
                    final_layers.append(list(layer_queue)[:-2])
                if (
                    list(layer_queue_operations)[:-7] == layer_type_4
                    and "Conv_0" in list(layer_queue)[:-7]
                ):
                    final_layers.append(list(layer_queue)[:-7])
                if list(layer_queue_operations)[2:] == layer_type_5:
                    final_layers.append(list(layer_queue)[2:])
        return final_layers
    elif self.model_name == "i3d":
        layer_type_1 = ["Conv", "Relu", "Conv", "Relu", "Conv", "Conv", "Add"]
        layer_type_2 = ["Relu", "Conv", "Relu", "Conv", "Relu", "Conv", "Add"]
        layer_queue = deque(maxlen=7)
        layer_queue_operations = deque(maxlen=7)
        for k in layers.keys():
            layer_queue_operations.append(layers[k]["operation"])
            layer_queue.append(k)
            if list(layer_queue_operations) == layer_type_1:
                final_layers.append(list(layer_queue))
            if list(layer_queue_operations) == layer_type_2:
                final_layers.append(list(layer_queue))
        return final_layers
    elif self.model_name == "c3d":
        layers_list = list(layers.keys())
        final_layers.append(layers_list)
        return final_layers
    elif self.model_name == "r2plus1d":
        layer_type_1 = ["Conv", "Relu", "Conv", "Relu", "MaxPool", "Conv", "Relu", "Conv", "Relu", "Conv", "Relu", "Conv", "Add"]
        layer_type_2 = ["Relu", "Conv", "Relu", "Conv", "Relu", "Conv", "Relu", "Conv", "Add"]
        layer_type_3 = ["Relu", "Conv", "Relu", "Conv", "Relu", "Conv", "Relu", "Conv", "Conv", "Relu", "Conv", "Add"]
        layer_type_4 = ["Relu", "GlobalAveragePool", "Gemm"]
        layer_queue = deque(maxlen=13)
        layer_queue_operations = deque(maxlen=13)
        for k in layers.keys():
            layer_queue_operations.append(layers[k]["operation"])
            layer_queue.append(k)
            if list(layer_queue_operations) == layer_type_1:
                final_layers.append(list(layer_queue))
            elif list(layer_queue_operations)[4:] == layer_type_2:
                final_layers.append(list(layer_queue)[4:])
            elif list(layer_queue_operations)[1:] == layer_type_3:
                final_layers.append(list(layer_queue)[1:])
            elif list(layer_queue_operations)[10:] == layer_type_4:
                final_layers.append(list(layer_queue)[10:])
        return final_layers
    elif self.model_name == "slowonly":
        layer_type_1 = ["Conv", "Relu", "MaxPool", "Conv", "Relu", "Conv", "Relu", "Conv", "Conv", "Add"]
        layer_type_2 = ["Relu", "Conv", "Relu", "Conv", "Relu", "Conv", "Add"]
        layer_type_3 = ["Relu", "Conv", "Relu", "Conv", "Relu", "Conv", "Conv", "Add"]
        layer_type_4 = ["Relu", "GlobalAveragePool", "Gemm"]
        layer_queue = deque(maxlen=10)
        layer_queue_operations = deque(maxlen=10)
        for k in layers.keys():
            layer_queue_operations.append(layers[k]["operation"])
            layer_queue.append(k)
            if list(layer_queue_operations) == layer_type_1:
                final_layers.append(list(layer_queue))
            elif list(layer_queue_operations)[3:] == layer_type_2:
                final_layers.append(list(layer_queue)[3:])
            elif list(layer_queue_operations)[2:] == layer_type_3:
                final_layers.append(list(layer_queue)[2:])
            elif list(layer_queue_operations)[7:] == layer_type_4:
                final_layers.append(list(layer_queue)[7:])
        return final_layers

def update_hw_pe(self, graph: nx.DiGraph, groupping: int = 1) -> None:
    nodes = utils.get_nodes_sorted(graph)
    for i in range(0, len(nodes), groupping):
        node_type = ""
        for j in range(groupping):
            if node_type != "":
                node_type += f"_{graph.nodes[nodes[i+j]]['hw_type']}"
            else:
                node_type += graph.nodes[nodes[i + j]]["hw_type"]
        if node_type not in self.hw_pe:
            self.hw_pe.append(node_type)

def schedule_ops(
    self, graph: nx.DiGraph, groupping: int = 1, plot_pe: bool = False
) -> list:

    static_schedule = []
    nodes = utils.get_nodes_sorted(graph)
    for i in range(0, len(nodes), groupping):
        hw_op = ""
        for j in range(groupping):
            if hw_op != "":
                hw_op += f"_{graph.nodes[nodes[i+j]]['hw_type']}"
            else:
                hw_op += graph.nodes[nodes[i + j]]["hw_type"]
        if hw_op in self.hw_pe:
            static_schedule.append(hw_op)
        else:
            _logger.critical(msg=f"Hardware type {hw_op} not found in PE list")
            exit()

    if plot_pe:
        plt.cla()
        comb_dict = dict(Counter(static_schedule))
        keys = list(comb_dict.keys())
        values = list(comb_dict.values())

        plt.barh(keys, values)
        for index, value in enumerate(values):
            plt.text(value, index, str(value))
        plt.tight_layout()
        plt.savefig(
            os.getcwd()
            + f"/fpga_modeling_reports/layer_grouppings/hw_pe_group_{groupping}.png"
        )

    return static_schedule

def update_layer_types(self, graph: nx.DiGraph, plot_types: bool = False) -> None:
    layer_channels = []
    for node in utils.get_nodes_sorted(graph):
        if graph.nodes[node]["hw_type"] == "Gemm":
            break
        layer_channels.append(graph.nodes[node]["hw"].input_shape[1])

    comb_dict = dict(Counter(layer_channels))
    channels_bins = utils.get_channels_bins(
        layer_channels, plot_lbow=False, plot_hist=False
    )

    layer_types = []
    for node in utils.get_nodes_sorted(graph):
        if graph.nodes[node]["hw_type"] == "Gemm":
            break
        channels_curr = graph.nodes[node]["hw"].input_shape[1]
        for c_bin in channels_bins:
            if channels_curr in c_bin:
                postfix = f"CG{int(c_bin.left)}-{int(c_bin.right)}"
                break
        graph.nodes[node]["hw_type"] += postfix
        layer_types.append(graph.nodes[node]["hw_type"])
        _logger.warning(f"{node} -> {graph.nodes[node]['hw_type']}")

    comb_dict = dict(Counter(layer_types))
    keys = list(comb_dict.keys())
    values = list(comb_dict.values())

    if plot_types:
        plt.barh(keys, values)
        for index, value in enumerate(values):
            plt.text(value, index, str(value))
        plt.tight_layout()
        plt.show()

def latency_driven_design(
    self,
    plot_summaries: bool = False,
) -> None:
    """
    Try to find the best configurations to be used for a hardware
    processing element to support all the convolutional layers in the graph.
    """

    model_layers = [layer for layer, config in self.layers.items()]

    graph = self.create_graph(model_layers)
    if not os.path.exists(
        os.getcwd() + "/fpga_modeling_reports/graphs/" + self.model_name + "/"
    ):
        os.makedirs(
            os.getcwd() + "/fpga_modeling_reports/graphs/" + self.model_name + "/"
        )
    visualize_graph(
        graph,
        os.getcwd()
        + "/fpga_modeling_reports/graphs/"
        + self.model_name
        + "/latency_driven_graph",
        True,
        "latency_driven_graph"
    )

    optimizer = SimulatedAnnealing(
        graph,
        config=self.config,
        cnn_model_name=self.model_name,
        enable_wandb=self.enable_wandb,
    )
    optimizer.run_optimizer_latency(alignedfactors=self.config.alignedfactors)
