import os
from collections import Counter, deque
from dataclasses import dataclass

import networkx as nx
import wandb
from fpga_hart import _logger
from fpga_hart.layers.layer_design import layer_design_points
from fpga_hart.network_representation.model_descriptor import ModelLayerDescriptor
from fpga_hart.optimizer.simulated_annealing import SimulatedAnnealing
from fpga_hart.utils import utils
from matplotlib import pyplot as plt


@dataclass
class PartitionDescriptor(ModelLayerDescriptor):
    def __post_init__(self) -> None:
        ModelLayerDescriptor.__post_init__(self)  # Initialize the parent class
        # _logger.setLevel(level=logging.DEBUG)
        self.partitions = self.create_partitions(self.layers)
        self.hw_pe = list()

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
                # layer_type_4 = ["Conv", "Conv"]
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
        run_name: str,
        plot_summaries: bool = False,
        alignedfactors: bool = False,
        wandb_config: wandb.Config = None,
    ) -> None:
        """
        Try to find the best configurations to be used for a hardware
        processing element to support all the convolutional layers in the graph.
        """

        # Here we get all the layers in the graph but the first two convolutional layers
        sub_layers = [
            layer
            for layer, config in self.layers.items()
            if config["operation"]
            in [
                "Conv",
                "Relu",
                "Sigmoid",
                "Swish",
                "GlobalAveragePool",
                "Add",
                "Mul",
                "Gemm",
            ]
        ]

        graph = self.create_graph(sub_layers)
        if not os.path.exists(
            os.getcwd() + "/fpga_modeling_reports/graphs/" + self.model_name + "/"
        ):
            os.makedirs(
                os.getcwd() + "/fpga_modeling_reports/graphs/" + self.model_name + "/"
            )
        self.visualize_graph(
            graph,
            os.getcwd()
            + "/fpga_modeling_reports/graphs/"
            + self.model_name
            + "/latency_driven_graph",
            run_id=None,
        )
        optimizer = SimulatedAnnealing(
            graph,
            t_min=5e-5,
            t_max=10,
            iterationPerTemp=10,
            cooling_rate=0.98,
            ml_flow_id=None,
            wandb_config=wandb_config,
            cnn_model_name=self.model_name,
        )
        if wandb_config is not None:
            wandb_config.update({"aligned_folding_factors": alignedfactors})
        optimizer.run_optimizer_latency(alignedfactors=alignedfactors)
        return

        conv_types = []
        for layer in sub_layers:
            conv_types.append(
                utils.get_conv_type(
                    self.layers[layer],
                    discriminate_kernel_size=True,
                    discriminate_stide=True,
                    discriminate_channels_filters=True,
                )
            )
        comb_dict = dict(Counter(conv_types))
        comb_dict = dict(sorted(comb_dict.items(), key=lambda item: item[0]))
        _logger.warning(comb_dict)
        if plot_summaries:
            keys = list(comb_dict.keys())
            values = list(comb_dict.values())
            plt.barh(keys, values)
            for index, value in enumerate(values):
                plt.text(value, index, str(value))
            plt.tight_layout()
            plt.show()

        cin_cout_list = []
        cin_list = []
        cout_list = []
        din_list = []
        win_list = []
        wout_list = []
        for layer, config in self.layers.items():
            if config["operation"] == "Conv":
                layer_type = utils.get_conv_type(
                    config,
                    discriminate_kernel_size=True,
                    discriminate_stide=True,
                    discriminate_channels_filters=True,
                )
                # if "Dw" in layer_type:
                _logger.warning(
                    f"Searching design points for layer {layer} with type {layer_type}"
                )
                cin, cout = layer_design_points(
                    name=layer,
                    description=config,
                    max_DSP_util=wandb_config.max_dsp_util,
                    max_BRAM_util=wandb_config.max_bram_util,
                    model_file="random_file.csv",
                    singlethreaded=True,
                )
                shape_in = config["shape_in"][0]
                _, cin, din, hin, win = shape_in
                shape_out = config["shape_out"]
                _, cout, dout, hout, wout = shape_out
                cin_cout_list.append(f"{int(cin)}_{int(cout)}")
                cin_list.append(cin)
                cout_list.append(cout)
                din_list.append(din)
                win_list.append(win)
                wout_list.append(wout)

        if plot_summaries:
            plt.cla()
            fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(20, 10))
            # fig.suptitle("Bigger 1 row x 2 columns axes with no data")
            axes[0][0].set_title("Channes In-Spatial Dims correlation")
            axes[0][0].scatter(cin_list, win_list)
            axes[0][0].set_xlabel("Channels In")
            axes[0][0].set_ylabel("Spatial Dims")

            axes[0][1].set_title("Channes In-Temporal Dim correlation")
            axes[0][1].scatter(cin_list, din_list)
            axes[0][1].set_xlabel("Channels In")
            axes[0][1].set_ylabel("Temporal Dim")

            axes[1][0].set_title("Channes In-Channels Out correlation")
            axes[1][0].scatter(cin_list, cout_list)
            axes[1][0].set_xlabel("Channels In")
            axes[1][0].set_ylabel("Channels Out")

            axes[1][1].set_title("Channes Out-Spatial Dims(out) correlation")
            axes[1][1].scatter(cout_list, wout_list)
            axes[1][1].set_xlabel("Channels Out")
            axes[1][1].set_ylabel("Spatial Dims(out)")

            plt.tight_layout()
            plt.show()

            comb_dict = dict(Counter(cin_cout_list))
            comb_dict = dict(sorted(comb_dict.items(), key=lambda item: item[0]))
            _logger.warning(comb_dict)
            keys = list(comb_dict.keys())
            values = list(comb_dict.values())
            plt.cla()
            plt.bar(keys, values)
            plt.tight_layout()
            plt.show()
        """
            Based on the results from the for loop above we can see that:
                1. The most used coarse in factor across all the DW convolutional layers is 18 (same as coarse out). -> DSP=28.12%, BRAM=92.31%
                2. The most used coarse in / coarse out factor combinations across all the PW convolutional layers (with resource contraint at 62 %, abstracting the DW block implementation resources) is 4_216 -> DSP=50.00%, BRAM=0.0% (the one we will use for now is 16_64 -> DSP=59.26%, BRAM=0.0%).
            We have assumed the fine factor to be the maximum possible for the sake of simplicity.
            Based on these observations we can procceed and create a harware building block supporting up to 18 parallel streams in its input and 48 parallel streams in its output, and try to map and create a schedule for all the convolutional layers of the network.
        """

    def find_common_layers(self, groupping: int = 1) -> None:
        """
        Finds combinations of layers in the model that can be mapped together into a single hardware IP.
        Currently, the following assumptions are made:
            1. The cannot be combinations containing either layers that split the graph into branches or layers that
                merge the branches back.

        Args:
            groupping (int, optional): Number of layers to group together as combined HW building blocks.
        """

        def validate_combination(combs: tuple) -> bool:
            prev_layer = None
            for val in combs:
                if prev_layer is None:
                    prev_layer = val
                    continue
                if val not in list(graph.successors((prev_layer))):
                    return False
                prev_layer = val
            return True

        layers = []
        for layer, config in self.layers.items():
            layers.append(layer)

        plt.figure(figsize=(20, 10))
        graph = self.create_graph(layers)
        self.update_layer_types(graph=graph, plot_types=False)

        for i in range(1, 4):
            self.update_hw_pe(graph=graph, groupping=i)
            self.schedule_ops(graph=graph, groupping=i, plot_pe=True)

        # self.visualize_graph(
        #     graph,
        #     os.getcwd() + "/fpga_modeling_reports/layer_grouppings/graph_complete",
        # )

        # blacklist_combinations = []
        # group_type_count = 0
        # count = 0
        # while True:
        #     valid_combinations = list(
        #         filter(
        #             validate_combination,
        #             itertools.combinations(
        #                 utils.get_nodes_sorted(graph), groupping),
        #         ),
        #     )
        #     # TODO: WHEN ADDING OR DELETING NODES THE ORDER OF THE NODES OF THE GRAPH IS CHANGED

        #     valid_combinations_types = []
        #     for c in valid_combinations:
        #         type_conb = tuple()
        #         split_nodes_count = 0
        #         merge_nodes_count = 0
        #         for n in c:
        #             if n not in graph.nodes():
        #                 continue
        #             if graph.out_degree(n) > 1:
        #                 split_nodes_count += 1
        #             if graph.in_degree(n) > 1:
        #                 merge_nodes_count += 1
        #             try:
        #                 type_conb += (graph.nodes[n]["hw_type"],)
        #             except Exception as e:
        #                 _logger.critical(f"{e}\nnode: {n} -> {graph.nodes[n]}")
        #                 exit()
        #         if (
        #             split_nodes_count <= 1
        #             and merge_nodes_count <= 1
        #             and type_conb not in blacklist_combinations
        #         ):
        #             valid_combinations_types.append(type_conb)

        #     comb_dict = dict(Counter(valid_combinations_types))
        #     comb_dict = {
        #         k: v
        #         for k, v in sorted(
        #             comb_dict.items(), key=lambda item: item[1], reverse=True
        #         )
        #     }
        #     if comb_dict[list(comb_dict.keys())[0]] <= 1:
        #         break

        #     frequent_comb = list(comb_dict.keys())[0]

        #     for c in valid_combinations:
        #         type_conb = tuple()
        #         for n in c:
        #             if n not in graph.nodes():
        #                 continue
        #             try:
        #                 type_conb += (graph.nodes[n]["hw_type"],)
        #             except Exception as e:
        #                 _logger.critical(f"{e}\nnode: {n} -> {graph.nodes[n]}")
        #                 exit()
        #         if type_conb in blacklist_combinations:
        #             continue

        #         drop_nodes = []
        #         drop_edges = []
        #         add_nodes = []
        #         add_in_edges = []
        #         add_out_edges = []
        #         if type_conb == frequent_comb:
        #             underlying_layers_hw = dict()
        #             underlying_layers = []
        #             underlying_layers_type = []
        #             for n in c:
        #                 drop_nodes.append(n)
        #                 edges_in = list(graph.in_edges(n))
        #                 drop_edges += edges_in
        #                 edges_out = list(graph.out_edges(n))
        #                 drop_edges += edges_out
        #                 if "group" in n:
        #                     underlying_layers += graph.nodes[n]["underlying_layers"]
        #                     underlying_layers_type += graph.nodes[n][
        #                         "underlying_layers_type"
        #                     ]
        #                     underlying_layers_hw.update(graph.nodes[n]["hw"])
        #                 else:
        #                     underlying_layers.append(n)
        #                     underlying_layers_type.append(
        #                         graph.nodes[n]["hw_type"])
        #                     underlying_layers_hw[n] = graph.nodes[n]["hw"]
        #             drop_edges = list(set(drop_edges))
        #             add_nodes.append(
        #                 (
        #                     f"grouped_layers_{count}_type_{group_type_count}",
        #                     dict(
        #                         hw_type=f"group_type_{group_type_count}",
        #                         underlying_layers=underlying_layers,
        #                         underlying_layers_type=underlying_layers_type,
        #                         hw=underlying_layers_hw,
        #                     ),
        #                 )
        #             )

        #             for i, n in enumerate(c):
        #                 if len(list(graph.predecessors(n))) > 0 and (i == 0):
        #                     for ie in graph.predecessors(n):
        #                         in_edge = (
        #                             ie,
        #                             f"grouped_layers_{count}_type_{group_type_count}",
        #                         )
        #                         add_in_edges += [in_edge]
        #                 if len(list(graph.successors(n))) > 0 and (i == len(c) - 1):
        #                     for oe in graph.successors(n):
        #                         out_edge = (
        #                             f"grouped_layers_{count}_type_{group_type_count}",
        #                             oe,
        #                         )
        #                         add_out_edges += [out_edge]
        #                 if len(list(graph.successors(n))) > 0 and (
        #                     graph.in_degree(n) > 1
        #                 ):
        #                     for ie in graph.in_edges(n):
        #                         if ie[0] not in drop_nodes:
        #                             in_edge = (
        #                                 ie[0],
        #                                 f"grouped_layers_{count}_type_{group_type_count}",
        #                             )
        #                             add_in_edges += [in_edge]
        #                 if len(list(graph.successors(n))) > 0 and (
        #                     graph.out_degree(n) > 1
        #                 ):
        #                     for oe in graph.out_edges(n):
        #                         if oe[1] not in drop_nodes:
        #                             out_edge = (
        #                                 f"grouped_layers_{count}_type_{group_type_count}",
        #                                 oe[1],
        #                             )
        #                             add_out_edges += [out_edge]
        #             _logger.debug(
        #                 f"replacing {c} with group type: {group_type_count}")
        #             count += 1
        #         if (
        #             len(add_in_edges) + len(add_out_edges) > 0
        #             and len(add_in_edges) <= 2
        #             and len(add_out_edges) <= 2
        #             and not (
        #                 len(add_in_edges) == 2 and len(add_out_edges) == 2
        #             )  # TODO: this part 'and not (len(add_in_edges) == 2 and len(add_out_edges) == 2)' should be revisited
        #         ):
        #             graph.remove_nodes_from(drop_nodes)
        #             graph.remove_edges_from(drop_edges)
        #             graph.add_nodes_from(add_nodes)
        #             graph.add_edges_from(add_in_edges + add_out_edges)
        #         elif (
        #             len(add_in_edges) > 2
        #             or len(add_out_edges) > 2
        #             or (
        #                 len(add_in_edges) == 2 and len(add_out_edges) == 2
        #             )  # TODO: this part 'or (len(add_in_edges) == 2 and len(add_out_edges) == 2)' should be revisited
        #         ):
        #             _logger.debug(f"skipping {c}")
        #             blacklist_combinations.append(type_conb)

        #     if frequent_comb in blacklist_combinations:
        #         continue
        #     _logger.debug(
        #         f"visualizing graph: graph_groups_{group_type_count}_{count}")
        #     self.visualize_graph(
        #         graph,
        #         os.getcwd()
        #         + f"/fpga_modeling_reports/layer_grouppings/graph_groups_{group_type_count}_{count}",
        #     )
        #     group_type_count += 1

        # for n in graph.nodes():
        #     if "group" in n:
        #         _logger.debug(
        #             f"{n} -> {graph.nodes[n]['underlying_layers_type']}")
        #     else:
        #         _logger.debug(f"{n} -> {graph.nodes[n]['hw_type']}")
