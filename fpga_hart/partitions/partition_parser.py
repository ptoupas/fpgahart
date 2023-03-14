import configparser
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import wandb
from fpga_hart import _logger
from fpga_hart.layers.activation_3d import Activation3DLayer
from fpga_hart.layers.batchnorm_3d import BatchNorm3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise_3d import ElementWise3DLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap_3d import GAP3DLayer
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.layers.squeeze_excitation import SqueezeExcitationLayer
from fpga_hart.optimizer.simulated_annealing.sa import SimulatedAnnealing
from fpga_hart.parser.model_descriptor import ModelLayerDescriptor
from fpga_hart.platform.platform import Platform
from fpga_hart.utils import utils
from fpga_hart.utils.graph_manipulation import visualize_graph

sns.set(rc={"figure.figsize": (15, 8)})
sns.set_style("whitegrid")

def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results


@dataclass
class PartitionParser(ModelLayerDescriptor):
    gap_approx: bool
    singlethreaded: bool
    per_layer_plot: bool
    config: wandb.Config
    enable_wandb: bool

    from fpga_hart.partitions.partition_descriptor import create_partitions

    def __post_init__(self) -> None:
        ModelLayerDescriptor.__post_init__(self)  # Initialize the parent class
        # _logger.setLevel(level=logging.DEBUG)

        self.partitions = self.create_partitions(self.layers)
        self.platform = Platform()

        columns = [
            "Partition Name",
            "Times Repeated",
            "Num Splits",
            "Times Weights Reloading",
            "latency(C)",
            "latency(S)",
            "GOP/s",
            "vols/s",
            "GOPs",
            "DSP %",
            "DSPs",
            "BRAM %",
            "BRAMs",
            "depth",
            "branch_depth",
            "dataSizeIn(MB)",
            "dataSizeOut(MB)",
        ]
        self.df = pd.DataFrame(columns=columns)

        self.model_avg_metrics = {}

        if not os.path.exists(os.path.join(os.getcwd(), "fpga_modeling_reports", self.model_name)):
            os.makedirs(os.path.join(os.getcwd(), "fpga_modeling_reports", self.model_name))

        self.partition_model_file = os.path.join(
            os.getcwd(), "fpga_modeling_reports", self.model_name, self.model_name + "_partitions.json"
        )

        if self.se_block:
            self.layer_model_file = os.path.join(
                os.getcwd(), "fpga_modeling_reports", self.model_name, self.model_name + "_se.json"
            )
            self.layer_model_file_par = os.path.join(
                os.getcwd(), "fpga_modeling_reports", self.model_name, self.model_name + "_se_pareto.json"
            )
        else:
            self.layer_model_file = os.path.join(
                os.getcwd(), "fpga_modeling_reports", self.model_name, self.model_name + ".json"
            )
            self.layer_model_file_par = os.path.join(
                os.getcwd(), "fpga_modeling_reports", self.model_name, self.model_name + "_pareto.json"
            )

    def is_partition_input(self, partition, node_ids):
        if len(node_ids) > 1:
            return False
        for layer in partition:
            if node_ids[0] == self.layers[layer]["node_out"]:
                return False
        return True

    def is_partition_output(self, partition, node_id):
        for layer in partition:
            if node_id in self.layers[layer]["node_in"]:
                return False
        return True

    def connected_nodes(self, partition, node_id):
        nodes = []
        for layer in partition:
            if node_id in self.layers[layer]["node_in"]:
                nodes.append(layer)
        return nodes

    def reduce_node_shapes(
        self,
        layer,
        channels_reduction_rate=2,
        depth_reduction_rate=1,
        height_reduction_rate=1,
        width_reduction_rate=1,
    ):
        for i, node_in in enumerate(self.layers[layer]["shape_in"]):
            new_shape_in = node_in.copy()
            new_shape_in[1] = new_shape_in[1] // channels_reduction_rate
            new_shape_in[2] = (
                new_shape_in[2] // depth_reduction_rate if new_shape_in[2] > 1 else 1
            )
            new_shape_in[3] = (
                new_shape_in[3] // height_reduction_rate if new_shape_in[3] > 1 else 1
            )
            new_shape_in[4] = (
                new_shape_in[4] // width_reduction_rate if new_shape_in[4] > 1 else 1
            )
            self.layers[layer]["shape_in"][i] = new_shape_in

        new_shape_out = self.layers[layer]["shape_out"].copy()
        new_shape_out[1] = new_shape_out[1] // channels_reduction_rate
        new_shape_out[2] = (
            new_shape_out[2] // depth_reduction_rate if new_shape_out[2] > 1 else 1
        )
        new_shape_out[3] = (
            new_shape_out[3] // height_reduction_rate if new_shape_out[3] > 1 else 1
        )
        new_shape_out[4] = (
            new_shape_out[4] // width_reduction_rate if new_shape_out[4] > 1 else 1
        )
        self.layers[layer]["shape_out"] = new_shape_out

        if self.layers[layer]["operation"] == "Conv":
            new_kernel_shape = self.layers[layer]["kernel"].copy()
            new_kernel_shape[0] = (
                new_kernel_shape[0] // channels_reduction_rate
                if new_kernel_shape[0] > 1
                else 1
            )
            new_kernel_shape[1] = (
                new_kernel_shape[1] // channels_reduction_rate
                if new_kernel_shape[1] > 1
                else 1
            )
            self.layers[layer]["kernel"] = new_kernel_shape

            if self.layers[layer]["bias"]:
                new_bias = self.layers[layer]["bias"].copy()
                new_bias[0] = new_bias[0] // channels_reduction_rate
                self.layers[layer]["bias"] = new_bias

            if self.layers[layer]["groups"] > 1:
                new_groups = self.layers[layer]["groups"] // channels_reduction_rate
                self.layers[layer]["groups"] = new_groups

    def create_graph(self, partition: list) -> nx.DiGraph:
        graph = nx.DiGraph()
        _logger.info("*" * 40)
        for layer in partition:
            _logger.info("Adding {} layer to graph...".format(layer))
            # if not 'Gemm_401' in partition:
            #     self.reduce_node_shapes(layer, channels_reduction_rate=4, depth_reduction_rate=1, height_reduction_rate=1, width_reduction_rate=1)
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
            elif self.layers[layer]["operation"] == "SqueezeExcitation":
                layer_type = self.layers[layer]["operation"]
                hw_layer = SqueezeExcitationLayer(
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
                hw_type = utils.get_conv_type(
                    layer=self.layers[layer],
                    discriminate_kernel_size=True,
                    discriminate_stide=False,
                    discriminate_padding=False,
                )
            elif self.layers[layer]["operation"] in ["MaxPool", "AveragePool"]:
                hw_type = utils.get_pool_type(
                    layer=self.layers[layer],
                    discriminate_kernel_size=True,
                    discriminate_stide=False,
                    discriminate_padding=False,
                )
            else:
                hw_type = layer_type  # self.layers[layer]["operation"]
            # TODO: use the LAYER_TYPE enum for the layer type param
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

    def model_partition(self, partition: list, name: str) -> None:

        graph = self.create_graph(partition)

        if not os.path.exists(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/partition_graphs/"):
            os.makedirs(os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/partition_graphs/")
        visualize_graph(
            graph,
            os.getcwd() + "/fpga_modeling_reports/" + self.model_name + "/partition_graphs/" + name,
            self.enable_wandb,
            name,
        )

        _logger.info("Partition: {}: ".format(name))
        # optimizer = SimulatedAnnealing(graph, config=self.config, platform=self.platform, branch_mem=branch_buffer, partition_name=name, gap_approx=self.gap_approx, enable_wandb=self.enable_wandb, cnn_model_name=self.model_name)
        optimizer = SimulatedAnnealing(
            graph,
            config=self.config,
            platform=self.platform,
            partition_name=name,
            gap_approx=self.gap_approx,
            enable_wandb=self.enable_wandb,
            cnn_model_name=self.model_name,
        )

        mwpc, solution_mem, solution_dp, extra_reconfig, weights_reloading = optimizer.run_solver(mode="partition")
        if mwpc is None or solution_mem is None or solution_dp is None:
            raise Exception(f"Optimization failed for layer {name}")

        num_graphs = len(solution_mem)

        for i, (solution, wr) in enumerate(zip(solution_dp, weights_reloading)):
            part_name = name + "_split" + str(i) if num_graphs > 1 else name

            partition_results = deepcopy(solution)
            log_metrics = {}
            log_metrics["latency(C)"] = partition_results["latency(C)"]
            log_metrics["latency(S)"] = partition_results["latency(S)"]
            log_metrics["GOP/s"] = partition_results["GOP/s"]
            log_metrics["vols/s"] = partition_results["vols/s"]
            log_metrics["GOPs"] = partition_results["GOPs"]
            log_metrics["DSP %"] = partition_results["DSP"]
            log_metrics["BRAM %"] = partition_results["BRAM"]
            log_metrics["depth"] = partition_results["depth"]

            self.model_avg_metrics = log_metrics

            times_repeat = 1 if name.count('+') == 0 else name.count('+') + 1
            self.df.loc[len(self.df.index)] = [
                part_name,
                times_repeat,
                extra_reconfig+1,
                wr,
                partition_results["latency(C)"],
                partition_results["latency(S)"],
                partition_results["GOP/s"],
                partition_results["vols/s"],
                partition_results["GOPs"],
                partition_results["DSP"],
                partition_results["DSP_RAW"],
                partition_results["BRAM"],
                partition_results["BRAM_RAW"],
                partition_results["depth"],
                json.dumps(partition_results["branch_depth"], indent=2),
                partition_results["dataSizeIn"],
                partition_results["dataSizeOut"],
            ]

            report_dict = {}
            if self.enable_wandb:
                report_dict[part_name] = {
                    "Times Repeated": times_repeat,
                    "Num Splits": extra_reconfig+1,
                    "Times Weights Reloading": wr,
                    "config": partition_results["config"],
                    "structure": partition_results["structure"],
                }
                artifact = wandb.Artifact("partitions", type="json")
                with artifact.new_file("partition_config.json") as f:
                    json.dump(report_dict, f, indent=2)
                wandb.log_artifact(artifact)

            report_dict[part_name] = {
                "Times Repeated": times_repeat,
                "Num Splits": extra_reconfig+1,
                "Times Weights Reloading": wr,
                "Latency(C)": partition_results["latency(C)"],
                "Latency(S)": partition_results["latency(S)"],
                "GOP/s": partition_results["GOP/s"],
                "vols/s": partition_results["vols/s"],
                "GOPs": partition_results["GOPs"],
                "DSP %": partition_results["DSP"],
                "DSPs": partition_results["DSP_RAW"],
                "BRAM %": partition_results["BRAM"],
                "BRAMs": partition_results["BRAM_RAW"],
                "depth": partition_results["depth"],
                "branch_depth": partition_results["branch_depth"],
                "dataSizeIn(MB)": partition_results["dataSizeIn"],
                "dataSizeOut(MB)": partition_results["dataSizeOut"],
                "config": partition_results["config"],
                "structure": partition_results["structure"],
            }
            utils.update_report_file(self.partition_model_file, report_dict)
        return extra_reconfig

    def idetify_sequential_duplicates(self):
        partitions = {}
        for i, partition in enumerate(self.partitions):
            graph = self.create_graph(partition)
            nodes_list = []
            for node in graph.nodes():
                hw = graph.nodes[node]["hw"]
                if graph.nodes[node]["type"] == "Conv":
                    nodes_list.append(
                        [
                            hw.input_shape,
                            hw.output_shape,
                            hw.kernel_shape,
                            hw.padding,
                            hw.stride,
                            hw.groups,
                        ]
                    )
                elif graph.nodes[node]["type"] == "Pooling":
                    nodes_list.append(
                        [
                            hw.input_shape,
                            hw.output_shape,
                            hw.kernel_shape,
                            hw.padding,
                            hw.stride,
                        ]
                    )
                elif graph.nodes[node]["type"] == "ElementWise":
                    nodes_list.append(
                        [hw.input_shape_1, hw.input_shape_2, hw.output_shape]
                    )
                else:
                    nodes_list.append([hw.input_shape, hw.output_shape])
            partitions[i] = nodes_list

        prev_partition = None
        remove_duplicates = []
        inherit_duplicates = {}
        for i, partition in enumerate(self.partitions):
            v = partitions[i]
            if prev_partition is not None:
                if v == prev_partition:
                    remove_duplicates.append(i)
                    inherit_duplicates[i] = i - 1
            prev_partition = v

        for k, v in inherit_duplicates.items():
            if v in inherit_duplicates:
                inherit_duplicates[k] = inherit_duplicates[v]
        res = {}
        for i, v in inherit_duplicates.items():
            res[v] = [i] if v not in res.keys() else res[v] + [i]

        for index in sorted(remove_duplicates, reverse=True):
            del self.partitions[index]
        return res

    def parse(self):
        if os.path.exists(self.partition_model_file):
            os.remove(self.partition_model_file)

        if False:
            #TODO: Find a way to combine this with the partition split functionality
            duplicates_dict = self.idetify_sequential_duplicates()
        num_dev_reconfig = len(self.partitions) - 1
        print("Initial number of device reconfigurations: {}".format(num_dev_reconfig))

        start = time.time()

        if False:
            #TODO: Find a way to combine this with the partition split functionality
            name_offset = 0
            for i, partition in enumerate(self.partitions):
                part_name = "part_{}".format(i+name_offset)
                if i+name_offset in duplicates_dict:
                    part_name += "+"
                    part_name += "+".join([str(x) for x in duplicates_dict[i+name_offset]])
                times_called = 1 if i+name_offset not in duplicates_dict else 1 + len(duplicates_dict[i+name_offset])
                name_offset += times_called - 1
                num_dev_reconfig += self.model_partition(partition, name=part_name)
        for i, partition in enumerate(self.partitions):
            part_name = "part_{}".format(i)
            num_dev_reconfig += self.model_partition(partition, name=part_name)

        print("Final number of device reconfigurations: {}.".format(num_dev_reconfig))

        for key in self.model_avg_metrics:
            self.model_avg_metrics[key] = self.df[key].repeat(self.df["Times Repeated"].to_list()).mean()
        self.model_avg_metrics["latency(C) Sum"] = int((self.df["latency(C)"] * self.df["Times Repeated"]).sum())
        self.model_avg_metrics["latency(S) Sum"] = (self.df["latency(S)"] * self.df["Times Repeated"]).sum()
        self.model_avg_metrics["GOPs Sum"] = (self.df["GOPs"] * self.df["Times Repeated"]).sum()
        self.model_avg_metrics["depth Sum"] = int((self.df["depth"] * self.df["Times Repeated"]).sum())

        if not self.enable_wandb:
            log_results_path = os.path.join(os.getcwd(), "fpga_modeling_reports", self.model_name, "partition_results")
            if not os.path.exists(log_results_path):
                os.makedirs(log_results_path)

        batch_size = np.arange(1, 500, 1)
        lat_sec = ((self.model_avg_metrics["latency(C) Sum"] - self.model_avg_metrics["depth Sum"]) * batch_size + self.model_avg_metrics["depth Sum"]) / (self.platform.clock_freq * 1e6) + (self.platform.reconfiguration_time * num_dev_reconfig)
        plt.plot(batch_size, lat_sec)
        plt.xlabel("Batch Size")
        plt.ylabel("Seconds")
        plt.title("Latency vs Batch Size")
        if self.enable_wandb:
            wandb.log({"Latency vs Batch Size": plt})
        else:
            plt.savefig(os.path.join(log_results_path, "latency_vs_batch_size.png"))
        through_gops_sec = (self.model_avg_metrics["GOPs Sum"] * batch_size) / lat_sec
        plt.cla()
        plt.clf()
        plt.plot(batch_size, through_gops_sec)
        plt.xlabel("Batch Size")
        plt.ylabel("GOPs/s")
        plt.title("Throughput (GOPs/s) vs Batch Size")
        if self.enable_wandb:
            wandb.log({"Throughput (GOPs/s) vs Batch Size": plt})
        else:
            plt.savefig(os.path.join(log_results_path, "throughput_gops_vs_batch_size.png"))
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
            plt.savefig(os.path.join(log_results_path, "throughput_vols_vs_batch_size.png"))
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
            plt.savefig(os.path.join(log_results_path, "throughput_gops_dsp_vs_batch_size.png"))
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
            plt.savefig(os.path.join(log_results_path, "throughput_gops_dsp_cycle_vs_batch_size.png"))

        self.model_avg_metrics["latency(S)-reconfig"] = {
            "Batch 1": lat_sec[0],
            "Batch 30": lat_sec[29],
            "Batch 100": lat_sec[99]
        }

        self.model_avg_metrics["GOPs/s"] = {
            "Batch 1": through_gops_sec[0],
            "Batch 30": through_gops_sec[29],
            "Batch 100": through_gops_sec[99]
        }
        self.model_avg_metrics["Volumes/s"] = {
            "Batch 1": through_vols_sec[0],
            "Batch 30": through_vols_sec[29],
            "Batch 100": through_vols_sec[99]
        }
        self.model_avg_metrics["GOPs/s/DSP"] = {
            "Batch 1": gops_sec_dsp[0],
            "Batch 30": gops_sec_dsp[29],
            "Batch 100": gops_sec_dsp[99]
        }
        self.model_avg_metrics["GOPs/s/DSP/cycle"] = {
            "Batch 1": gops_sec_dsp_cycle[0],
            "Batch 30": gops_sec_dsp_cycle[29],
            "Batch 100": gops_sec_dsp_cycle[99]
        }

        del self.model_avg_metrics["latency(C)"]
        del self.model_avg_metrics["latency(S)"]
        del self.model_avg_metrics["GOPs"]
        del self.model_avg_metrics["depth"]

        if self.enable_wandb:
            wandb.log(self.model_avg_metrics)
            wandb.log({"Partition Results": wandb.Table(dataframe=self.df)})
        else:
            with open(self.partition_model_file, 'r') as fp:
                dictObj = json.load(fp)

            dictObj['metrics'] = self.model_avg_metrics

            with open(self.partition_model_file, 'w') as json_file:
                json.dump(dictObj, json_file,
                                    indent=2)
        end = time.time()
        _logger.info("Partition modeling took {:.2f} seconds".format(end - start))

    def model_custom_partition(self, name: str):
        if not os.path.exists(os.path.join(os.getcwd(), "fpga_modeling_reports", "custom_partitions", name)):
            os.makedirs(os.path.join(os.getcwd(), "fpga_modeling_reports", "custom_partitions", name))
        self.partition_model_file = os.path.join(
            os.getcwd(), "fpga_modeling_reports", "custom_partitions", name, f"{name}_layers.json"
        )
        if os.path.exists(self.partition_model_file):
            os.remove(self.partition_model_file)

        # custom_partition = ['Relu_80', 'Conv_81', 'Relu_83', 'Conv_84', 'GlobalAveragePool_86', 'Conv_87', 'Relu_88', 'Conv_89', 'Sigmoid_90']
        # custom_partition = ['Relu_80', 'Conv_81', 'Relu_83', 'Conv_84', 'GlobalAveragePool_86', 'Conv_87', 'Relu_88', 'Conv_89', 'Sigmoid_90', 'Mul_91', 'Swish_92']
        # custom_partition = ['Relu_80', 'Conv_81', 'Relu_83', 'Conv_84', 'GlobalAveragePool_86', 'Conv_87', 'Relu_88', 'Conv_89', 'Sigmoid_90', 'Mul_91', 'Swish_92', 'Conv_94']
        # custom_partition = ["Swish_92", "Conv_94"]

        # extra_reconfig = self.model_partition(custom_partition, name=name)
        # return

        custom_partition = ["custom_Conv_1", "custom_Relu_1"] # "custom_Conv_1", "custom_Relu_1"
        self.layers["custom_Conv_1"] = {
            "operation": "Conv",
            "shape_in": [[1, 6, 4, 4, 4]],
            "shape_out": [1, 12, 4, 4, 4],
            "node_in": ["1"],
            "node_out": "2",
            "branching": False,
            "kernel": [12, 6, 1, 3, 3],
            "bias": [12],
            "padding": [0, 1, 1],
            "stride": [1, 1, 1],
            "groups": 1,
            "dilation": [1, 1, 1],
        }
        self.layers["custom_Relu_1"] = {
            "operation": "Relu",
            "shape_in": [[1, 12, 4, 4, 4]],
            "shape_out": [1, 12, 4, 4, 4],
            "node_in": ["2"],
            "node_out": "3",
            "branching": False,
        }
        extra_reconfig = self.model_partition(custom_partition, name=name)
        return

        custom_partition = [
            "Custom_Relu",
            "Custom_Conv_1",
            "Custom_Swish",
            "Custom_Add",
        ]
        self.layers["Custom_Relu"] = {
            "operation": "Relu",
            "shape_in": [[1, 16, 8, 12, 12]],
            "shape_out": [1, 16, 8, 12, 12],
            "node_in": ["1"],
            "node_out": "2",
            "branching": False,
        }
        self.layers["Custom_Conv_1"] = {
            "operation": "Conv",
            "shape_in": [[1, 16, 8, 12, 12]],
            "shape_out": [1, 16, 8, 12, 12],
            "node_in": ["2"],
            "node_out": "3",
            "branching": False,
            "kernel": [16, 1, 3, 3, 3],
            "bias": [],
            "padding": [1, 1, 1],
            "stride": [1, 1, 1],
            "groups": 16,
            "dilation": [1, 1, 1],
        }
        self.layers["Custom_Conv_2"] = {
            "operation": "Conv",
            "shape_in": [[1, 16, 8, 12, 12]],
            "shape_out": [1, 16, 8, 12, 12],
            "node_in": ["2"],
            "node_out": "3",
            "branching": False,
            "kernel": [16, 16, 1, 1, 1],
            "bias": [],
            "padding": [0, 0, 0],
            "stride": [1, 1, 1],
            "groups": 1,
            "dilation": [1, 1, 1],
        }
        self.layers["Custom_Swish"] = {
            "operation": "Swish",
            "shape_in": [[1, 16, 8, 12, 12]],
            "shape_out": [1, 16, 8, 12, 12],
            "node_in": ["3"],
            "node_out": "4",
            "branching": False,
        }
        self.layers["Custom_Add"] = {
            "operation": "Add",
            "shape_in": [[1, 16, 8, 12, 12], [1, 16, 8, 12, 12]],
            "shape_out": [1, 16, 8, 12, 12],
            "node_in": ["2", "4"],
            "node_out": "5",
            "branching": False,
        }
        # extra_reconfig = self.model_partition(custom_partition, name=name)
        # return

        custom_partition = [
            "Relu_22",
            "Conv_23",
            "Relu_25",
            "Conv_26",
            "Swish_28",
            "Conv_30",
            "Add_32",
        ]
        # custom_partition = ['Relu_22', 'Conv_23', 'Relu_25', 'Swish_28', 'Conv_30', 'Add_32']

        self.layers["Relu_22"]["shape_in"] = [[1, 8, 6, 12, 12]]
        self.layers["Relu_22"]["shape_out"] = [1, 8, 6, 12, 12]

        self.layers["Conv_23"]["shape_in"] = [[1, 8, 6, 12, 12]]
        self.layers["Conv_23"]["shape_out"] = [1, 12, 6, 12, 12]
        self.layers["Conv_23"]["kernel"] = [12, 8, 1, 1, 1]
        self.layers["Conv_23"]["bias"] = [12]

        self.layers["Relu_25"]["shape_in"] = [[1, 12, 6, 12, 12]]
        self.layers["Relu_25"]["shape_out"] = [1, 12, 6, 12, 12]

        self.layers["Conv_26"]["shape_in"] = [[1, 12, 6, 12, 12]]
        self.layers["Conv_26"]["shape_out"] = [1, 12, 6, 12, 12]
        self.layers["Conv_26"]["kernel"] = [12, 1, 3, 3, 3]
        self.layers["Conv_26"]["bias"] = [12]
        self.layers["Conv_26"]["groups"] = 12

        self.layers["Swish_28"]["shape_in"] = [[1, 12, 6, 12, 12]]
        self.layers["Swish_28"]["shape_out"] = [1, 12, 6, 12, 12]
        # self.layers['Swish_28']['node_in'] = ['594']

        self.layers["Conv_30"]["shape_in"] = [[1, 12, 6, 12, 12]]
        self.layers["Conv_30"]["shape_out"] = [1, 8, 6, 12, 12]
        self.layers["Conv_30"]["kernel"] = [8, 12, 1, 1, 1]
        self.layers["Conv_30"]["bias"] = [8]

        self.layers["Add_32"]["shape_in"] = [
            [1, 8, 6, 12, 12],
            [1, 8, 6, 12, 12],
        ]
        self.layers["Add_32"]["shape_out"] = [1, 8, 6, 12, 12]

        extra_reconfig = self.model_partition(custom_partition, name=name)

        custom_partition = [
            "Relu_33",
            "Conv_34",
            "Relu_36",
            "Conv_37",
            "GlobalAveragePool_39",
            "Conv_40",
            "Relu_41",
            "Conv_42",
            "Sigmoid_43",
            "Mul_44",
            "Swish_45",
            "Conv_47",
            "Add_49",
        ]
        # custom_partition = ['Conv_37', 'GlobalAveragePool_39', 'Conv_40', 'Relu_41', 'Conv_42', 'Sigmoid_43', 'Mul_44']

        self.layers["Relu_33"]["shape_in"] = [[1, 8, 6, 12, 12]]
        self.layers["Relu_33"]["shape_out"] = [1, 8, 6, 12, 12]

        self.layers["Conv_34"]["shape_in"] = [[1, 8, 6, 12, 12]]
        self.layers["Conv_34"]["shape_out"] = [1, 12, 6, 12, 12]
        self.layers["Conv_34"]["kernel"] = [12, 8, 1, 1, 1]
        self.layers["Conv_34"]["bias"] = [12]

        self.layers["Relu_36"]["shape_in"] = [[1, 12, 6, 12, 12]]
        self.layers["Relu_36"]["shape_out"] = [1, 12, 6, 12, 12]

        self.layers["Conv_37"]["shape_in"] = [[1, 12, 6, 12, 12]]
        self.layers["Conv_37"]["shape_out"] = [1, 12, 6, 12, 12]
        self.layers["Conv_37"]["kernel"] = [12, 1, 3, 3, 3]
        self.layers["Conv_37"]["bias"] = [12]
        self.layers["Conv_37"]["groups"] = 12

        self.layers["GlobalAveragePool_39"]["shape_in"] = [[1, 12, 6, 12, 12]]
        self.layers["GlobalAveragePool_39"]["shape_out"] = [
            1,
            12,
            1,
            1,
            1,
        ]

        self.layers["Conv_40"]["shape_in"] = [[1, 12, 1, 1, 1]]
        self.layers["Conv_40"]["shape_out"] = [1, 8, 1, 1, 1]
        self.layers["Conv_40"]["kernel"] = [8, 12, 1, 1, 1]
        self.layers["Conv_40"]["bias"] = [8]

        self.layers["Relu_41"]["shape_in"] = [[1, 8, 1, 1, 1]]
        self.layers["Relu_41"]["shape_out"] = [1, 8, 1, 1, 1]

        self.layers["Conv_42"]["shape_in"] = [[1, 8, 1, 1, 1]]
        self.layers["Conv_42"]["shape_out"] = [1, 12, 1, 1, 1]
        self.layers["Conv_42"]["kernel"] = [12, 8, 1, 1, 1]
        self.layers["Conv_42"]["bias"] = [12]

        self.layers["Sigmoid_43"]["shape_in"] = [[1, 12, 1, 1, 1]]
        self.layers["Sigmoid_43"]["shape_out"] = [1, 12, 1, 1, 1]

        self.layers["Mul_44"]["shape_in"] = [
            [1, 12, 6, 12, 12],
            [1, 12, 1, 1, 1],
        ]
        self.layers["Mul_44"]["shape_out"] = [1, 12, 6, 12, 12]

        self.layers["Swish_45"]["shape_in"] = [[1, 12, 6, 12, 12]]
        self.layers["Swish_45"]["shape_out"] = [1, 12, 6, 12, 12]

        self.layers["Conv_47"]["shape_in"] = [[1, 12, 6, 12, 12]]
        self.layers["Conv_47"]["shape_out"] = [1, 8, 6, 12, 12]
        self.layers["Conv_47"]["kernel"] = [8, 12, 1, 1, 1]
        self.layers["Conv_47"]["bias"] = [8]

        self.layers["Add_49"]["shape_in"] = [
            [1, 8, 6, 12, 12],
            [1, 8, 6, 12, 12],
        ]
        self.layers["Add_49"]["shape_out"] = [1, 8, 6, 12, 12]

        extra_reconfig = self.model_partition(custom_partition, name=name)

        custom_partition = [
            "Relu_50",
            "Conv_51",
            "Relu_53",
            "Conv_54",
            "GlobalAveragePool_56",
            "Conv_57",
            "Relu_58",
            "Conv_59",
            "Sigmoid_60",
            "Mul_61",
            "Swish_62",
            "Conv_64",
            "Conv_66",
            "Add_68",
        ]

        self.layers["Relu_50"]["shape_in"] = [[1, 8, 6, 12, 12]]
        self.layers["Relu_50"]["shape_out"] = [1, 8, 6, 12, 12]

        self.layers["Conv_51"]["shape_in"] = [[1, 8, 6, 12, 12]]
        self.layers["Conv_51"]["shape_out"] = [1, 12, 6, 12, 12]
        self.layers["Conv_51"]["kernel"] = [12, 8, 1, 1, 1]
        self.layers["Conv_51"]["bias"] = [12]

        self.layers["Relu_53"]["shape_in"] = [[1, 12, 6, 12, 12]]
        self.layers["Relu_53"]["shape_out"] = [1, 12, 6, 12, 12]

        self.layers["Conv_54"]["shape_in"] = [[1, 12, 6, 12, 12]]
        self.layers["Conv_54"]["shape_out"] = [1, 12, 6, 6, 6]
        self.layers["Conv_54"]["kernel"] = [12, 1, 3, 3, 3]
        self.layers["Conv_54"]["bias"] = [12]
        self.layers["Conv_54"]["groups"] = 12

        self.layers["GlobalAveragePool_56"]["shape_in"] = [[1, 12, 6, 6, 6]]
        self.layers["GlobalAveragePool_56"]["shape_out"] = [
            1,
            12,
            1,
            1,
            1,
        ]

        self.layers["Conv_57"]["shape_in"] = [[1, 12, 1, 1, 1]]
        self.layers["Conv_57"]["shape_out"] = [1, 8, 1, 1, 1]
        self.layers["Conv_57"]["kernel"] = [8, 12, 1, 1, 1]
        self.layers["Conv_57"]["bias"] = [8]

        self.layers["Relu_58"]["shape_in"] = [[1, 8, 1, 1, 1]]
        self.layers["Relu_58"]["shape_out"] = [1, 8, 1, 1, 1]

        self.layers["Conv_59"]["shape_in"] = [[1, 8, 1, 1, 1]]
        self.layers["Conv_59"]["shape_out"] = [1, 12, 1, 1, 1]
        self.layers["Conv_59"]["kernel"] = [12, 8, 1, 1, 1]
        self.layers["Conv_59"]["bias"] = [12]

        self.layers["Sigmoid_60"]["shape_in"] = [[1, 12, 1, 1, 1]]
        self.layers["Sigmoid_60"]["shape_out"] = [1, 12, 1, 1, 1]

        self.layers["Mul_61"]["shape_in"] = [
            [1, 12, 6, 6, 6],
            [1, 12, 1, 1, 1],
        ]
        self.layers["Mul_61"]["shape_out"] = [1, 12, 6, 6, 6]

        self.layers["Swish_62"]["shape_in"] = [[1, 12, 6, 6, 6]]
        self.layers["Swish_62"]["shape_out"] = [1, 12, 6, 6, 6]

        self.layers["Conv_64"]["shape_in"] = [[1, 12, 6, 6, 6]]
        self.layers["Conv_64"]["shape_out"] = [1, 10, 6, 6, 6]
        self.layers["Conv_64"]["kernel"] = [10, 12, 1, 1, 1]
        self.layers["Conv_64"]["bias"] = [10]

        self.layers["Conv_66"]["shape_in"] = [[1, 8, 6, 12, 12]]
        self.layers["Conv_66"]["shape_out"] = [1, 10, 6, 6, 6]
        self.layers["Conv_66"]["kernel"] = [10, 8, 1, 1, 1]
        self.layers["Conv_66"]["bias"] = [10]

        self.layers["Add_68"]["shape_in"] = [
            [1, 10, 6, 6, 6],
            [1, 10, 6, 6, 6],
        ]
        self.layers["Add_68"]["shape_out"] = [1, 10, 6, 6, 6]

        extra_reconfig = self.model_partition(custom_partition, name=name)
