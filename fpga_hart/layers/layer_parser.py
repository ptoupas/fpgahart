import csv
import logging
import os
from dataclasses import dataclass

import numpy as np
import wandb
import yaml
from fpga_hart import _logger
from fpga_hart.layers.layer_design import layer_design_points
from fpga_hart.network_representation.model_descriptor import \
    ModelLayerDescriptor
from fpga_hart.utils import utils


def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results


@dataclass
class LayerParser(ModelLayerDescriptor):
    wandb_config: wandb.Config
    singlethreaded: bool = False
    per_layer_plot: bool = False

    def __post_init__(self) -> None:
        ModelLayerDescriptor.__post_init__(self)  # Initialize the parent class
        # _logger.setLevel(level=logging.DEBUG)

        if not os.path.exists(os.path.join(os.getcwd(), "fpga_modeling_reports", self.model_name)):
            os.makedirs(os.path.join(os.getcwd(), "fpga_modeling_reports", self.model_name))

        self.layer_model_file = os.path.join(
            os.getcwd(), "fpga_modeling_reports", self.model_name, f"{self.model_name}_layers.json"
        )

        with open("fpga_hart/config/report_temlate.yaml", "r") as yaml_file:
            self.report_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

        self.pareto_results = False
        if self.pareto_results:
            self.layer_model_file_par = os.path.join(
                os.getcwd(),
                "fpga_modeling_reports",
                self.model_name, "individual_layers_pareto.json",
            )

    def model_layer(self, layer: str, layer_description: dict) -> None:
        _logger.info("Modeling {} layer...".format(layer))
        layer_design_points(
            layer,
            layer_description,
            95.0 if self.wandb_config == None else self.wandb_config.max_dsp_util,
            95.0 if self.wandb_config == None else self.wandb_config.max_bram_util,
            self.layer_model_file,
            self.report_dict,
            self.singlethreaded,
        )

    def parse(self) -> None:

        for name, descriptor in self.layers.items():
            self.model_layer(name, descriptor)

        if self.pareto_results:
            utils.drop_duplicates_csv(self.layer_model_file)
            utils.get_paretto_csv(self.layer_model_file_par, self.layer_model_file)
            if self.per_layer_plot:
                utils.plot_layers_csv(self.layer_model_file_par, self.model_name)

    def model_custom_layer(self, layer_type: str) -> None:
        supported_types = ["Conv", "Pool", "GlobalAveragePool", "Gemm", "Mul", "Add", "Relu", "Sigmoid", "Swish"]
        if layer_type not in supported_types:
            raise ValueError(
                "Layer type {} not supported. Supported layer types are: {}".format(
                    layer_type, supported_types
                )
            )

        if layer_type in ["Relu", "Sigmoid", "Swish"]:
            layer_type = "Activation"
        elif layer_type in ["Mul", "Add"]:
            layer_type = "Elementwise"

        if not os.path.exists(os.path.join(os.getcwd(), "fpga_modeling_reports", "custom_layers", layer_type)):
            os.makedirs(os.path.join(os.getcwd(), "fpga_modeling_reports", "custom_layers", layer_type))
        self.layer_model_file = os.path.join(
            os.getcwd(), "fpga_modeling_reports", "custom_layers", layer_type, f"{layer_type}_layers.json"
        )
        if os.path.exists(self.layer_model_file):
            os.remove(self.layer_model_file)

        if layer_type == "Pool":
            name = "custom_pool_layer"
            layer_descriptor = {
                "operation": "MaxPool",
                "shape_in": [[1, 64, 8, 128, 128]],
                "shape_out": [1, 64, 8, 64, 64],
                "node_in": ["323"],
                "node_out": "324",
                "branching": False,
                "kernel": [1, 3, 3],
                "padding": [0, 1, 1],
                "stride": [1, 2, 2],
            }
        elif layer_type == "Activation":
            name = "custom_activation_layer"
            layer_descriptor = {
                "operation": "Sigmoid",
                "shape_in": [[1, 432, 16, 8, 8]],
                "shape_out": [1, 432, 16, 8, 8],
                "node_in": ["323"],
                "node_out": "324",
                "branching": False
            }
        elif layer_type == "Conv":
            name = "custom_conv_layer"
            layer_descriptor = {
                "operation": "Conv",
                "shape_in": [[1, 3, 8, 16, 16]],
                "shape_out": [1, 6, 8, 8, 8],
                "node_in": ["575"],
                "node_out": "576",
                "branching": False,
                "kernel": [6, 3, 3, 3, 3],
                "bias": [6],
                "padding": [1, 1, 1],
                "stride": [1, 2, 2],
                "groups": 1,
                "dilation": [1, 1, 1],
            }

            # layer_descriptor = {
            #     "operation": "Conv",
            #     "shape_in": [[1, 4, 4, 6, 6]],
            #     "shape_out": [1, 4, 4, 6, 6],
            #     "node_in": ["575"],
            #     "node_out": "576",
            #     "branching": False,
            #     "kernel": [4, 1, 3, 3, 3],
            #     "bias": [4],
            #     "padding": [1, 1, 1],
            #     "stride": [1, 1, 1],
            #     "groups": 4,
            #     "dilation": [1, 1, 1],
            # }

            # layer_descriptor = {
            #     "operation": "Conv",
            #     "shape_in": [[1, 4, 8, 12, 12]],
            #     "shape_out": [1, 6, 8, 12, 12],
            #     "node_in": ["575"],
            #     "node_out": "576",
            #     "branching": False,
            #     "kernel": [6, 4, 1, 1, 1],
            #     "bias": [6],
            #     "padding": [0, 0, 0],
            #     "stride": [1, 1, 1],
            #     "groups": 1,
            #     "dilation": [1, 1, 1],
            # }
        elif layer_type == "Gemm":
            name = "custom_gemm_layer"
            layer_descriptor = {
                "operation": "Gemm",
                "shape_in": [[1, 50]],
                "shape_out": [1, 100],
                "node_in": ["960"],
                "node_out": "970",
                "branching": False,
                "kernel": [50, 100],
                "bias": [100],
            }
        elif layer_type == "GlobalAveragePool":
            name = "custom_gap_layer"
            layer_descriptor = {
                "operation": "GlobalAveragePool",
                "shape_in": [[1, 432, 16, 8, 8]],
                "shape_out": [1, 432, 1, 1, 1],
                "node_in": ["960"],
                "node_out": "970",
                "branching": False,
            }
        elif layer_type == "Elementwise":
            name = "custom_elemwise_layer"
            layer_descriptor = {
                "operation": "Mul",
                "shape_in": [[1, 432, 16, 8, 8], [1, 432, 1, 1, 1]],
                "shape_out": [1, 432, 16, 8, 8],
                "node_in": ["323"],
                "node_out": "324",
                "branching": False
            }

            # layer_descriptor = {
            #     "operation": "Add",
            #     "shape_in": [[1, 192, 16, 8, 8], [1, 192, 16, 8, 8]],
            #     "shape_out": [1, 192, 16, 8, 8],
            #     "node_in": ["323"],
            #     "node_out": "324",
            #     "branching": False
            # }

        self.model_layer(name, layer_descriptor)
