import os
from dataclasses import dataclass

import yaml

import wandb
from fpga_hart import _logger
from fpga_hart.layers.layer_design import layer_design_points
from fpga_hart.parser.model_descriptor import \
    ModelLayerDescriptor
from fpga_hart.utils import utils


def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results


@dataclass
class LayerParser(ModelLayerDescriptor):
    config: wandb.Config
    singlethreaded: bool = False
    per_layer_plot: bool = False
    enable_wandb: bool = False

    def __post_init__(self) -> None:
        ModelLayerDescriptor.__post_init__(self)  # Initialize the parent class
        # _logger.setLevel(level=logging.DEBUG)

        if not os.path.exists(os.path.join(os.getcwd(), "fpga_modeling_reports", self.model_name)):
            os.makedirs(os.path.join(os.getcwd(), "fpga_modeling_reports", self.model_name))

        self.layer_model_file = os.path.join(
            os.getcwd(), "fpga_modeling_reports", self.model_name, f"{self.model_name}_layers.json"
        )

        with open("fpga_hart/config/report_template.yaml", "r") as yaml_file:
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
            self.config,
            self.layer_model_file,
            self.report_dict,
            self.singlethreaded,
        )

    def parse(self) -> None:
        if os.path.exists(self.layer_model_file):
            os.remove(self.layer_model_file)

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
            op_type = "MaxPool"
            name = "custom_Pool_layer"
            layer_descriptor = {
                "operation": op_type,
                "shape_in": [[1, 3, 8, 16, 16]],
                "shape_out": [1, 3, 8, 8, 8],
                "node_in": ["323"],
                "node_out": "324",
                "branching": False,
                "kernel": [1, 3, 3],
                "padding": [0, 1, 1],
                "stride": [1, 2, 2],
            }
        elif layer_type == "Activation":
            op_type = "Sigmoid"
            name = f"custom_{op_type}_layer"
            layer_descriptor = {
                "operation": op_type,
                "shape_in": [[1, 432, 16, 8, 8]],
                "shape_out": [1, 432, 16, 8, 8],
                "node_in": ["323"],
                "node_out": "324",
                "branching": False
            }
        elif layer_type == "Conv":
            name = "custom_Conv_layer"
            layer_descriptor = {
                "operation": "Conv",
                "shape_in": [[1, 2, 3, 4, 4]],
                "shape_out": [1, 4, 3, 4, 4],
                "node_in": ["575"],
                "node_out": "576",
                "branching": False,
                "kernel": [4, 2, 3, 1, 1],
                "bias": [4],
                "padding": [1, 0, 0],
                "stride": [1, 1, 1],
                "groups": 1,
                "dilation": [1, 1, 1],
            }

            # layer_descriptor = {
            #     "operation": "Conv",
            #     "shape_in": [[1, 4, 3, 4, 4]],
            #     "shape_out": [1, 4, 3, 4, 4],
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
            name = "custom_Gemm_layer"
            layer_descriptor = {
                "operation": "Gemm",
                "shape_in": [[1, 10]],
                "shape_out": [1, 20],
                "node_in": ["960"],
                "node_out": "970",
                "branching": False,
                "kernel": [10, 20],
                "bias": [20],
            }
        elif layer_type == "GlobalAveragePool":
            name = "custom_Gap_layer"
            layer_descriptor = {
                "operation": "GlobalAveragePool",
                "shape_in": [[1, 432, 16, 8, 8]],
                "shape_out": [1, 432, 1, 1, 1],
                "node_in": ["960"],
                "node_out": "970",
                "branching": False,
            }
        elif layer_type == "Elementwise":
            op_type = "Mul"
            name = f"custom_{op_type}_layer"
            layer_descriptor = {
                "operation": op_type,
                "shape_in": [[1, 432, 16, 8, 8], [1, 432, 1, 1, 1]],
                "shape_out": [1, 432, 16, 8, 8],
                "node_in": ["323"],
                "node_out": "324",
                "branching": False
            }

        self.model_layer(name, layer_descriptor)
