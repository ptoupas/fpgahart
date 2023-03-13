from collections import deque
from dataclasses import dataclass

import numpy as np

from fpga_hart import _logger
from fpga_hart.parser.onnx_parser import OnnxModelParser


@dataclass
class ModelLayerDescriptor(OnnxModelParser):
    se_block: bool

    def __post_init__(self) -> None:
        OnnxModelParser.__post_init__(self)  # Initialize the parent class
        # _logger.setLevel(level=logging.INFO)
        self.layers = {}
        self.create_layers()

    def create_layers(self) -> None:
        if self.se_block:
            se_module = deque(maxlen=6)
        swish_module = deque(maxlen=2)

        for k in self.torch_layers.keys():

            name = k
            if "Flatten" in name:
                continue
            operation = self.torch_layers[k]["operation"]
            input_shape = [self.torch_layers[k]["input"][0]]
            output_shape = self.torch_layers[k]["output"]
            if operation in ["Relu", "Sigmoid", "Elu", "HardSigmoid", "LeakyRelu", "PRelu", "Selu", "Tanh", "Celu", "HardSwish", "Softmax"] and len(input_shape[0]) == 2 and len(output_shape) == 2:
                input_shape[0] = input_shape[0] + [1, 1, 1]
                output_shape = output_shape + [1, 1, 1]
            input_node = [self.torch_layers[k]["input_id"][0]]
            if self.model_name == "x3d_m":
                if name == "Gemm_401":
                    input_node = [self.torch_layers["GlobalAveragePool_391"]["output_id"]]
            if self.model_name == "slowonly":
                if name == "Gemm_181":
                    input_node = [self.torch_layers["GlobalAveragePool_172"]["output_id"]]
            if self.model_name == "r2plus1d":
                if name == "Gemm_239":
                    input_node = [self.torch_layers["GlobalAveragePool_230"]["output_id"]]
            if self.model_name == "c3d":
                if name == "Gemm_32":
                    input_node = [self.torch_layers["Relu_25"]["output_id"]]
                if name == "Gemm_22":
                    input_node = [self.torch_layers["MaxPool_20"]["output_id"]]
            output_node = self.torch_layers[k]["output_id"]

            self.layers[name] = {
                "operation": operation,
                "shape_in": input_shape,
                "shape_out": output_shape,
                "node_in": input_node,
                "node_out": output_node,
                "branching": False,
            }

            if operation == "Conv":
                self.layers[name]["kernel"] = self.torch_layers[k]["kernel"]
                self.layers[name]["bias"] = self.torch_layers[k]["bias"]
                self.layers[name]["padding"] = self.torch_layers[k]["padding"]
                self.layers[name]["stride"] = self.torch_layers[k]["stride"]
                self.layers[name]["groups"] = self.torch_layers[k]["groups"]
                self.layers[name]["dilation"] = self.torch_layers[k]["dilation"]
                if (
                    self.torch_layers[k]["input"][0][1]
                    == self.torch_layers[k]["output"][1]
                    == self.torch_layers[k]["groups"]
                ):
                    self.layers[name]["conv_type"] = "depthwise"
                elif np.prod(self.torch_layers[k]["kernel"][-3:]) == 1:
                    self.layers[name]["conv_type"] = "pointwise"
                elif (
                    self.torch_layers[k]["kernel"][2] != 1
                    and np.prod(self.torch_layers[k]["kernel"][-2:]) == 1
                ):
                    self.layers[name]["conv_type"] = "3d_conv_temporal"
                elif self.torch_layers[k]["kernel"][2] == 1 and (
                    self.torch_layers[k]["kernel"][3] > 1
                    and self.torch_layers[k]["kernel"][4] > 1
                ):
                    self.layers[name]["conv_type"] = "3d_conv_spatial"
                else:
                    self.layers[name]["conv_type"] = "3d_conv"
            elif operation == "AveragePool" or operation == "MaxPool":
                self.layers[name]["kernel"] = self.torch_layers[k]["kernel"]
                self.layers[name]["padding"] = self.torch_layers[k]["padding"]
                self.layers[name]["stride"] = self.torch_layers[k]["stride"]
            elif operation == "BatchNormalization":
                self.layers[name]["kernel"] = self.torch_layers[k]["kernel"]
                self.layers[name]["bias"] = self.torch_layers[k]["bias"]
                self.layers[name]["running_mean"] = self.torch_layers[k]["running_mean"]
                self.layers[name]["running_var"] = self.torch_layers[k]["running_var"]
            elif operation == "Gemm":
                self.layers[name]["kernel"] = self.torch_layers[k]["kernel"]
                self.layers[name]["bias"] = self.torch_layers[k]["bias"]

            if operation == "Add" or operation == "Mul" or operation == "MatMul":
                self.layers[name]["shape_in"].append(self.torch_layers[k]["input"][1])
                self.layers[name]["node_in"].append(self.torch_layers[k]["input_id"][1])
                if self.model_name == "x3d_m" and operation == "MatMul":
                    self.layers[name]["kernel"] = self.torch_layers[k]["kernel"]

            swish_module.append([operation, name])
            if swish_module[0][0] == "Sigmoid" and swish_module[1][0] == "Mul":
                mul_shapein_1 = self.torch_layers[swish_module[1][1]]["input"][0]
                mul_shapein_2 = self.torch_layers[swish_module[1][1]]["input"][1]
                if mul_shapein_1 == mul_shapein_2:
                    _logger.debug("Creating Swish Activation Module")

                    sigmoid_name = swish_module[0][1]
                    operation = self.torch_layers[sigmoid_name]["operation"]
                    input_shape = [self.torch_layers[sigmoid_name]["input"][0]]
                    input_node = [self.torch_layers[sigmoid_name]["input_id"][0]]
                    swish_input_shape = input_shape
                    swish_input_node = input_node
                    output_shape = self.torch_layers[sigmoid_name]["output"]
                    output_node = self.torch_layers[sigmoid_name]["output_id"]

                    sigmoid = {
                        "operation": operation,
                        "shape_in": input_shape,
                        "shape_out": output_shape,
                        "node_in": input_node,
                        "node_out": output_node,
                        "branching": False,
                    }

                    mul_name = swish_module[1][1]
                    operation = self.torch_layers[mul_name]["operation"]
                    input_shape = self.torch_layers[mul_name]["input"]
                    input_node = self.torch_layers[mul_name]["input_id"]
                    output_shape = self.torch_layers[mul_name]["output"]
                    output_node = self.torch_layers[mul_name]["output_id"]
                    swish_output_shape = output_shape
                    swish_output_node = output_node

                    mul = {
                        "operation": operation,
                        "shape_in": input_shape,
                        "shape_out": output_shape,
                        "node_in": input_node,
                        "node_out": output_node,
                        "branching": False,
                    }

                    name = "Swish_" + swish_module[0][1].split("_")[1]
                    operation = "Swish"
                    self.layers[name] = {
                        "operation": operation,
                        "shape_in": swish_input_shape,
                        "shape_out": swish_output_shape,
                        "node_in": swish_input_node,
                        "node_out": swish_output_node,
                        "shape_branch": swish_input_shape,
                        "branching": True,
                        "primitive_ops": {sigmoid_name: sigmoid, mul_name: mul},
                    }

                    del self.layers[sigmoid_name]
                    del self.layers[mul_name]

            if self.se_block:
                se_module.append([operation, name])
                if (
                    se_module[0][0] == "GlobalAveragePool"
                    and se_module[1][0] == "Conv"
                    and se_module[2][0] == "Relu"
                    and se_module[3][0] == "Conv"
                    and se_module[4][0] == "Sigmoid"
                    and se_module[5][0] == "Mul"
                ):
                    _logger.debug("Creating Squeeze and Excitation Module")

                    gap_name = se_module[0][1]
                    operation = self.torch_layers[gap_name]["operation"]
                    input_shape = [self.torch_layers[gap_name]["input"][0]]
                    input_node = [self.torch_layers[gap_name]["input_id"][0]]
                    se_input_shape = input_shape
                    se_input_node = input_node
                    output_shape = self.torch_layers[gap_name]["output"]
                    output_node = self.torch_layers[gap_name]["output_id"]

                    gap = {
                        "operation": operation,
                        "shape_in": input_shape,
                        "shape_out": output_shape,
                        "node_in": input_node,
                        "node_out": output_node,
                        "branching": False,
                    }

                    conv1_name = se_module[1][1]
                    operation = self.torch_layers[conv1_name]["operation"]
                    input_shape = [self.torch_layers[conv1_name]["input"][0]]
                    input_node = [self.torch_layers[conv1_name]["input_id"][0]]
                    output_shape = self.torch_layers[conv1_name]["output"]
                    output_node = self.torch_layers[conv1_name]["output_id"]

                    conv1 = {
                        "operation": operation,
                        "shape_in": input_shape,
                        "shape_out": output_shape,
                        "node_in": input_node,
                        "node_out": output_node,
                        "kernel": self.torch_layers[conv1_name]["kernel"],
                        "bias": self.torch_layers[conv1_name]["bias"],
                        "padding": self.torch_layers[conv1_name]["padding"],
                        "stride": self.torch_layers[conv1_name]["stride"],
                        "groups": self.torch_layers[conv1_name]["groups"],
                        "dilation": self.torch_layers[conv1_name]["dilation"],
                        "branching": False,
                    }

                    relu_name = se_module[2][1]
                    operation = self.torch_layers[relu_name]["operation"]
                    input_shape = [self.torch_layers[relu_name]["input"][0]]
                    input_node = [self.torch_layers[relu_name]["input_id"][0]]
                    output_shape = self.torch_layers[relu_name]["output"]
                    output_node = self.torch_layers[relu_name]["output_id"]

                    relu = {
                        "operation": operation,
                        "shape_in": input_shape,
                        "shape_out": output_shape,
                        "node_in": input_node,
                        "node_out": output_node,
                        "branching": False,
                    }

                    conv2_name = se_module[3][1]
                    operation = self.torch_layers[conv2_name]["operation"]
                    input_shape = [self.torch_layers[conv2_name]["input"][0]]
                    input_node = [self.torch_layers[conv2_name]["input_id"][0]]
                    output_shape = self.torch_layers[conv2_name]["output"]
                    output_node = self.torch_layers[conv2_name]["output_id"]

                    conv2 = {
                        "operation": operation,
                        "shape_in": input_shape,
                        "shape_out": output_shape,
                        "node_in": input_node,
                        "node_out": output_node,
                        "kernel": self.torch_layers[conv2_name]["kernel"],
                        "bias": self.torch_layers[conv2_name]["bias"],
                        "padding": self.torch_layers[conv2_name]["padding"],
                        "stride": self.torch_layers[conv2_name]["stride"],
                        "groups": self.torch_layers[conv2_name]["groups"],
                        "dilation": self.torch_layers[conv2_name]["dilation"],
                        "branching": False,
                    }

                    sigmoid_name = se_module[4][1]
                    operation = self.torch_layers[sigmoid_name]["operation"]
                    input_shape = [self.torch_layers[sigmoid_name]["input"][0]]
                    input_node = [self.torch_layers[sigmoid_name]["input_id"][0]]
                    output_shape = self.torch_layers[sigmoid_name]["output"]
                    output_node = self.torch_layers[sigmoid_name]["output_id"]
                    se_branch_shape = output_shape

                    sigmoid = {
                        "operation": operation,
                        "shape_in": input_shape,
                        "shape_out": output_shape,
                        "node_in": input_node,
                        "node_out": output_node,
                        "branching": False,
                    }

                    mul_name = se_module[5][1]
                    operation = self.torch_layers[mul_name]["operation"]
                    input_shape = self.torch_layers[mul_name]["input"]
                    input_node = self.torch_layers[mul_name]["input_id"]
                    output_shape = self.torch_layers[mul_name]["output"]
                    output_node = self.torch_layers[mul_name]["output_id"]
                    se_output_shape = output_shape
                    se_output_node = output_node

                    mul = {
                        "operation": operation,
                        "shape_in": input_shape,
                        "shape_out": output_shape,
                        "node_in": input_node,
                        "node_out": output_node,
                        "branching": False,
                    }

                    name = "Se_" + se_module[0][1].split("_")[1]
                    operation = "SqueezeExcitation"
                    self.layers[name] = {
                        "operation": operation,
                        "shape_in": se_input_shape,
                        "shape_out": se_output_shape,
                        "node_in": se_input_node,
                        "node_out": se_output_node,
                        "shape_branch": se_branch_shape,
                        "branching": True,
                        "primitive_ops": {
                            gap_name: gap,
                            conv1_name: conv1,
                            relu_name: relu,
                            conv2_name: conv2,
                            sigmoid_name: sigmoid,
                            mul_name: mul,
                        },
                    }

                    del self.layers[gap_name]
                    del self.layers[conv1_name]
                    del self.layers[relu_name]
                    del self.layers[conv2_name]
                    del self.layers[sigmoid_name]
                    del self.layers[mul_name]
