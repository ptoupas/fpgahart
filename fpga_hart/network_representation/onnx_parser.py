import configparser
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import onnx
import onnx.numpy_helper
import onnxoptimizer as optimizer
import onnxruntime as ort
from onnxsim import simplify

from fpga_hart import _logger


def add_input_from_initializer(model: onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph: onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.input}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.input.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    return add_const_value_infos_to_graph(model.graph)


@dataclass
class OnnxModelParser:
    model_name: str

    def __post_init__(self) -> None:
        # _logger.setLevel(level=logging.INFO)
        self.model_path = os.path.join(os.getcwd(), "models", self.model_name + ".onnx")
        self.optimized_model_path = os.path.join(os.getcwd(), "models", self.model_name + "_optimized.onnx")
        self.torch_layers = {}
        self.init_onnx_model()

    def init_onnx_model(self) -> None:
        self.onnx_model = onnx.load(self.model_path)
        self.initial_model_inputs = [node.name for node in self.onnx_model.graph.input]
        self.initial_model_outputs = [node.name for node in self.onnx_model.graph.output]

        onnx.checker.check_model(self.onnx_model)
        onnx.helper.strip_doc_string(self.onnx_model)
        self.onnx_model, check = simplify(self.onnx_model)
        assert check, "Simplified ONNX model could not be validated"

        add_input_from_initializer(self.onnx_model)
        self.convert_matmul_to_gemm()
        self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
        passes = [
            "extract_constant_to_initializer",
            "eliminate_unused_initializer",
            "eliminate_nop_transpose",
            "eliminate_nop_pad",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_bn_into_conv",
        ]
        self.onnx_model = optimizer.optimize(self.onnx_model, passes=passes)
        onnx.checker.check_model(self.onnx_model)

        self.onnx_model.graph.value_info.append(self.onnx_model.graph.output[0])
        onnx.save(
            self.onnx_model,
            self.optimized_model_path,
        )
        self.get_config()
        self.parse_layers()

    def get_node_weight_bias(self, node_name: str) -> Tuple[np.ndarray, np.ndarray]:
        node = [n for n in self.onnx_model.graph.node if n.name == node_name][0]
        if node.op_type not in ["Conv", "Gemm", "BatchNormalization"]:
            raise ValueError(f"Node {node_name} is not a Conv, Gemm or BatchNormalization node and has no weights")

        weight = None
        bias = None

        weight_initializer = node.input[1]
        weight_initializer_idx = [i for i, n in enumerate(self.onnx_model.graph.initializer) if n.name == weight_initializer][0]
        weight = onnx.numpy_helper.to_array(self.onnx_model.graph.initializer[weight_initializer_idx])
        bias_initializer = node.input[2] if len(node.input) > 2 else None
        if bias_initializer is not None:
            bias_initializer_idx = [i for i, n in enumerate(self.onnx_model.graph.initializer) if n.name == bias_initializer][0]
            bias = onnx.numpy_helper.to_array(self.onnx_model.graph.initializer[bias_initializer_idx])

        return weight, bias

    def add_outputs_to_model(self, outputs: list) -> None:
        output_tensors =[node.output[0] for node in self.onnx_model.graph.node if node.name in outputs]

        for out_t in output_tensors:
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = out_t
            self.onnx_model.graph.output.append(intermediate_layer_value_info)

    def onnx_forward(self, x: dict) -> Tuple[list, list]:
        assert len(self.initial_model_inputs) == 1, "Only one input supported in the onnx model"

        ort_sess = ort.InferenceSession(self.onnx_model.SerializeToString())
        output_nodes_names = [self.get_node_from_tensor_output(out.name).name for out in ort_sess.get_outputs()]

        outputs = ort_sess.run(None, x)

        return outputs, output_nodes_names

    def get_model_input_nodes(self) -> list:
        in_node_list = []
        for tensor in self.onnx_model.graph.input:
            if tensor.name in self.initial_model_inputs:
                in_node_list.append(self.get_node_from_tensor_input(tensor.name).name)
        return in_node_list

    def get_node_from_tensor_output(self, tensor_name: str) -> onnx.NodeProto:
        for node in self.onnx_model.graph.node:
            if tensor_name in node.output:
                return node
        raise ValueError(f"Tensor {tensor_name} not found in the model")

    def get_node_from_tensor_input(self, tensor_name: str) -> onnx.NodeProto:
        for node in self.onnx_model.graph.node:
            if tensor_name in node.input:
                return node
        raise ValueError(f"Tensor {tensor_name} not found in the model")

    def get_model_input_shapes(self) -> list:
        in_shapes_list = []
        for tensor in self.onnx_model.graph.input:
            if tensor.name in self.initial_model_inputs:
                in_shapes_list.append([dim.dim_value for dim in tensor.type.tensor_type.shape.dim])
        return in_shapes_list

    def get_prev_nodes_from_node(self, node_name) -> list:
        prev_nodes = []
        node = [n for n in self.onnx_model.graph.node if n.name == node_name][0]
        for tensor in node.input:
            if tensor in self.initial_model_inputs:
                continue
            if not tensor in [ninit.name for ninit in self.onnx_model.graph.initializer]:
                prev_nodes.append(self.get_node_from_tensor_output(tensor))
        return prev_nodes

    def get_next_nodes_from_node(self, node_name) -> list:
        next_nodes = []
        node = [n for n in self.onnx_model.graph.node if n.name == node_name][0]
        for tensor in node.output:
            if not tensor in [ninit.name for ninit in self.onnx_model.graph.initializer]:
                next_nodes.append(self.get_node_from_tensor_input(tensor))
        return next_nodes

    def get_config(self) -> None:
        config = configparser.ConfigParser()
        config.read(
            os.path.join(os.getcwd(), "fpga_hart", "config", "config_pytorch.ini")
        )
        self.supported_operations = config.get("Onnx Supported", "layers").split(",")

    def get_tensor_shape(self, tensor_name: str, is_initializer: bool = False) -> list:
        if is_initializer:
            for inp in self.onnx_model.graph.initializer:
                if tensor_name == inp.name:
                    tensor = inp
                    break
            return list(tensor.dims)
        else:
            for inp in self.onnx_model.graph.input:
                if tensor_name == inp.name:
                    tensor = inp
                    break

        tensor_type = tensor.type.tensor_type
        tensor_shape = []
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    tensor_shape.append(d.dim_value)
                elif d.HasField("dim_param"):
                    tensor_shape.append(d.dim_param)
                else:
                    _logger.critical("Couldn't read the dimensions of tensor")
                    sys.exit()
        return tensor_shape

    def parse_layers(self) -> None:
        input_shape = self.get_tensor_shape(self.onnx_model.graph.input[0].name)

        _logger.debug("Model input shape = {}".format(input_shape))
        # assert len(self.onnx_model.graph.input) == 1, "Model has multiple inputs or the initializers are duplicated to inputs as well. Aborting..."

        layers_outputs = {}
        isFirstLayer = True
        for n, v in zip(self.onnx_model.graph.node, self.onnx_model.graph.value_info):
            if n.op_type in self.supported_operations:
                if self.model_name == "x3d_m":
                    if n.name == "Gemm_399":
                        layers_outputs[n.input[0]] = [1, 432]
                if self.model_name == "slowonly":
                    if n.name == "Gemm_179":
                        layers_outputs[n.input[0]] = [1, 2048]
                if self.model_name == "r2plus1d":
                    if n.name == "Gemm_237":
                        layers_outputs[n.input[0]] = [1, 512]
                if self.model_name == "c3d":
                    if n.name == "Gemm_32":
                        layers_outputs[n.input[0]] = [1, 4096]

                layer_input_ids = []
                layer_input_shapes = []
                dilation = []
                groups = 0
                kernel = []
                bias = []
                running_mean = []
                running_var = []
                padding = []
                stride = []

                if n.op_type == "Conv":
                    layer_input_ids.append(n.input[0])
                    if isFirstLayer:
                        layer_input_shapes.append(input_shape)
                        isFirstLayer = False
                    else:
                        layer_input_shapes.append(layers_outputs[n.input[0]])

                    for attr in n.attribute:
                        if attr.name == "dilations":
                            dilation = list(attr.ints)
                        elif attr.name == "group":
                            groups = attr.i
                        elif attr.name == "pads":
                            padding = list(attr.ints[:3])
                        elif attr.name == "strides":
                            stride = list(attr.ints)

                    for i_num, param_name in enumerate(n.input):
                        if i_num == 1:
                            kernel = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )
                        if i_num == 2:
                            bias = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )
                        if i_num == 3:
                            running_mean = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )
                        if i_num == 4:
                            running_var = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )

                elif n.op_type == "BatchNormalization":
                    layer_input_ids.append(n.input[0])
                    layer_input_shapes.append(layers_outputs[n.input[0]])

                    for i_num, param_name in enumerate(n.input):
                        if i_num == 1:
                            kernel = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )
                        if i_num == 2:
                            bias = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )
                        if i_num == 3:
                            running_mean = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )
                        if i_num == 4:
                            running_var = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )

                elif n.op_type == "GlobalAveragePool":
                    layer_input_ids.append(n.input[0])
                    layer_input_shapes.append(layers_outputs[n.input[0]])

                elif n.op_type == "AveragePool" or n.op_type == "MaxPool":
                    layer_input_ids.append(n.input[0])
                    layer_input_shapes.append(layers_outputs[n.input[0]])

                    for attr in n.attribute:
                        if attr.name == "pads":
                            padding = list(attr.ints[:3])
                        elif attr.name == "strides":
                            stride = list(attr.ints)
                        elif attr.name == "kernel_shape":
                            kernel = list(attr.ints)

                elif n.op_type == "Mul" or n.op_type == "Add" or n.op_type == "Div":
                    layer_input_ids.append(n.input[0])
                    layer_input_ids.append(n.input[1])
                    layer_input_shapes.append(layers_outputs[n.input[0]])
                    layer_input_shapes.append(layers_outputs[n.input[1]])

                elif n.op_type == "Gemm":
                    layer_input_ids.append(n.input[0])
                    layer_input_shapes.append(layers_outputs[n.input[0]])

                    for i_num, param_name in enumerate(n.input):
                        if i_num == 1:
                            kernel = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )
                        if i_num == 2:
                            bias = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )
                        if i_num == 3:
                            running_mean = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )
                        if i_num == 4:
                            running_var = self.get_tensor_shape(
                                param_name, is_initializer=True
                            )

                elif n.op_type == "MatMul":
                    layer_input_ids.append(n.input[0])
                    layer_input_ids.append(n.input[1])
                    layer_input_shapes.append(layers_outputs[n.input[0]])
                    layer_input_shapes.append(layers_outputs[n.input[1]])

                    if self.model_name == "x3d_m" or self.model_name == "x3d_m_seq":
                        kernel = layers_outputs[n.input[1]]

                elif (
                    n.op_type == "Relu"
                    or n.op_type == "Sigmoid"
                    or n.op_type == "Elu"
                    or n.op_type == "HardSigmoid"
                    or n.op_type == "LeakyRelu"
                    or n.op_type == "PRelu"
                    or n.op_type == "Selu"
                    or n.op_type == "Tanh"
                    or n.op_type == "Celu"
                    or n.op_type == "HardSwish"
                    or n.op_type == "Softmax"
                ):
                    layer_input_ids.append(n.input[0])
                    layer_input_shapes.append(layers_outputs[n.input[0]])

                out_shape = []
                for dim in v.type.tensor_type.shape.dim:
                    out_shape.append(dim.dim_value)
                layers_outputs[n.output[0]] = out_shape

                self.torch_layers[n.name] = {
                    "operation": n.op_type,
                    "input": layer_input_shapes,
                    "input_id": layer_input_ids,
                    "output": out_shape,
                    "output_id": n.output[0],
                    "kernel": kernel,
                    "padding": padding,
                    "stride": stride,
                    "groups": groups,
                    "dilation": dilation,
                    "bias": bias,
                    "running_mean": running_mean,
                    "running_var": running_var,
                }

                _logger.debug(
                    "Node ({}) inputs: {} - outputs: {}".format(
                        n.name, n.input, n.output
                    )
                )
            else:
                _logger.debug(
                    "Node ({}) of type {} is not currently supported".format(
                        n.name, n.op_type
                    )
                )

    def get_model_initializer(
        self, name: str, to_tensor: bool = True
    ) -> Tuple[np.ndarray, onnx.TensorProto]:
        for node in self.onnx_model.graph.initializer:
            if node.name == name:  # exact match
                if to_tensor:
                    return onnx.numpy_helper.to_array(node)
                else:
                    return node

    def _format_name(self, name: str) -> str:
        return name.rstrip(":0").rstrip("_Y")

    def get_model_node(self, name: str) -> onnx.NodeProto:
        for node in self.onnx_model.graph.node:
            if node.output[0] == name:  # formatted match
                return node

    def get_model_input(self, name: str) -> onnx.ValueInfoProto:
        for node in self.onnx_model.graph.input:
            if node.name == name:  # exact match
                return node

    def convert_matmul_to_gemm(self) -> None:
        # iterate over nodes in the graph
        for index, node in enumerate(self.onnx_model.graph.node):
            if node.op_type == "MatMul":
                # update the weights
                transpose_connection = False
                init = self.get_model_initializer(node.input[1], to_tensor=False)
                if init is None:
                    transpose_connection = True
                    input_node = self.get_model_node(node.input[1])
                    init = self.get_model_initializer(
                        input_node.input[0], to_tensor=False
                    )
                init_index = list(self.onnx_model.graph.initializer).index(init)
                weights = onnx.numpy_helper.to_array(init)
                # weights = np.swapaxes(weights, 0, 1)
                if transpose_connection:
                    new_init = onnx.helper.make_tensor(
                        name=input_node.input[0],
                        data_type=init.data_type,
                        dims=weights.shape,
                        vals=weights.flatten().tolist(),
                    )
                else:
                    new_init = onnx.helper.make_tensor(
                        name=node.input[1],
                        data_type=init.data_type,
                        dims=weights.shape,
                        vals=weights.flatten().tolist(),
                    )
                # update weight's value info
                if transpose_connection:
                    init_value_info = self.get_model_input(input_node.input[0])
                else:
                    init_value_info = self.get_model_input(node.input[1])
                init_value_info_index = list(self.onnx_model.graph.input).index(
                    init_value_info
                )
                if transpose_connection:
                    new_init_value_info = onnx.helper.make_tensor_value_info(
                        input_node.input[0], onnx.TensorProto.FLOAT, weights.shape
                    )
                else:
                    new_init_value_info = onnx.helper.make_tensor_value_info(
                        node.input[1], onnx.TensorProto.FLOAT, weights.shape
                    )
                # update the graph
                self.onnx_model.graph.initializer.remove(init)
                self.onnx_model.graph.initializer.insert(init_index, new_init)
                self.onnx_model.graph.input.remove(init_value_info)
                self.onnx_model.graph.input.insert(
                    init_value_info_index, new_init_value_info
                )
                # add an empty bias term
                if transpose_connection:
                    new_bias = onnx.helper.make_tensor(
                        name=".".join([input_node.input[0], "bias"]),
                        data_type=init.data_type,
                        dims=(weights.shape[1],),
                        vals=np.zeros(weights.shape[1]).flatten().tolist(),
                    )
                else:
                    new_bias = onnx.helper.make_tensor(
                        name=".".join([node.input[1], "bias"]),
                        data_type=init.data_type,
                        dims=(weights.shape[1],),
                        vals=np.zeros(weights.shape[1]).flatten().tolist(),
                    )
                new_bias_value_info = onnx.helper.make_tensor_value_info(
                    new_bias.name, onnx.TensorProto.FLOAT, [weights.shape[1]]
                )
                # update the graph
                self.onnx_model.graph.initializer.insert(-1, new_bias)
                self.onnx_model.graph.input.insert(-1, new_bias_value_info)
                # create a new matmul node
                if transpose_connection:
                    new_node = onnx.helper.make_node(
                        "Gemm",
                        name="Gemm" + node.name.split("MatMul")[-1],
                        inputs=[
                            node.input[0],
                            input_node.input[0],
                            ".".join([input_node.input[0], "bias"]),
                        ],
                        outputs=node.output,
                    )
                else:
                    new_node = onnx.helper.make_node(
                        "Gemm",
                        name="Gemm" + node.name.split("MatMul")[-1],
                        inputs=[*node.input, ".".join([node.input[1], "bias"])],
                        outputs=node.output,
                    )
                # remove old node and add new one
                self.onnx_model.graph.node.remove(node)
                self.onnx_model.graph.node.insert(index, new_node)
                if transpose_connection:
                    self.onnx_model.graph.node.remove(input_node)
