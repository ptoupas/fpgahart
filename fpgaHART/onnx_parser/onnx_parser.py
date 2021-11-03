import onnx
import onnx.numpy_helper
import os
import sys
import logging
import configparser
import onnxoptimizer as optimizer
import numpy as np

logging.basicConfig(level=logging.WARNING)

def add_input_from_initializer(model : onnx.ModelProto):
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

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
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

class OnnxModelParser():
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_path = os.path.join(os.getcwd(), 'models', self.model_name + '.onnx')

        self.torch_layers = {}

        self.onnx_model = onnx.load(self.model_path)
        onnx.checker.check_model(self.onnx_model)
        add_input_from_initializer(self.onnx_model)
        self.onnx_model = self.convert_matmul_to_gemm()
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

        self.fix_classifier_shapes()
        self.get_config()
        self.parse_layers()


    def fix_classifier_shapes(self):
        if self.model_name == 'x3d_m':
            for i in range(len(self.onnx_model.graph.value_info)):
                if self.onnx_model.graph.value_info[i].name == '968':
                    self.onnx_model.graph.value_info[i].CopyFrom(onnx.helper.make_tensor_value_info('968', onnx.TensorProto.FLOAT, [1, 432]))
                elif self.onnx_model.graph.value_info[i].name == '970':
                    self.onnx_model.graph.value_info[i].CopyFrom(onnx.helper.make_tensor_value_info('970', onnx.TensorProto.FLOAT, [1, 2048]))
                elif self.onnx_model.graph.value_info[i].name == '971':
                    self.onnx_model.graph.value_info[i].CopyFrom(onnx.helper.make_tensor_value_info('971', onnx.TensorProto.FLOAT, [1, 2048]))
            self.onnx_model.graph.value_info.append(self.onnx_model.graph.output[0])


    def get_config(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), 'fpgaHART', 'config', 'config_pytorch.ini'))
        self.supported_operations = config.get('Onnx Supported', 'layers').split(',')

    def get_tensor_shape(self, tensor_name, is_initializer=False):
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
        if (tensor_type.HasField("shape")):
            for d in tensor_type.shape.dim:
                if (d.HasField("dim_value")):
                    tensor_shape.append(d.dim_value)
                elif (d.HasField("dim_param")):
                    tensor_shape.append(d.dim_param)
                else:
                    logging.critical("Couldn't read the dimensions of tensor")
                    sys.exit() 
        return tensor_shape

    def parse_layers(self):
        input_shape = self.get_tensor_shape(self.onnx_model.graph.input[0].name)

        logging.info("Model input shape = {}".format(input_shape))
        # assert len(self.onnx_model.graph.input) == 1, "Model has multiple inputs or the initializers are duplicated to inputs as well. Aborting..."

        layers_outputs = {}
        isFirstLayer = True
        for n, v in zip(self.onnx_model.graph.node, self.onnx_model.graph.value_info):
            if n.op_type in self.supported_operations:
                if self.model_name == 'x3d_m':
                    if n.name == 'Gemm_401':
                        layers_outputs[n.input[0]] = [1,432]

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
                        
                if n.op_type == 'Conv':
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
                            kernel = self.get_tensor_shape(param_name, is_initializer=True)
                        if i_num == 2:
                            bias = self.get_tensor_shape(param_name, is_initializer=True)
                        if i_num == 3:
                            running_mean = self.get_tensor_shape(param_name, is_initializer=True)
                        if i_num == 4:
                            running_var = self.get_tensor_shape(param_name, is_initializer=True)

                elif n.op_type == 'BatchNormalization':
                    layer_input_ids.append(n.input[0])
                    layer_input_shapes.append(layers_outputs[n.input[0]])

                    for i_num, param_name in enumerate(n.input):
                        if i_num == 1:
                            kernel = self.get_tensor_shape(param_name, is_initializer=True)
                        if i_num == 2:
                            bias = self.get_tensor_shape(param_name, is_initializer=True)
                        if i_num == 3:
                            running_mean = self.get_tensor_shape(param_name, is_initializer=True)
                        if i_num == 4:
                            running_var = self.get_tensor_shape(param_name, is_initializer=True)

                elif n.op_type == 'GlobalAveragePool':
                    layer_input_ids.append(n.input[0])
                    layer_input_shapes.append(layers_outputs[n.input[0]])

                elif n.op_type == 'AveragePool' or n.op_type == 'MaxPool':
                    layer_input_ids.append(n.input[0])
                    layer_input_shapes.append(layers_outputs[n.input[0]])

                    for attr in n.attribute:
                        if attr.name == "pads":
                            padding = list(attr.ints[:3])
                        elif attr.name == "strides":
                            stride = list(attr.ints)

                elif n.op_type == 'Mul' or n.op_type == 'Add' or n.op_type == 'Div':
                    layer_input_ids.append(n.input[0])
                    layer_input_ids.append(n.input[1])
                    layer_input_shapes.append(layers_outputs[n.input[0]])
                    layer_input_shapes.append(layers_outputs[n.input[1]])

                elif n.op_type == 'Gemm':
                    layer_input_ids.append(n.input[0])
                    layer_input_shapes.append(layers_outputs[n.input[0]])

                    for i_num, param_name in enumerate(n.input):
                        if i_num == 1:
                            kernel = self.get_tensor_shape(param_name, is_initializer=True)
                        if i_num == 2:
                            bias = self.get_tensor_shape(param_name, is_initializer=True)
                        if i_num == 3:
                            running_mean = self.get_tensor_shape(param_name, is_initializer=True)
                        if i_num == 4:
                            running_var = self.get_tensor_shape(param_name, is_initializer=True)

                elif n.op_type == 'MatMul':
                    layer_input_ids.append(n.input[0])
                    layer_input_ids.append(n.input[1])
                    layer_input_shapes.append(layers_outputs[n.input[0]])
                    layer_input_shapes.append(layers_outputs[n.input[1]])

                    if self.model_name == 'x3d_m':
                        kernel = layers_outputs[n.input[1]]

                elif n.op_type == 'Relu' or n.op_type == 'Sigmoid' or n.op_type == 'Elu' or n.op_type == 'HardSigmoid' or n.op_type == 'LeakyRelu' or n.op_type == 'PRelu' or n.op_type == 'Selu' or n.op_type == 'Tanh' or n.op_type == 'Celu' or n.op_type == 'HardSwish' or n.op_type == 'Softmax':
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
                    "running_var": running_var,}

                logging.info("Node ({}) inputs: {} - outputs: {}".format(n.name, n.input, n.output))
            else:
                logging.info("Node ({}) of type {} is not currently supported".format(n.name, n.op_type))

    def get_model_initializer(self, name, to_tensor=True):
        for node in self.onnx_model.graph.initializer:
            if node.name == name: # exact match
                if to_tensor:
                    return onnx.numpy_helper.to_array(node)
                else:
                    return node

    def _format_name(self, name):
        return name.rstrip(":0").rstrip("_Y")

    def _name(self, node):
        #return _format_name( node.name if node.name else node.output[0] )
        return self._format_name( node.output[0] )

    def get_model_node(self, name):
        for node in self.onnx_model.graph.node:
            if self._name(node) == name: # formatted match
                return node

    def get_model_input(self, name):
        for node in self.onnx_model.graph.input:
            if node.name == name: # exact match
                return node

    def convert_matmul_to_gemm(self):
        # iterate over nodes in the graph
        for index, node in enumerate(self.onnx_model.graph.node):
            if node.op_type == "MatMul":
                # update the weights
                transpose_connection = False
                init = self.get_model_initializer(node.input[1], to_tensor=False)
                if init is None:
                    transpose_connection = True
                    input_node = self.get_model_node(node.input[1])
                    init = self.get_model_initializer(input_node.input[0], to_tensor=False)
                init_index = list(self.onnx_model.graph.initializer).index(init)
                weights = onnx.numpy_helper.to_array(init)
                weights = np.swapaxes(weights,0,1)
                if transpose_connection:
                    new_init = onnx.helper.make_tensor(
                        name=input_node.input[0],
                        data_type=init.data_type,
                        dims=weights.shape,
                        vals=weights.flatten().tolist())
                else:
                    new_init = onnx.helper.make_tensor(
                    name=node.input[1],
                    data_type=init.data_type,
                    dims=weights.shape,
                    vals=weights.flatten().tolist())
                # update weight's value info
                if transpose_connection:
                    init_value_info = self.get_model_input(input_node.input[0])
                else:
                    init_value_info = self.get_model_input(node.input[1])
                init_value_info_index = list(self.onnx_model.graph.input).index(init_value_info)
                if transpose_connection:
                    new_init_value_info = onnx.helper.make_tensor_value_info(
                            input_node.input[0],
                            onnx.TensorProto.FLOAT,
                            weights.shape)
                else:
                    new_init_value_info = onnx.helper.make_tensor_value_info(
                    node.input[1],
                    onnx.TensorProto.FLOAT,
                    weights.shape)
                # update the graph
                self.onnx_model.graph.initializer.remove(init)
                self.onnx_model.graph.initializer.insert(init_index,new_init)
                self.onnx_model.graph.input.remove(init_value_info)
                self.onnx_model.graph.input.insert(init_value_info_index, new_init_value_info)
                # add an empty bias term
                new_bias = onnx.helper.make_tensor(
                    name=".".join([input_node.input[0],"bias"]),
                    data_type=init.data_type,
                    dims=(weights.shape[1],),
                    vals=np.zeros(weights.shape[1]).flatten().tolist())
                new_bias_value_info = onnx.helper.make_tensor_value_info(
                        new_bias.name,
                        onnx.TensorProto.FLOAT,
                        [weights.shape[1]])
                # update the graph
                self.onnx_model.graph.initializer.insert(-1,new_bias)
                self.onnx_model.graph.input.insert(-1,new_bias_value_info)
                # create a new matmul node
                if transpose_connection:
                    new_node = onnx.helper.make_node(
                        "Gemm",
                        name="Gemm" + node.name.split("MatMul")[-1],
                        inputs=[node.input[0], input_node.input[0], ".".join([input_node.input[0],"bias"])],
                        outputs=node.output
                    )
                else:
                    new_node = onnx.helper.make_node(
                        "Gemm",
                        name="Gemm" + node.name.split("MatMul")[-1],
                        inputs=[*node.input, ".".join([input_node.input[0],"bias"])],
                        outputs=node.output
                    )
                # remove old node and add new one
                self.onnx_model.graph.node.remove(node)
                self.onnx_model.graph.node.insert(index, new_node)
                if transpose_connection:
                    self.onnx_model.graph.node.remove(input_node)
        # return the new model
        return self.onnx_model