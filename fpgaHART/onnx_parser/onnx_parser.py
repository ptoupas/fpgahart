import onnx
import os
import sys
import logging
import configparser

logging.basicConfig(level=logging.WARNING)

class OnnxModelParser():
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_path = os.path.join(os.getcwd(), 'models', self.model_name + '.onnx')

        self.torch_layers = {}

        self.onnx_model = onnx.load(self.model_path)
        self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
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
        config.read(os.path.join(os.getcwd(), 'fpgaHART', 'onnx_parser', 'config.ini'))
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
        assert len(self.onnx_model.graph.input) == 1, "Model has multiple inputs or the initializers are duplicated to inputs as well. Aborting..."

        layers_outputs = {}
        isFirstLayer = True
        for n, v in zip(self.onnx_model.graph.node, self.onnx_model.graph.value_info):
            if n.op_type in self.supported_operations:
                if self.model_name == 'x3d_m':
                    if n.name == 'MatMul_401':
                        layers_outputs[n.input[0]] = [1,432]
                        layers_outputs[n.input[1]] = [432,2048]

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