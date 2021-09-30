import onnx
import os
import sys
import logging
import configparser

logging.basicConfig(level=logging.INFO)

class OnnxModelParser():
    def __init__(self, model_name):
        self.model_name = model_name + '.onnx'
        self.model_path = os.path.join(os.getcwd(), 'models', self.model_name)

        self.onnx_model = onnx.load(self.model_path)
        self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
        onnx.checker.check_model(self.onnx_model)

        self.get_config()
        self.parse_layers()

    def get_config(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), 'fpgaHART', 'onnx_parser', 'config.ini'))
        self.supported_operations = config.get('Onnx Supported', 'layers').split(',')

    def get_tensor_shape(self, input, is_initializer=False):
        if is_initializer:
            return list(input.dims)

        tensor_type = input.type.tensor_type
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
        input_shape = self.get_tensor_shape(self.onnx_model.graph.input[0])

        logging.info("Model input shape = {}".format(input_shape))
        assert len(self.onnx_model.graph.input) == 1, "Model has multiple inputs or the initializers are duplicated to inputs as well. Aborting..."

        for n, v in zip(self.onnx_model.graph.node, self.onnx_model.graph.value_info[:-1]):
            if n.op_type in self.supported_operations:
                out_shape = []
                for dim in v.type.tensor_type.shape.dim:
                    out_shape.append(dim.dim_value)
                logging.info("Node ({}) inputs: {} - outputs: {} -> {}".format(n.name, n.input, n.output, out_shape))
            else:
                logging.info("Node ({}) of type {} is not currently supported".format(n.name, n.op_type))