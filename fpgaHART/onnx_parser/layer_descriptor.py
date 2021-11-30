from .onnx_parser import OnnxModelParser
from collections import deque
import logging

logging.basicConfig(level=logging.WARNING)

class ModelLayerDescriptor(OnnxModelParser):
    def __init__(self, model_name, breakdown_se):
        super().__init__(model_name)

        self.layers = {}
        self.breakdown_se = breakdown_se
        self.create_layers()

    def create_layers(self):
        if not self.breakdown_se:
            se_module = deque(maxlen=6)
        swish_module = deque(maxlen=2)
        prev_output_id = -1

        for k in self.torch_layers.keys():

            curr_output_id = int(self.torch_layers[k]['output_id'])
            if not prev_output_id == -1:
                assert curr_output_id >= prev_output_id + 1, "Modules are not in the correct order. Revise the graph creation"
            prev_output_id = curr_output_id
        
            name = k
            operation = self.torch_layers[k]['operation']
            input_shape = [self.torch_layers[k]['input'][0]]
            output_shape = self.torch_layers[k]['output']
            input_node = [self.torch_layers[k]['input_id'][0]]
            if name == 'Gemm_401':
                input_node = self.torch_layers['GlobalAveragePool_391']['output_id']
            output_node = self.torch_layers[k]['output_id']

            self.layers[name] = {"operation": operation,
                                 "shape_in": input_shape,
                                 "shape_out": output_shape,
                                 "node_in": input_node,
                                 "node_out": output_node,
                                 "branching": False}

            if operation == 'Conv':
                self.layers[name]['kernel'] = self.torch_layers[k]['kernel']
                self.layers[name]['bias'] = self.torch_layers[k]['bias']
                self.layers[name]['padding'] = self.torch_layers[k]['padding']
                self.layers[name]['stride'] = self.torch_layers[k]['stride']
                self.layers[name]['groups'] = self.torch_layers[k]['groups']
                self.layers[name]['dilation'] = self.torch_layers[k]['dilation']
            elif operation == 'BatchNormalization':
                self.layers[name]['kernel'] = self.torch_layers[k]['kernel']
                self.layers[name]['bias'] = self.torch_layers[k]['bias']
                self.layers[name]['running_mean'] = self.torch_layers[k]['running_mean']
                self.layers[name]['running_var'] = self.torch_layers[k]['running_var']
            elif operation == 'Gemm':
                self.layers[name]['kernel'] = self.torch_layers[k]['kernel']
                self.layers[name]['bias'] = self.torch_layers[k]['bias']

            if operation == 'Add' or operation == 'Mul' or operation == 'MatMul':
                self.layers[name]['shape_in'].append(self.torch_layers[k]['input'][1])
                self.layers[name]['node_in'].append(self.torch_layers[k]['input_id'][1])
                if self.model_name == 'x3d_m' and operation == 'MatMul':
                    self.layers[name]['kernel'] = self.torch_layers[k]['kernel']

            swish_module.append([operation, name])
            if swish_module[0][0] == 'Sigmoid' and swish_module[1][0] == 'Mul':
                mul_shapein_1 = self.torch_layers[swish_module[1][1]]['input'][0]
                mul_shapein_2 = self.torch_layers[swish_module[1][1]]['input'][1]
                if mul_shapein_1 == mul_shapein_2:
                  logging.debug("Creating Swish Activation Module")

                  sigmoid_name = swish_module[0][1]
                  operation = self.torch_layers[sigmoid_name]['operation']
                  input_shape = [self.torch_layers[sigmoid_name]['input'][0]]
                  input_node = [self.torch_layers[sigmoid_name]['input_id'][0]]
                  swish_input_shape = input_shape
                  swish_input_node = input_node
                  output_shape = self.torch_layers[sigmoid_name]['output']
                  output_node = self.torch_layers[sigmoid_name]['output_id']

                  sigmoid = {"operation": operation,
                             "shape_in": input_shape,
                             "shape_out": output_shape,
                             "node_in": input_node,
                             "node_out": output_node,
                             "branching": False}

                  mul_name = swish_module[1][1]
                  operation = self.torch_layers[mul_name]['operation']
                  input_shape = self.torch_layers[mul_name]['input']
                  input_node = self.torch_layers[mul_name]['input_id']
                  output_shape = self.torch_layers[mul_name]['output']
                  output_node = self.torch_layers[mul_name]['output_id']
                  swish_output_shape = output_shape
                  swish_output_node = output_node

                  mul = {"operation": operation,
                         "shape_in": input_shape,
                         "shape_out": output_shape,
                         "node_in": input_node,
                         "node_out": output_node,
                         "branching": False}

                  name = 'Swish_' + swish_module[0][1].split('_')[1]
                  operation = 'Swish'
                  self.layers[name] = {"operation": operation,
                                        "shape_in": swish_input_shape,
                                        "shape_out": swish_output_shape,
                                        "node_in": swish_input_node,
                                        "node_out": swish_output_node,
                                        "shape_branch": swish_input_shape,
                                        "branching": True,
                                        "primitive_ops": {
                                                sigmoid_name: sigmoid,
                                                mul_name: mul}
                                       }

                  del self.layers[sigmoid_name]
                  del self.layers[mul_name]

            if not self.breakdown_se:
                se_module.append([operation, name])
                if se_module[0][0] == 'GlobalAveragePool' and se_module[1][0] == 'Conv' and se_module[2][0] == 'Relu' and se_module[3][0] == 'Conv' and se_module[4][0] == 'Sigmoid' and se_module[5][0] == 'Mul':
                    logging.debug("Creating Squeeze and Excitation Module")

                    gap_name = se_module[0][1]
                    operation = self.torch_layers[gap_name]['operation']
                    input_shape = [self.torch_layers[gap_name]['input'][0]]
                    input_node = [self.torch_layers[gap_name]['input_id'][0]]
                    se_input_shape = input_shape
                    se_input_node = input_node
                    output_shape = self.torch_layers[gap_name]['output']
                    output_node = self.torch_layers[gap_name]['output_id']

                    gap = {"operation": operation,
                        "shape_in": input_shape,
                        "shape_out": output_shape,
                        "node_in": input_node,
                        "node_out": output_node,
                        "branching": False}

                    conv1_name = se_module[1][1]
                    operation = self.torch_layers[conv1_name]['operation']
                    input_shape = [self.torch_layers[conv1_name]['input'][0]]
                    input_node = [self.torch_layers[conv1_name]['input_id'][0]]
                    output_shape = self.torch_layers[conv1_name]['output']
                    output_node = self.torch_layers[conv1_name]['output_id']

                    conv1 = {"operation": operation,
                            "shape_in": input_shape,
                            "shape_out": output_shape,
                            "node_in": input_node,
                            "node_out": output_node,
                            "kernel": self.torch_layers[conv1_name]['kernel'],
                            "bias": self.torch_layers[conv1_name]['bias'],
                            "padding": self.torch_layers[conv1_name]['padding'],
                            "stride": self.torch_layers[conv1_name]['stride'],
                            "groups": self.torch_layers[conv1_name]['groups'],
                            "dilation": self.torch_layers[conv1_name]['dilation'],
                            "branching": False}

                    relu_name = se_module[2][1]
                    operation = self.torch_layers[relu_name]['operation']
                    input_shape = [self.torch_layers[relu_name]['input'][0]]
                    input_node = [self.torch_layers[relu_name]['input_id'][0]]
                    output_shape = self.torch_layers[relu_name]['output']
                    output_node = self.torch_layers[relu_name]['output_id']

                    relu = {"operation": operation,
                            "shape_in": input_shape,
                            "shape_out": output_shape,
                            "node_in": input_node,
                            "node_out": output_node,
                            "branching": False}

                    conv2_name = se_module[3][1]
                    operation = self.torch_layers[conv2_name]['operation']
                    input_shape = [self.torch_layers[conv2_name]['input'][0]]
                    input_node = [self.torch_layers[conv2_name]['input_id'][0]]
                    output_shape = self.torch_layers[conv2_name]['output']
                    output_node = self.torch_layers[conv2_name]['output_id']

                    conv2 = {"operation": operation,
                            "shape_in": input_shape,
                            "shape_out": output_shape,
                            "node_in": input_node,
                            "node_out": output_node,
                            "kernel": self.torch_layers[conv2_name]['kernel'],
                            "bias": self.torch_layers[conv2_name]['bias'],
                            "padding": self.torch_layers[conv2_name]['padding'],
                            "stride": self.torch_layers[conv2_name]['stride'],
                            "groups": self.torch_layers[conv2_name]['groups'],
                            "dilation": self.torch_layers[conv2_name]['dilation'],
                            "branching": False}

                    sigmoid_name = se_module[4][1]
                    operation = self.torch_layers[sigmoid_name]['operation']
                    input_shape = [self.torch_layers[sigmoid_name]['input'][0]]
                    input_node = [self.torch_layers[sigmoid_name]['input_id'][0]]
                    output_shape = self.torch_layers[sigmoid_name]['output']
                    output_node = self.torch_layers[sigmoid_name]['output_id']
                    se_branch_shape = output_shape

                    sigmoid = {"operation": operation,
                            "shape_in": input_shape,
                            "shape_out": output_shape,
                            "node_in": input_node,
                            "node_out": output_node,
                            "branching": False}

                    mul_name = se_module[5][1]
                    operation = self.torch_layers[mul_name]['operation']
                    input_shape = self.torch_layers[mul_name]['input']
                    input_node = self.torch_layers[mul_name]['input_id']
                    output_shape = self.torch_layers[mul_name]['output']
                    output_node = self.torch_layers[mul_name]['output_id']
                    se_output_shape = output_shape
                    se_output_node = output_node

                    mul = {"operation": operation,
                        "shape_in": input_shape,
                        "shape_out": output_shape,
                        "node_in": input_node,
                        "node_out": output_node,
                        "branching": False}

                    name = 'Se_' + se_module[0][1].split('_')[1]
                    operation = 'SqueezeExcitation'
                    self.layers[name] = {"operation": operation,
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
                                                    mul_name: mul}
                                        }

                    del self.layers[gap_name]
                    del self.layers[conv1_name]
                    del self.layers[relu_name]
                    del self.layers[conv2_name]
                    del self.layers[sigmoid_name]
                    del self.layers[mul_name]      