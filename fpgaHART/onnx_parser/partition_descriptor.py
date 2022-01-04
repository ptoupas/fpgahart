from .layer_descriptor import ModelLayerDescriptor
from collections import deque
import logging

logging.basicConfig(level=logging.WARNING)

class PartitionDescriptor(ModelLayerDescriptor):
    def __init__(self, model_name, se_block):
        super().__init__(model_name, se_block)

        self.partitions = self.create_partitions(self.layers)

    def create_partitions(self, layers):
        final_layers = []

        if self.model_name == 'x3d_m':
            if not self.se_block:
                layer_type_1 = ['Relu', 'Conv', 'Relu', 'Conv', 'GlobalAveragePool', 'Conv', 'Relu', 'Conv', 'Sigmoid', 'Mul', 'Swish', 'Conv', 'Conv', 'Add']
                layer_type_2 = ['Relu', 'Conv', 'Relu', 'Conv', 'GlobalAveragePool', 'Conv', 'Relu', 'Conv', 'Sigmoid', 'Mul', 'Swish', 'Conv', 'Add']
                layer_type_3 = ['Relu', 'Conv', 'Relu', 'Conv', 'Swish', 'Conv', 'Add']
                layer_type_4 = ['Conv', 'Conv', 'Relu', 'Conv', 'Relu', 'Conv', 'GlobalAveragePool', 'Conv', 'Relu', 'Conv', 'Sigmoid', 'Mul', 'Swish', 'Conv', 'Conv', 'Add']
                layer_type_5 = ['Relu', 'Conv', 'Relu', 'GlobalAveragePool', 'Gemm', 'Relu', 'Gemm']
                layer_queue = deque(maxlen=16)
                layer_queue_operations = deque(maxlen=16)
                for k in layers.keys():
                    layer_queue_operations.append(layers[k]['operation'])
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
                layer_type_1 = ['Relu', 'Conv', 'Relu', 'Conv', 'SqueezeExcitation', 'Swish', 'Conv', 'Conv', 'Add']
                layer_type_2 = ['Relu', 'Conv', 'Relu', 'Conv', 'SqueezeExcitation', 'Swish', 'Conv', 'Add']
                layer_type_3 = ['Relu', 'Conv', 'Relu', 'Conv', 'Swish', 'Conv', 'Add']
                layer_type_4 = ['Conv', 'Conv']
                layer_type_5 = ['Relu', 'Conv', 'Relu', 'GlobalAveragePool', 'Gemm', 'Relu', 'Gemm']
                layer_queue = deque(maxlen=9)
                layer_queue_operations = deque(maxlen=9)
                for k in layers.keys():
                    layer_queue_operations.append(layers[k]['operation'])
                    layer_queue.append(k)
                    if list(layer_queue_operations) == layer_type_1:
                        final_layers.append(list(layer_queue))
                    if list(layer_queue_operations)[:-1] == layer_type_2:
                        final_layers.append(list(layer_queue)[:-1])
                    if list(layer_queue_operations)[:-2] == layer_type_3:
                        final_layers.append(list(layer_queue)[:-2])
                    if list(layer_queue_operations)[:-7] == layer_type_4 and 'Conv_0' in list(layer_queue)[:-7]:
                        final_layers.append(list(layer_queue)[:-7])
                    if list(layer_queue_operations)[2:] == layer_type_5:
                        final_layers.append(list(layer_queue)[2:])
            return final_layers
        elif self.model_name == 'i3d':
            layer_type_1 = ['Conv', 'Relu', 'Conv', 'Relu', 'Conv', 'Conv', 'Add']
            layer_type_2 = ['Relu', 'Conv', 'Relu', 'Conv', 'Relu', 'Conv', 'Add']
            layer_queue = deque(maxlen=7)
            layer_queue_operations = deque(maxlen=7)
            for k in layers.keys():
                layer_queue_operations.append(layers[k]['operation'])
                layer_queue.append(k)
                if list(layer_queue_operations) == layer_type_1:
                    final_layers.append(list(layer_queue))
                if list(layer_queue_operations) == layer_type_2:
                    final_layers.append(list(layer_queue))
            return final_layers
          