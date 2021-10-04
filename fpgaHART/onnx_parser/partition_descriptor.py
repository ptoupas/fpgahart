from .layer_descriptor import ModelLayerDescriptor
from collections import deque
import logging

logging.basicConfig(level=logging.WARNING)

class PartitionDescriptor(ModelLayerDescriptor):
    def __init__(self, model_name):
        super().__init__(model_name)

        self.partitions = self.create_partitions(self.layers)

    def create_partitions(self, layers):
        final_layers = []

        if self.model_name == 'x3d_m':
            layer_type_1 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'SqueezeExcitation', 'Swish', 'Conv', 'BatchNormalization', 'Conv', 'BatchNormalization', 'Add']
            layer_type_2 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'SqueezeExcitation', 'Swish', 'Conv', 'BatchNormalization', 'Add']
            layer_type_3 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Swish', 'Conv', 'BatchNormalization', 'Add']
            layer_type_4 = ['Conv', 'Conv', 'BatchNormalization']
            layer_type_5 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'GlobalAveragePool', 'MatMul', 'Relu', 'Gemm']
            layer_queue = deque(maxlen=13)
            layer_queue_operations = deque(maxlen=13)
            for k in layers.keys():
                layer_queue_operations.append(layers[k]['operation'])
                layer_queue.append(k)
                if list(layer_queue_operations) == layer_type_1:
                    final_layers.append(list(layer_queue))
                if list(layer_queue_operations)[:-2] == layer_type_2:
                    final_layers.append(list(layer_queue)[:-2])
                if list(layer_queue_operations)[:-3] == layer_type_3:
                    final_layers.append(list(layer_queue)[:-3])
                if list(layer_queue_operations)[:-10] == layer_type_4:
                    final_layers.append(list(layer_queue)[:-10])
                if list(layer_queue_operations)[5:] == layer_type_5:
                    final_layers.append(list(layer_queue)[5:])
            return final_layers
        elif self.model_name == 'i3d':
            layer_type_1 = ['Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Conv', 'BatchNormalization', 'Add']
            layer_type_2 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Add']
            layer_queue = deque(maxlen=11)
            layer_queue_operations = deque(maxlen=11)
            for k in layers.keys():
                layer_queue_operations.append(layers[k]['operation'])
                layer_queue.append(k)
                if list(layer_queue_operations) == layer_type_1:
                    final_layers.append(list(layer_queue))
                if list(layer_queue_operations)[:-1] == layer_type_2:
                    final_layers.append(list(layer_queue)[:-1])
            return final_layers
          