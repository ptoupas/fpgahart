import argparse
# from fpgaHART.onnx_parser.onnx_parser import OnnxModelParser
# from fpgaHART.onnx_parser.layer_descriptor import ModelLayerDescriptor
from fpgaHART.onnx_parser.partition_descriptor import PartitionDescriptor
from fpgaHART.layers.convolutional_3d import Convolutional3DLayer
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description='fpgaHART toolflow parser')
    parser.add_argument('model_name', help='name of the HAR model')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # OnnxModelParser(args.model_name)
    # ModelLayerDescriptor(args.model_name)
    partitions = PartitionDescriptor(args.model_name)

    for n,l in partitions.layers.items():
        if l['operation'] == 'Conv':
            conv = Convolutional3DLayer(l)
            if conv.depthwise:
                convtype = 'DepthWise'
            elif conv.pointwise:
                convtype = 'PointWise'
            else:
                convtype = '3DConv'
            print("{} ({}). In shape {}, Out shape {}, Kernel shape {}".format(n, convtype, conv.input_shape, conv.output_shape, conv.kernel_shape))
            coarsein = [1/conv.channels, 0.2, 0.8, 1]
            coarseout = [1/conv.filters, 0.3, 0.7, 1]
            total = [[1], coarseout]
            combinations = itertools.product(*total)
            
            for (c1, c2) in combinations:
                conv.get_design_point(1, c1, c2, conv.mem_words_per_cycle*0.4, conv.mem_words_per_cycle*0.6)
        elif l['operation'] == 'SqueezeExcitation':
            for n_se,l_se in l['primitive_ops'].items():
                if l_se['operation'] == 'Conv':
                    conv = Convolutional3DLayer(l_se)
                    if conv.depthwise:
                        convtype = 'DepthWise'
                    elif conv.pointwise:
                        convtype = 'PointWise'
                    else:
                        convtype = '3DConv'
                    print("{} ({}). In shape {}, Out shape {}, Kernel shape {}".format(n, convtype, conv.input_shape, conv.output_shape, conv.kernel_shape))

                    coarsein = [1/conv.channels, 0.2, 0.5, 0.8, 1]
                    coarseout = [1/conv.filters, 0.3, 0.6, 0.75, 1]
                    total = [[1], coarseout]
                    combinations = itertools.product(*total)
                    
                    for (c1, c2) in combinations:
                        conv.get_design_point(1, c1, c2, conv.mem_words_per_cycle*0.4, conv.mem_words_per_cycle*0.6)