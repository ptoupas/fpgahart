import os
import argparse
# from fpgaHART.onnx_parser.onnx_parser import OnnxModelParser
# from fpgaHART.onnx_parser.layer_descriptor import ModelLayerDescriptor
from fpgaHART.onnx_parser.partition_descriptor import PartitionDescriptor
from fpgaHART.layers.convolutional_3d import Convolutional3DLayer
from fpgaHART.layers.squeeze_excitation import SqueezeExcitationLayer
from fpgaHART.layers.gap import GAPLayer
from fpgaHART.layers.elemwise import ElementWiseLayer
from fpgaHART.utils import utils
from fpgaHART.partitions.partition_parser import PartitionParser
import itertools
import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from multiprocessing import Pool

sns.set(rc={'figure.figsize':(15,8)})
sns.set_style("whitegrid")

def parse_args():
    parser = argparse.ArgumentParser(description='fpgaHART toolflow parser')
    parser.add_argument('model_name', help='name of the HAR model')
    parser.add_argument('--singlethreaded', action='store_true', help='whether to use single thread solution or not')
    parser.add_argument('--se_block', action='store_true', help='whether to treat squeeze excitation as a block/layer or not')
    parser.add_argument('--plot_layers', action='store_true', help='whether to plot design points per layer or not')
    parser.add_argument('--gap_approx', action='store_true', help='whether to use historical data as approximation for GAP layers or not')

    args = parser.parse_args()
    return args

def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results

if __name__ == '__main__':
    args = parse_args()

    parser = PartitionParser(args.model_name, args.singlethreaded, args.plot_layers, args.se_block, args.gap_approx)
    
    # parser.model_custom_partition()
    # parser.model_individual_layers()
    parser.parse()