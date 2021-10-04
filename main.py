import argparse
# from fpgaHART.onnx_parser.onnx_parser import OnnxModelParser
# from fpgaHART.onnx_parser.layer_descriptor import ModelLayerDescriptor
from fpgaHART.onnx_parser.partition_descriptor import PartitionDescriptor

def parse_args():
    parser = argparse.ArgumentParser(description='fpgaHART toolflow parser')
    parser.add_argument('model_name', help='name of the HAR model')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # OnnxModelParser(args.model_name)
    # ModelLayerDescriptor(args.model_name)
    PartitionDescriptor(args.model_name)