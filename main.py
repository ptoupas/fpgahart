import argparse
from fpgaHART.onnx_parser.onnx_parser import OnnxModelParser

def parse_args():
    parser = argparse.ArgumentParser(description='fpgaHART toolflow parser')
    parser.add_argument('model_name', help='name of the HAR model')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    OnnxModelParser(args.model_name)