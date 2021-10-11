import os
import argparse
# from fpgaHART.onnx_parser.onnx_parser import OnnxModelParser
# from fpgaHART.onnx_parser.layer_descriptor import ModelLayerDescriptor
from fpgaHART.onnx_parser.partition_descriptor import PartitionDescriptor
from fpgaHART.layers.convolutional_3d import Convolutional3DLayer
import itertools
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(15,8)})
sns.set_style("whitegrid")

def parse_args():
    parser = argparse.ArgumentParser(description='fpgaHART toolflow parser')
    parser.add_argument('model_name', help='name of the HAR model')
    parser.add_argument('--optimization', type=str, default='brute_force', choices=['brute_force', 'Powell'], help='Optimization Strategy')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # OnnxModelParser(args.model_name)
    # ModelLayerDescriptor(args.model_name)
    partitions = PartitionDescriptor(args.model_name)

    for n,l in partitions.layers.items():
        if l['operation'] == 'Conv':
            conv = Convolutional3DLayer(l, args.optimization)
            if conv.depthwise:
                convtype = 'DepthWise'
            elif conv.pointwise:
                convtype = 'PointWise'
            else:
                convtype = '3DConv'
            print("*"*40)
            print("{} ({}). In shape {}, Out shape {}, Kernel shape {}".format(n, convtype, conv.input_shape, conv.output_shape, conv.kernel_shape))
            
            if args.optimization == 'brute_force':
                fine = [0.25, 0.5, 0.75, 1]
                coarsein = [1/conv.channels, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
                coarseout = [1/conv.filters, 0.1, 0.3, 0.5, 0.7, 0.9, 0.1]
            else:
                fine = [1]
                coarsein = [1]
                coarseout = [1]
            mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
            total = [fine, coarsein, coarseout, mem_bw]
            combinations = itertools.product(*total)
            
            max_throughput = -1
            best = []
            for (f, c1, c2, (bw_in, bw_out)) in combinations:
                f_fine, f_fine_elems, f_coarseIn, f_coarseIn_ch, f_coarseOut, f_coarseOut_fil, mem_bw_in, mem_bw_out, dsps_util, max_parallel_muls, bram_util, latency_sec, latency_cycles, throughput_gops, thr_in, thr_out, depth, total_ops, mem_bounded_in, mem_bounded_out = conv.get_design_point(f, c1, c2, conv.mem_words_per_cycle*bw_in, conv.mem_words_per_cycle*bw_out)
                if throughput_gops > max_throughput and (dsps_util < 90. and bram_util < 90.):
                    max_throughput = throughput_gops
                    best = [f_fine, f_fine_elems, f_coarseIn, f_coarseIn_ch, f_coarseOut, f_coarseOut_fil, mem_bw_in, mem_bw_out, dsps_util, max_parallel_muls, bram_util, latency_sec, latency_cycles, throughput_gops, thr_in, thr_out, depth, total_ops, mem_bounded_in, mem_bounded_out]
        
            print("(fine={:.2f}({}), cIn={:.2f}({}), cOut={:.2f}({}), bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f} ({}), BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8], best[9], best[10], best[11], best[12], best[13], best[14], best[15], best[16], best[17], best[18], best[19]))
            print("*"*40)
        elif l['operation'] == 'SqueezeExcitation':
            for n_se,l_se in l['primitive_ops'].items():
                if l_se['operation'] == 'Conv':
                    conv = Convolutional3DLayer(l_se, args.optimization)
                    if conv.depthwise:
                        convtype = 'DepthWise'
                    elif conv.pointwise:
                        convtype = 'PointWise'
                    else:
                        convtype = '3DConv'
                    print("*"*40)
                    print("{} ({}). In shape {}, Out shape {}, Kernel shape {}".format(n, convtype, conv.input_shape, conv.output_shape, conv.kernel_shape))
                    
                    if args.optimization == 'brute_force':
                        fine = [0.25, 0.5, 0.75, 1]
                        coarsein = [1/conv.channels, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
                        coarseout = [1/conv.filters, 0.1, 0.3, 0.5, 0.7, 0.9, 0.1]
                    else:
                        fine = [1]
                        coarsein = [1]
                        coarseout = [1]
                    mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
                    total = [fine, coarsein, coarseout, mem_bw]
                    combinations = itertools.product(*total)

                    max_throughput = -1
                    best = []
                    for (f, c1, c2, (bw_in, bw_out)) in combinations:
                        f_fine, f_fine_elems, f_coarseIn, f_coarseIn_ch, f_coarseOut, f_coarseOut_fil, mem_bw_in, mem_bw_out, dsps_util, max_parallel_muls, bram_util, latency_sec, latency_cycles, throughput_gops, thr_in, thr_out, depth, total_ops, mem_bounded_in, mem_bounded_out = conv.get_design_point(f, c1, c2, conv.mem_words_per_cycle*bw_in, conv.mem_words_per_cycle*bw_out)
                        if throughput_gops > max_throughput and (dsps_util < 90. and bram_util < 90.):
                            max_throughput = throughput_gops
                            best = [f_fine, f_fine_elems, f_coarseIn, f_coarseIn_ch, f_coarseOut, f_coarseOut_fil, mem_bw_in, mem_bw_out, dsps_util, max_parallel_muls, bram_util, latency_sec, latency_cycles, throughput_gops, thr_in, thr_out, depth, total_ops, mem_bounded_in, mem_bounded_out]

                    print("(fine={:.2f}({}), cIn={:.2f}({}), cOut={:.2f}({}), bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f} ({}), BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8], best[9], best[10], best[11], best[12], best[13], best[14], best[15], best[16], best[17], best[18], best[19]))
                    print("*"*40)