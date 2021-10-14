import os
import argparse
# from fpgaHART.onnx_parser.onnx_parser import OnnxModelParser
# from fpgaHART.onnx_parser.layer_descriptor import ModelLayerDescriptor
from fpgaHART.onnx_parser.partition_descriptor import PartitionDescriptor
from fpgaHART.layers.convolutional_3d import Convolutional3DLayer
from fpgaHART.layers.squeeze_excitation import SqueezeExcitationLayer
from fpgaHART.layers.gap import GAPLayer
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
    parser.add_argument('--optimization', type=str, default='brute_force', choices=['brute_force', 'Powell', 'trust-constr'], help='Optimization Strategy')
    parser.add_argument('--singlethreaded', action='store_true', help='whether to use single thread solution or not')

    args = parser.parse_args()
    return args

def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results

if __name__ == '__main__':
    args = parse_args()

    # OnnxModelParser(args.model_name)
    # ModelLayerDescriptor(args.model_name)
    partitions = PartitionDescriptor(args.model_name)

    start = time.time()
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
            if not args.singlethreaded:
                processes_pool = Pool(10)
                input_vars = []
                for (f, c1, c2, (bw_in, bw_out)) in combinations:
                    input_vars.append([f, c1, c2, conv.mem_words_per_cycle*bw_in, conv.mem_words_per_cycle*bw_out])
                results = multithreaded_modeling(conv.get_design_point, input_vars, processes_pool)
                for r in results:
                    if r[13] > max_throughput and (r[8] < 90. and r[10] < 90.):
                        max_throughput = r[13]
                        best = list(r)
            else:
                for (f, c1, c2, (bw_in, bw_out)) in combinations:
                    params = conv.get_design_point(f, c1, c2, conv.mem_words_per_cycle*bw_in, conv.mem_words_per_cycle*bw_out)
                    if params[13] > max_throughput and (params[8] < 90. and params[10] < 90.):
                        max_throughput = params[13]
                        best = list(params)   

            print("(fine={:.2f}({}), cIn={:.2f}({}), cOut={:.2f}({}), bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f} ({}), BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8], best[9], best[10], best[11], best[12], best[13]*1e-9, best[14], best[15], best[16], best[17]*1e-9, best[18], best[19]))
            print("*"*40)
        elif l['operation'] == 'SqueezeExcitation':
            se = SqueezeExcitationLayer(l, args.optimization)
            print("{}. In shape {}, Out shape {}".format(n, se.input_shape, se.output_shape))

            layer_channels = []
            layer_filters = []
            for layers in se.sequencial.values():
                layer_channels.append(layers.channels)
                layer_filters.append(layers.filters)
            if args.optimization == 'brute_force':
                gap_coarseinout = [1/layer_channels[0], 0.2, 0.4, 0.5, 0.6, 0.8, 1]
                coarsein_1 = [1/layer_channels[1], 0.2, 0.4, 0.5, 0.6, 0.8, 1]
                coarseout_1 = [1/layer_filters[1], 0.1, 0.3, 0.5, 0.7, 0.9, 0.1]
                coarsein_2 = [1/layer_channels[2], 0.2, 0.4, 0.5, 0.6, 0.8, 1]
                coarseout_2 = [1/layer_filters[2], 0.1, 0.3, 0.5, 0.7, 0.9, 0.1]
            else:
                gap_coarseinout = [1]
                coarsein_1 = [1]
                coarseout_1 = [1]
                coarsein_2 = [1]
                coarseout_2 = [1]
            fine_1 = [1]
            fine_2 = [1]
            mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
            
            total = [gap_coarseinout, fine_1, coarsein_1, coarseout_1, fine_2, coarsein_2, coarseout_2, mem_bw]
            combinations = itertools.product(*total)

            max_throughput = -1
            best = []
            if not args.singlethreaded:
                processes_pool = Pool(10)
                input_vars = []
                for (cinout, f1, c11, c21, f2, c12, c22, (bw_in, bw_out)) in combinations:
                    input_vars.append([cinout, f1, c11, c21, f2, c12, c22, se.mem_words_per_cycle*bw_in, se.mem_words_per_cycle*bw_out])
                results = multithreaded_modeling(se.get_design_point, input_vars, processes_pool)
                for r in results:
                    if r[13] > max_throughput and (r[9] < 90. and r[10] < 90.):
                        max_throughput = r[13]
                        best = list(r)
            else:
                for (cinout, f1, c11, c21, f2, c12, c22, (bw_in, bw_out)) in combinations:
                    params = se.get_design_point(cinout, f1, c11, c21, f2, c12, c22, se.mem_words_per_cycle*bw_in, se.mem_words_per_cycle*bw_out)
                    if params[13] > max_throughput and (params[9] < 90. and params[10] < 90.):
                        max_throughput = params[13]
                        best = list(params)
            print("(cinout = {:.2f}, fine1={:.2f}, coarsein1={:.2f}, coarseout1={:.2f}, fine2={:.2f}, coarsein2={:.2f}, coarseout2={:.2f}, bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f}, BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8], best[9], best[10], best[11], best[12], best[13]*1e-9, best[14], best[15], best[16], best[17]*1e-9, best[18], best[19]))
            print("*"*40)

            for n_se,l_se in l['primitive_ops'].items():
                if l_se['operation'] == 'Conv':
                    continue
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
                        f_fine, f_fine_elems, f_coarseIn, f_coarseIn_ch, f_coarseOut, f_coarseOut_fil, mem_bw_in, mem_bw_out, dsps_util, max_parallel_muls, bram_util, latency_sec, latency_cycles, throughput_ops, thr_in, thr_out, depth, total_ops, mem_bounded_in, mem_bounded_out = conv.get_design_point(f, c1, c2, conv.mem_words_per_cycle*bw_in, conv.mem_words_per_cycle*bw_out)
                        if throughput_ops > max_throughput and (dsps_util < 90. and bram_util < 90.):
                            max_throughput = throughput_ops
                            best = [f_fine, f_fine_elems, f_coarseIn, f_coarseIn_ch, f_coarseOut, f_coarseOut_fil, mem_bw_in, mem_bw_out, dsps_util, max_parallel_muls, bram_util, latency_sec, latency_cycles, throughput_ops, thr_in, thr_out, depth, total_ops, mem_bounded_in, mem_bounded_out]
                    
                    print("(fine={:.2f}({}), cIn={:.2f}({}), cOut={:.2f}({}), bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f} ({}), BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8], best[9], best[10], best[11], best[12], best[13]*1e-9, best[14], best[15], best[16], best[17]*1e-9, best[18], best[19]))
                    print("*"*40)
                elif l_se['operation'] == 'GlobalAveragePool':
                    continue
                    gap = GAPLayer(l_se, args.optimization)

                    print("*"*40)
                    print("{}. In shape {}, Out shape {}".format(n_se, gap.input_shape, gap.output_shape))
                    coarseinout = [1/gap.channels, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
                    mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
                    total = [coarseinout, mem_bw]
                    combinations = itertools.product(*total)
                    
                    min_latency = 10000000000
                    best = []
                    if not args.singlethreaded:
                        processes_pool = Pool(10)
                        input_vars = []
                        for (cinout, (bw_in, bw_out)) in combinations:
                            input_vars.append([cinout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out])
                        results = multithreaded_modeling(gap.get_design_point, input_vars, processes_pool)
                        for r in results:
                            if r[7] < min_latency and (r[3] < 90. and r[5] < 90.):
                                min_latency = r[7]
                                best = list(r)
                    else:
                        for (cinout, (bw_in, bw_out)) in combinations:
                            params = gap.get_design_point(cinout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out)
                            if params[7] < min_latency and (params[3] < 90. and params[5] < 90.):
                                min_latency = params[7]
                                best = list(params)

                    print("(coarseinout={:.2f}, bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f}, BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[5], best[6], best[7], best[8]*1e-9, best[9], best[10], best[11], best[12]*1e-9, best[13], best[14]))
                    print("*"*40)
        elif l['operation'] == 'GlobalAveragePool':
            continue
            gap = GAPLayer(l, args.optimization)

            print("*"*40)
            print("{}. In shape {}, Out shape {}".format(n, gap.input_shape, gap.output_shape))
            coarseinout = [1/gap.channels, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
            mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
            total = [coarseinout, mem_bw]
            combinations = itertools.product(*total)
            
            min_latency = 10000000000
            best = []
            if not args.singlethreaded:
                processes_pool = Pool(10)
                input_vars = []
                for (cinout, (bw_in, bw_out)) in combinations:
                    input_vars.append([cinout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out])
                results = multithreaded_modeling(gap.get_design_point, input_vars, processes_pool)
                for r in results:
                    if r[7] < min_latency and (r[3] < 90. and r[5] < 90.):
                        min_latency = r[7]
                        best = list(r)
            else:
                for (cinout, (bw_in, bw_out)) in combinations:
                    params = gap.get_design_point(cinout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out)
                    if params[7] < min_latency and (params[3] < 90. and params[5] < 90.):
                        min_latency = params[7]
                        best = list(params)

            print("(coarseinout={:.2f}, bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f}, BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[5], best[6], best[7], best[8]*1e-9, best[9], best[10], best[11], best[12]*1e-9, best[13], best[14]))
            print("*"*40)
            
    end = time.time()
    print("Execution time: {:.3f}".format(end-start))
