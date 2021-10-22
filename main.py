import os
import argparse
# from fpgaHART.onnx_parser.onnx_parser import OnnxModelParser
# from fpgaHART.onnx_parser.layer_descriptor import ModelLayerDescriptor
from fpgaHART.onnx_parser.partition_descriptor import PartitionDescriptor
from fpgaHART.layers.convolutional_3d import Convolutional3DLayer
from fpgaHART.layers.squeeze_excitation import SqueezeExcitationLayer
from fpgaHART.layers.gap import GAPLayer
from fpgaHART.primitive_blocks.elemwise import ElementWiseLayer
from fpgaHART.utils import utils
from fpgaHART.partitions.partition_compose import PartitionComposer
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
    parser.add_argument('--plot_layers', action='store_true', help='whether to plot design points per layer or not')

    args = parser.parse_args()
    return args

def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results

if __name__ == '__main__':
    args = parse_args()

    composer = PartitionComposer(args.model_name, args.optimization, args.singlethreaded, args.plot_layers)
    composer.parser()
    exit()
    # OnnxModelParser(args.model_name)
    # ModelLayerDescriptor(args.model_name)
    partitions = PartitionDescriptor(args.model_name)

    start = time.time()
    for n,l in partitions.layers.items():
        if l['operation'] == 'Conv':
            # continue
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
            
            calculate_pareto = True
            throughput_gops = []
            throughput_vols = []
            latency = []
            dsp_util = []
            bram_util = []

            min_latency = 1000000000000
            best = []
            if not args.singlethreaded:
                processes_pool = Pool(4)
                input_vars = []
                for (f, c1, c2, (bw_in, bw_out)) in combinations:
                    input_vars.append([f, c1, c2, conv.mem_words_per_cycle*bw_in, conv.mem_words_per_cycle*bw_out])
                results = multithreaded_modeling(conv.get_design_point, input_vars, processes_pool)
                processes_pool.close()
                for r in results:
                    throughput_gops.append(r[13]*1e-9)
                    throughput_vols.append(r[15])
                    latency.append(r[12])
                    dsp_util.append(r[8])
                    bram_util.append(r[10])
                    if r[12] < min_latency and (r[8] < 90. and r[10] < 90.):
                        min_latency = r[12]
                        best = list(r)
                    elif r[12] == min_latency and (r[8] < 90. and r[10] < 90.):
                        if r[8] < best[8]:
                            min_latency = r[8]
                            best = list(r)
            else:
                for (f, c1, c2, (bw_in, bw_out)) in combinations:
                    params = conv.get_design_point(f, c1, c2, conv.mem_words_per_cycle*bw_in, conv.mem_words_per_cycle*bw_out)
                    throughput_gops.append(params[13]*1e-9)
                    throughput_vols.append(params[15])
                    latency.append(params[12])
                    dsp_util.append(params[8])
                    bram_util.append(params[10])
                    if params[12] < min_latency and (params[8] < 90. and params[10] < 90.):
                        min_latency = params[12]
                        best = list(params)
                    elif params[12] == min_latency and (params[8] < 90. and params[10] < 90.):
                        if params[8] < best[8]:
                            min_latency = params[8]
                            best = list(params) 

            utils.plot_graph(throughput_gops, throughput_vols, latency, dsp_util, bram_util, n, args.model_name, calculate_pareto)
            print("(fine={:.2f}({}), cIn={:.2f}({}), cOut={:.2f}({}), bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f} ({}), BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8], best[9], best[10], best[11], best[12], best[13]*1e-9, best[14], best[15], best[16], best[17]*1e-9, best[18], best[19]))
            print("*"*40)

        elif l['operation'] == 'SqueezeExcitation':
            continue
            se = SqueezeExcitationLayer(l, args.optimization)
            print("{}. In shape {}, Out shape {}".format(n, se.input_shape, se.output_shape))

            layers_in_shape = []
            layers_out_shape = []
            for layers in se.sequencial.values():
                if isinstance(layers, ElementWiseLayer):
                    layers_in_shape.append(layers.input_shape_1)
                else:
                    layers_in_shape.append(layers.input_shape)
                layers_out_shape.append(layers.output_shape)

            if args.optimization == 'brute_force':
                gap_coarsein = [1/(layers_in_shape[0][1]*layers_in_shape[0][2]*layers_in_shape[0][3]*layers_in_shape[0][4]), 1/(layers_in_shape[0][1]*layers_in_shape[0][2]*layers_in_shape[0][3]), 1/(layers_in_shape[0][1]*layers_in_shape[0][2]), 1/(layers_in_shape[0][1]), 1]
                gap_coarseout = [0.5, 1] #[1/layers_out_shape[0][1], 0.5, 1]
                coarsein_1 = [1/layers_in_shape[1][1], 0.25, 0.5, 1]
                coarseout_1 = [1/layers_out_shape[1][1], 0.25, 0.5, 1]
                coarsein_2 = [1/layers_in_shape[2][1], 0.25, 0.5, 1]
                coarseout_2 = [1/layers_out_shape[2][1], 0.25, 0.5, 1]
                mul_total_size = layers_out_shape[3][1] * layers_out_shape[3][2] * layers_out_shape[3][3] * layers_out_shape[3][4]
                mul_coarseinout = [(layers_out_shape[3][1]/mul_total_size)*0.1, (layers_out_shape[3][1]/mul_total_size)*0.25, (layers_out_shape[3][1]/mul_total_size)*0.5, layers_out_shape[3][1]/mul_total_size, ((layers_out_shape[3][2] * layers_out_shape[3][1])/mul_total_size)*0.05, ((layers_out_shape[3][2] * layers_out_shape[3][1])/mul_total_size)*0.1, (layers_out_shape[3][2] * layers_out_shape[3][1])/mul_total_size]
            else:
                gap_coarsein = [1]
                gap_coarseout = [1]
                coarsein_1 = [1]
                coarseout_1 = [1]
                coarsein_2 = [1]
                coarseout_2 = [1]
                mul_coarseinout = [1]

            fine_1 = [1]
            fine_2 = [1]
            mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
            
            total = [gap_coarsein, gap_coarseout, fine_1, coarsein_1, coarseout_1, fine_2, coarsein_2, coarseout_2, mul_coarseinout, mem_bw]
            combinations = itertools.product(*total)

            calculate_pareto = False
            throughput_gops = []
            throughput_vols = []
            latency = []
            dsp_util = []
            bram_util = []

            min_latency = 1000000000000
            best = []
            if not args.singlethreaded:
                processes_pool = Pool(4)
                input_vars = []
                for (gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, (bw_in, bw_out)) in combinations:
                    input_vars.append([gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, se.mem_words_per_cycle*bw_in, se.mem_words_per_cycle*bw_out])
                results = multithreaded_modeling(se.get_design_point, input_vars, processes_pool)
                processes_pool.close()
                for r in results:
                    throughput_gops.append(r[15]*1e-9)
                    throughput_vols.append(r[17])
                    latency.append(r[14])
                    dsp_util.append(r[11])
                    bram_util.append(r[12])
                    if r[14] < min_latency and (r[11] < 90. and r[12] < 90.):
                        min_latency = r[14]
                        best = list(r)
            else:
                for (gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, (bw_in, bw_out)) in combinations:
                    params = se.get_design_point(gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, se.mem_words_per_cycle*bw_in, se.mem_words_per_cycle*bw_out)
                    throughput_gops.append(params[15]*1e-9)
                    throughput_vols.append(params[17])
                    latency.append(params[14])
                    dsp_util.append(params[11])
                    bram_util.append(params[12])
                    if params[14] < min_latency and (params[11] < 90. and params[12] < 90.):
                        min_latency = params[14]
                        best = list(params)

            # utils.plot_graph(throughput_gops, throughput_vols, latency, dsp_util, bram_util, n, args.model_name, calculate_pareto)
            print("(gapcin = {:.2f}, gapcout = {:.2f}, fine1={:.2f}, coarsein1={:.2f}, coarseout1={:.2f}, fine2={:.2f}, coarsein2={:.2f}, coarseout2={:.2f}, fmul={:.10f}, bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f}, BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8], best[9], best[10], best[11], best[12], best[13], best[14], best[15]*1e-9, best[16], best[17], best[18], best[19]*1e-9, best[20], best[21]))
            print("*"*40)
            # exit()

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
                    print("{} ({}). In shape {}, Out shape {}, Kernel shape {}".format(n_se, convtype, conv.input_shape, conv.output_shape, conv.kernel_shape))
                    
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

                    calculate_pareto = True
                    throughput_gops = []
                    throughput_vols = []
                    latency = []
                    dsp_util = []
                    bram_util = []

                    min_latency = 1000000000000
                    best = []

                    if not args.singlethreaded:
                        processes_pool = Pool(4)
                        input_vars = []
                        for (f, c1, c2, (bw_in, bw_out)) in combinations:
                            input_vars.append([f, c1, c2, conv.mem_words_per_cycle*bw_in, conv.mem_words_per_cycle*bw_out])
                        results = multithreaded_modeling(conv.get_design_point, input_vars, processes_pool)
                        processes_pool.close()
                        for r in results:
                            throughput_gops.append(r[13]*1e-9)
                            throughput_vols.append(r[15])
                            latency.append(r[12])
                            dsp_util.append(r[8])
                            bram_util.append(r[10])
                            if r[12] < min_latency and (r[8] < 90. and r[10] < 90.):
                                min_latency = r[12]
                                best = list(r)
                            elif r[12] == min_latency and (r[8] < 90. and r[10] < 90.):
                                if r[8] < best[8]:
                                    min_latency = r[8]
                                    best = list(r)
                    else:
                        for (f, c1, c2, (bw_in, bw_out)) in combinations:
                            params = conv.get_design_point(f, c1, c2, conv.mem_words_per_cycle*bw_in, conv.mem_words_per_cycle*bw_out)
                            throughput_gops.append(params[13]*1e-9)
                            throughput_vols.append(params[15])
                            latency.append(params[12])
                            dsp_util.append(params[8])
                            bram_util.append(params[10])
                            if params[12] < min_latency and (params[8] < 90. and params[10] < 90.):
                                min_latency = params[12]
                                best = list(params)
                            elif params[12] == min_latency and (params[8] < 90. and params[10] < 90.):
                                if params[8] < best[8]:
                                    min_latency = params[8]
                                    best = list(params) 
                    
                    utils.plot_graph(throughput_gops, throughput_vols, latency, dsp_util, bram_util, n_se, args.model_name, calculate_pareto)
                    print("(fine={:.2f}({}), cIn={:.2f}({}), cOut={:.2f}({}), bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f} ({}), BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8], best[9], best[10], best[11], best[12], best[13]*1e-9, best[14], best[15], best[16], best[17]*1e-9, best[18], best[19]))
                    print("*"*40)

                elif l_se['operation'] == 'GlobalAveragePool':
                    continue
                    gap = GAPLayer(l_se, args.optimization)

                    print("*"*40)
                    print("{} (GAP). In shape {}, Out shape {}".format(n_se, gap.input_shape, gap.output_shape))

                    if args.optimization == 'brute_force':
                        # factor range [1, channels_1*rows_1*columns_1*depth_1] -> [1/(channels_1*rows_1*columns_1*depth_1), 1]
                        coarse_in = [1/(gap.channels*gap.depth_in*gap.rows_in*gap.cols_in), 1/(gap.channels*gap.depth_in*gap.rows_in), 1/(gap.channels*gap.depth_in), 1/(gap.channels), 1]
                        coarse_out = [1/gap.filters, 0.5, 1]
                    else:
                        coarse_in = [1]
                        coarse_out = [1]
                    mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
                    # mem_bw = [(10000000000, 10000000000)]
                    total = [coarse_in, coarse_out, mem_bw]
                    combinations = itertools.product(*total)
                    
                    calculate_pareto = True
                    throughput_gops = []
                    throughput_vols = []
                    latency = []
                    dsp_util = []
                    bram_util = []

                    min_latency = 1000000000000
                    best = []
                    if not args.singlethreaded:
                        processes_pool = Pool(4)
                        input_vars = []
                        for (cin, cout, (bw_in, bw_out)) in combinations:
                            input_vars.append([cin, cout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out])
                        results = multithreaded_modeling(gap.get_design_point, input_vars, processes_pool)
                        processes_pool.close()
                        for r in results:
                            throughput_gops.append(r[9]*1e-9)
                            throughput_vols.append(r[11])
                            latency.append(r[8])
                            dsp_util.append(r[4])
                            bram_util.append(r[6])
                            if r[8] < min_latency and (r[4] < 90. and r[6] < 90.):
                                min_latency = r[8]
                                best = list(r)
                            elif r[8] == min_latency and (r[4] < 90. and r[6] < 90.):
                                if r[4] < best[4]:
                                    min_latency = r[8]
                                    best = list(r) 
                    else:
                        for (cin, cout, (bw_in, bw_out)) in combinations:
                            params = gap.get_design_point(cin, cout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out)
                            throughput_gops.append(params[9]*1e-9)
                            throughput_vols.append(params[11])
                            latency.append(params[8])
                            dsp_util.append(params[4])
                            bram_util.append(params[6])
                            if params[8] < min_latency and (params[4] < 90. and params[6] < 90.):
                                min_latency = params[8]
                                best = list(params)
                            elif params[8] == min_latency and (params[4] < 90. and params[6] < 90.):
                                if params[4] < best[4]:
                                    min_latency = params[8]
                                    best = list(params) 

                    utils.plot_graph(throughput_gops, throughput_vols, latency, dsp_util, bram_util, n_se, args.model_name, calculate_pareto)
                    print("(coarsein={:.10f}, coarseout={:.2f}, bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f}, BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[6], best[7], best[8], best[9]*1e-9, best[10], best[11], best[12], best[13]*1e-9, best[14], best[15]))
                    print("*"*40)

                elif l_se['operation'] == 'Mul':
                    continue
                    elemwise = ElementWiseLayer(l_se, args.optimization)

                    print("*"*40)
                    print("{} ({}). In shape 1 {}, In shape 2 {}, Out shape {}, broadcasting = {}".format(n_se, l_se['operation'], elemwise.input_shape_1, elemwise.input_shape_2, elemwise.output_shape, elemwise.broadcasting))

                    if args.optimization == 'brute_force':
                        _, channels, depth, rows, cols = elemwise.input_shape_1
                        total_size = channels * depth * rows * cols
                        coarseinout = [(channels/total_size)*0.25, (channels/total_size)*0.5, channels/total_size, ((depth * channels)/total_size)*0.05, ((depth * channels)/total_size)*0.1, ((depth * channels)/total_size)*0.2, ((depth * channels)/total_size)*0.3]
                        # coarseinout = [(channels/total_size)*0.25, (channels/total_size)*0.5, channels/total_size, ((depth * channels)/total_size)*0.05, ((depth * channels)/total_size)*0.1, ((depth * channels)/total_size)*0.5, (depth * channels)/total_size, ((depth * channels * rows)/total_size)*0.5, (depth * channels * rows)/total_size, 0.5, 1]
                    else:
                        coarseinout = [1]
                    mem_bw = [(0.05,0.05,0.9), (0.1,0.1,0.8), (0.2,0.1,0.7), (0.2,0.2,0.6), (0.3,0.2,0.5), (0.4,0.1,0.4), (0.5,0.2,0.3), (0.1,0.7,0.2), (0.7,0.2,0.1), (0.4,0.4,0.2), (0.85,0.05,0.1)]
                    total = [coarseinout, mem_bw]
                    combinations = itertools.product(*total)

                    calculate_pareto = True
                    throughput_gops = []
                    throughput_vols = []
                    latency = []
                    dsp_util = []
                    bram_util = []

                    min_latency = 10000000000
                    best = []
                    if not args.singlethreaded:
                        processes_pool = Pool(4)
                        input_vars = []
                        for (cinout, (bw_in_1, bw_in_2, bw_out)) in combinations:
                            input_vars.append([cinout, elemwise.mem_words_per_cycle*bw_in_1, elemwise.mem_words_per_cycle*bw_in_2, elemwise.mem_words_per_cycle*bw_out])
                        results = multithreaded_modeling(elemwise.get_design_point, input_vars, processes_pool)
                        processes_pool.close()
                        for r in results:
                            throughput_gops.append(r[9]*1e-9)
                            throughput_vols.append(r[11])
                            latency.append(r[8])
                            dsp_util.append(r[4])
                            bram_util.append(r[6])
                            if r[8] < min_latency and (r[4] < 90. and r[6] < 90.):
                                min_latency = r[8]
                                best = list(r)
                    else:
                        for (cinout, (bw_in_1, bw_in_2, bw_out)) in combinations:
                            params = elemwise.get_design_point(cinout, elemwise.mem_words_per_cycle*bw_in_1, elemwise.mem_words_per_cycle*bw_in_2, elemwise.mem_words_per_cycle*bw_out)
                            throughput_gops.append(params[9]*1e-9)
                            throughput_vols.append(params[11])
                            latency.append(params[8])
                            dsp_util.append(params[4])
                            bram_util.append(params[6])
                            if params[8] < min_latency and (params[4] < 90. and params[6] < 90.):
                                min_latency = params[8]
                                best = list(params)

                    utils.plot_graph(throughput_gops, throughput_vols, latency, dsp_util, bram_util, n_se, args.model_name, calculate_pareto)
                    print("(coarseinout={:}, bwIn1={:.2f}, bwIn2={:.2f}, bwOut={:.2f}) DSP % = {:.2f}, BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In 1={}, Mem Bound In 2={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[6], best[7], best[8], best[9]*1e-9, best[10], best[11], best[12], best[13]*1e-9, best[14], best[15], best[16]))
                    print("*"*40)

        elif l['operation'] == 'GlobalAveragePool':
            continue
            gap = GAPLayer(l, args.optimization)

            print("*"*40)
            print("{}. In shape {}, Out shape {}".format(n, gap.input_shape, gap.output_shape))
            if args.optimization == 'brute_force':
                # factor range [1, channels_1*rows_1*columns_1*depth_1] -> [1/(channels_1*rows_1*columns_1*depth_1), 1]
                coarse_in = [1/(gap.channels*gap.depth_in*gap.rows_in*gap.cols_in), 1/(gap.channels*gap.depth_in*gap.rows_in), 1/(gap.channels*gap.depth_in), 1/(gap.channels), 1]
                coarse_out = [1/gap.filters, 0.5, 1]
            else:
                coarse_in = [1]
                coarse_out = [1]
            mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
            # mem_bw = [(10000000000, 10000000000)]
            total = [coarse_in, coarse_out, mem_bw]
            combinations = itertools.product(*total)
            
            calculate_pareto = True
            throughput_gops = []
            throughput_vols = []
            latency = []
            dsp_util = []
            bram_util = []

            min_latency = 10000000000
            best = []
            if not args.singlethreaded:
                processes_pool = Pool(4)
                input_vars = []
                for (cin, cout, (bw_in, bw_out)) in combinations:
                    input_vars.append([cin, cout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out])
                results = multithreaded_modeling(gap.get_design_point, input_vars, processes_pool)
                processes_pool.close()
                for r in results:
                    throughput_gops.append(r[9]*1e-9)
                    throughput_vols.append(r[11])
                    latency.append(r[8])
                    dsp_util.append(r[4])
                    bram_util.append(r[6])
                    if r[8] < min_latency and (r[4] < 90. and r[6] < 90.):
                        min_latency = r[8]
                        best = list(r)
                    elif r[8] == min_latency and (r[4] < 90. and r[6] < 90.):
                        if r[4] < best[4]:
                            min_latency = r[8]
                            best = list(r) 
            else:
                for (cin, cout, (bw_in, bw_out)) in combinations:
                    params = gap.get_design_point(cin, cout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out)
                    throughput_gops.append(params[9]*1e-9)
                    throughput_vols.append(params[11])
                    latency.append(params[8])
                    dsp_util.append(params[4])
                    bram_util.append(params[6])
                    if params[8] < min_latency and (params[4] < 90. and params[6] < 90.):
                        min_latency = params[8]
                        best = list(params)
                    elif params[8] == min_latency and (params[4] < 90. and params[6] < 90.):
                        if params[4] < best[4]:
                            min_latency = params[8]
                            best = list(params)

            utils.plot_graph(throughput_gops, throughput_vols, latency, dsp_util, bram_util, n, args.model_name, calculate_pareto)
            print("(coarsein={:.2f}, coarseout={:.2f}, bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f}, BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[6], best[7], best[8], best[9]*1e-9, best[10], best[11], best[12], best[13]*1e-9, best[14], best[15]))
            print("*"*40)

        elif l['operation'] == 'Mul' or l['operation'] == 'Add' or l['operation'] == 'Div':
            continue
            elemwise = ElementWiseLayer(l, args.optimization)

            print("*"*40)
            print("{} ({}). In shape 1 {}, In shape 2 {}, Out shape {}, broadcasting = {}".format(n, l['operation'], elemwise.input_shape_1, elemwise.input_shape_2, elemwise.output_shape, elemwise.broadcasting))

            if args.optimization == 'brute_force':
                _, channels, depth, rows, cols = elemwise.input_shape_1
                total_size = channels * depth * rows * cols
                coarseinout = [(channels/total_size)*0.5, channels/total_size, ((depth * channels)/total_size)*0.5, (depth * channels)/total_size, ((depth * channels * rows)/total_size)*0.5, (depth * channels * rows)/total_size, 0.5, 1]
            else:
                coarseinout = [1]
            mem_bw = [(0.05,0.05,0.9), (0.1,0.1,0.8), (0.2,0.1,0.7), (0.2,0.2,0.6), (0.3,0.2,0.5), (0.4,0.1,0.4), (0.5,0.2,0.3), (0.1,0.7,0.2), (0.7,0.2,0.1)]
            total = [coarseinout, mem_bw]
            combinations = itertools.product(*total)
            
            calculate_pareto = True
            throughput_gops = []
            throughput_vols = []
            latency = []
            dsp_util = []
            bram_util = []

            min_latency = 10000000000
            best = []
            if not args.singlethreaded:
                processes_pool = Pool(4)
                input_vars = []
                for (cinout, (bw_in_1, bw_in_2, bw_out)) in combinations:
                    input_vars.append([cinout, elemwise.mem_words_per_cycle*bw_in_1, elemwise.mem_words_per_cycle*bw_in_2, elemwise.mem_words_per_cycle*bw_out])
                results = multithreaded_modeling(elemwise.get_design_point, input_vars, processes_pool)
                processes_pool.close()
                for r in results:
                    throughput_gops.append(r[9]*1e-9)
                    throughput_vols.append(r[11])
                    latency.append(r[8])
                    dsp_util.append(r[4])
                    bram_util.append(r[6])
                    if r[8] < min_latency and (r[4] < 90. and r[6] < 90.):
                        min_latency = r[8]
                        best = list(r)
            else:
                for (cinout, (bw_in_1, bw_in_2, bw_out)) in combinations:
                    params = elemwise.get_design_point(cinout, elemwise.mem_words_per_cycle*bw_in_1, elemwise.mem_words_per_cycle*bw_in_2, elemwise.mem_words_per_cycle*bw_out)
                    throughput_gops.append(params[9]*1e-9)
                    throughput_vols.append(params[11])
                    latency.append(params[8])
                    dsp_util.append(params[4])
                    bram_util.append(params[6])
                    if params[8] < min_latency and (params[4] < 90. and params[6] < 90.):
                        min_latency = params[8]
                        best = list(params)

            utils.plot_graph(throughput_gops, throughput_vols, latency, dsp_util, bram_util, n, args.model_name, calculate_pareto)
            print("(coarseinout={:}, bwIn1={:.2f}, bwIn2={:.2f}, bwOut={:.2f}) DSP % = {:.2f}, BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In 1={}, Mem Bound In 2={}, Mem Bound Out={}".format(best[0], best[1], best[2], best[3], best[4], best[6], best[7], best[8], best[9]*1e-9, best[10], best[11], best[12], best[13]*1e-9, best[14], best[15], best[16]))
            print("*"*40)

    end = time.time()
    print("Execution time: {:.3f}".format(end-start))
