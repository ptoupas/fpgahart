from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.batchnorm_3d import BatchNorm3DLayer
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..layers.gap import GAPLayer
from ..layers.elemwise import ElementWiseLayer
from ..layers.activation import ActivationLayer
from ..layers.fully_connected import FCLayer
from ..optimizer.simulated_annealing import SimulatedAnnealing
from ..utils import utils
import itertools
from multiprocessing import Pool
import csv
import os
import numpy as np
import networkx as nx

def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results

def layer_compose(name, description, model_file, singlethreaded):
    if description['operation'] == 'Conv':
        return conv_compose(name, description, model_file, singlethreaded)
    elif description['operation'] == 'BatchNormalization':
        return batchnorm_compose(name, description, model_file, singlethreaded)
    elif description['operation'] == 'GlobalAveragePool':
        return gap_compose(name, description, model_file, singlethreaded)
    elif description['operation'] == 'Relu' or description['operation'] == 'Sigmoid' or description['operation'] == 'Swish':
        return activation_compose(name, description, model_file, singlethreaded)
    elif description['operation'] == 'SqueezeExcitation':
        return se_compose(name, description, model_file, singlethreaded)
    elif description['operation'] == 'Add' or description['operation'] == 'Mul':
        return elemwise_compose(name, description, model_file, singlethreaded)
    elif description['operation'] == 'Gemm' or description['operation'] == 'MatMul':
        return fc_compose(name, description, model_file, singlethreaded)
    else:
        assert False, "{} operation in layer {} is not supported".format(description['operation'], name)

def contains_list(part, whole):
    for p in part:
        for i in range(len(p)):
            if p[i] == whole[i]:
                continue
            else:
                return False
    if part:
        return True
    else:
        return False

def conv_compose(name, description, model_file, singlethreaded):
    conv = Convolutional3DLayer(description)
    
    if conv.depthwise:
        convtype = 'DepthWise'
    elif conv.pointwise:
        convtype = 'PointWise'
    else:
        convtype = '3DConv'
    
    kernel_size = conv.kernel_shape
    fine = utils.get_fine_feasible(kernel_size) / np.prod(np.array(kernel_size))
    coarsein = utils.get_factors(conv.channels) / np.int64(conv.channels)
    coarseout = utils.get_factors(conv.filters) / np.int64(conv.filters)

    # mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
    mem_bw = [(10000000, 10000000)]

    if convtype == 'PointWise':
        fine = [1]
    total = [fine, coarsein, coarseout, mem_bw]
    combinations = itertools.product(*total)
    
    print("Calculating {} design points for layer {} ({}).".format(len(fine)*len(coarsein)*len(coarseout)*len(mem_bw), name, convtype))

    throughput_gops = []
    throughput_vols = []
    latency = []
    dsp_util = []
    bram_util = []

    min_latency = 1000000000000
    best = {}
    with open(model_file, mode='a') as layer_dp:
        csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not singlethreaded:
            processes_pool = Pool(10)
            input_vars = []
            for (f, c1, c2, (bw_in, bw_out)) in combinations:
                input_vars.append([f, c1, c2, conv.mem_words_per_cycle*bw_in, conv.mem_words_per_cycle*bw_out])
            results = multithreaded_modeling(conv.get_design_point, input_vars, processes_pool)
            processes_pool.close()
            for r in results:
                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
        else:
            for (f, c1, c2, (bw_in, bw_out)) in combinations:
                r = conv.get_design_point(f, c1, c2, conv.mem_words_per_cycle*bw_in, conv.mem_words_per_cycle*bw_out)

                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
    
    print("Latency: {}.\n(fine={:.2f}->{}, cIn={:.2f}->{}, cOut={:.2f}->{}, bwIn={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSPs(%)={}({:.2f}), BRAMs(%)={}({:.2f}), RateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['latency(S)'], best['config'][0], best['config'][5], best['config'][1], best['config'][6], best['config'][2], best['config'][7], best['config'][3], best['config'][4], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP_RAW'], best['DSP'], best['BRAM_RAW'], best['BRAM'], best['rateIn'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn'], best['memBoundedOut']))
    print("*"*40)
    print("Searching for optimal point with simulated annealing for layer {} ({}).".format(name, convtype))
    graph = nx.DiGraph()
    graph.add_node(name, type=description['operation'], hw=conv)
    
    optimizer = SimulatedAnnealing(graph, branch_mem=0)
    optimizer.run_optimizer_layer(name)
    print("*"*40)

    return throughput_gops, throughput_vols, latency, dsp_util, bram_util

def batchnorm_compose(name, description, model_file, singlethreaded):
    bn = BatchNorm3DLayer(description)

    coarse_inout = utils.get_factors(bn.channels) / np.int64(bn.channels)

    # mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
    mem_bw = [(10000000, 10000000)]

    total = [coarse_inout, mem_bw]
    combinations = itertools.product(*total)
    
    print("Calculating {} design points for layer {}.".format(len(coarse_inout)*len(mem_bw), name))

    throughput_gops = []
    throughput_vols = []
    latency = []
    dsp_util = []
    bram_util = []

    min_latency = 1000000000000
    best = {}

    with open(model_file, mode='a') as layer_dp:
        csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not singlethreaded:
            processes_pool = Pool(10)
            input_vars = []
            for (cinout, (bw_in, bw_out)) in combinations:
                input_vars.append([cinout, bn.mem_words_per_cycle*bw_in, bn.mem_words_per_cycle*bw_out])
            results = multithreaded_modeling(bn.get_design_point, input_vars, processes_pool)
            processes_pool.close()
            for r in results:
                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
        else:
            for (cinout, (bw_in, bw_out)) in combinations:
                r = bn.get_design_point(cinout, bn.mem_words_per_cycle*bw_in, bn.mem_words_per_cycle*bw_out)

                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
    
    print("Latency: {}.\n(cinout={:.2f}, bwIn={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, RateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['latency(S)'], best['config'][0], best['config'][1], best['config'][2], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn'], best['memBoundedOut']))
    print("*"*40)
    print("Searching for optimal point with simulated annealing for layer {}.".format(name))
    graph = nx.DiGraph()
    graph.add_node(name, type=description['operation'], hw=bn)
    
    optimizer = SimulatedAnnealing(graph, branch_mem=0)
    optimizer.run_optimizer_layer(name)
    print("*"*40)

    return throughput_gops, throughput_vols, latency, dsp_util, bram_util

def gap_compose(name, description, model_file, singlethreaded):
    gap = GAPLayer(description)

    coarse_inout = utils.get_factors(gap.channels) / np.int64(gap.channels)

    # mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
    mem_bw = [(10000000, 10000000)]

    total = [coarse_inout, mem_bw]
    combinations = itertools.product(*total)
    
    print("Calculating {} design points for layer {}.".format(len(coarse_inout)*len(mem_bw), name))

    throughput_gops = []
    throughput_vols = []
    latency = []
    dsp_util = []
    bram_util = []

    min_latency = 1000000000000
    best = {}

    with open(model_file, mode='a') as layer_dp:
        csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not singlethreaded:
            processes_pool = Pool(10)
            input_vars = []
            for (cinout, (bw_in, bw_out)) in combinations:
                input_vars.append([cinout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out])
            results = multithreaded_modeling(gap.get_design_point, input_vars, processes_pool)
            processes_pool.close()
            for r in results:
                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
        else:
            for (cinout, (bw_in, bw_out)) in combinations:
                r = gap.get_design_point(cinout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out)

                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
    
    print("Latency: {}.\n(cinout={:.2f}, bwIn={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, RateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['latency(S)'], best['config'][0], best['config'][1], best['config'][2], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn'], best['memBoundedOut']))
    print("*"*40)
    print("Searching for optimal point with simulated annealing for layer {}.".format(name))
    graph = nx.DiGraph()
    graph.add_node(name, type=description['operation'], hw=gap)
    
    optimizer = SimulatedAnnealing(graph, branch_mem=0)
    optimizer.run_optimizer_layer(name)
    print("*"*40)

    return throughput_gops, throughput_vols, latency, dsp_util, bram_util

def activation_compose(name, description, model_file, singlethreaded):
    activ = ActivationLayer(description)

    coarse_inout = utils.get_factors(activ.channels) / np.int64(activ.channels)

    # mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
    mem_bw = [(10000000, 10000000)]

    total = [coarse_inout, mem_bw]
    combinations = itertools.product(*total)
    
    print("Calculating {} design points for layer {}.".format(len(coarse_inout)*len(mem_bw), name))

    throughput_gops = []
    throughput_vols = []
    latency = []
    dsp_util = []
    bram_util = []

    min_latency = 1000000000000
    best = {}

    with open(model_file, mode='a') as layer_dp:
        csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not singlethreaded:
            processes_pool = Pool(10)
            input_vars = []
            for (cinout, (bw_in, bw_out)) in combinations:
                input_vars.append([cinout, activ.mem_words_per_cycle*bw_in, activ.mem_words_per_cycle*bw_out])
            results = multithreaded_modeling(activ.get_design_point, input_vars, processes_pool)
            processes_pool.close()
            for r in results:
                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
        else:
            for (cinout, (bw_in, bw_out)) in combinations:
                r = activ.get_design_point(cinout, activ.mem_words_per_cycle*bw_in, activ.mem_words_per_cycle*bw_out)

                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
    
    print("Latency: {}.\n(cinout={:.2f}, bwIn={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, RateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['latency(S)'], best['config'][0], best['config'][1], best['config'][2], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn'], best['memBoundedOut']))
    print("*"*40)
    print("Searching for optimal point with simulated annealing for layer {}.".format(name))
    graph = nx.DiGraph()
    graph.add_node(name, type=description['operation'], hw=activ)
    
    optimizer = SimulatedAnnealing(graph, branch_mem=0)
    optimizer.run_optimizer_layer(name)
    print("*"*40)

    return throughput_gops, throughput_vols, latency, dsp_util, bram_util

def se_compose(name, description, model_file, singlethreaded):
    se = SqueezeExcitationLayer(description)

    # layers_in_shape = []
    # layers_out_shape = []
    # for layers in se.sequencial.values():
    #     if isinstance(layers, ElementWiseLayer):
    #         layers_in_shape.append(layers.input_shape_1)
    #     else:
    #         layers_in_shape.append(layers.input_shape)
    #     layers_out_shape.append(layers.output_shape)

    # gap_total_size = layers_in_shape[0][1]*layers_in_shape[0][2]*layers_in_shape[0][3]*layers_in_shape[0][4]
    # gap_coarsein = [1/gap_total_size, (layers_in_shape[0][1]/total_size)*0.1, (layers_in_shape[0][1]/total_size)*0.25, (layers_in_shape[0][1]/total_size)*0.5, layers_in_shape[0][1]/total_size, ((layers_in_shape[0][2] * layers_in_shape[0][1])/total_size)*0.1, ((layers_in_shape[0][2] * layers_in_shape[0][1])/total_size)*0.25, ((layers_in_shape[0][2] * layers_in_shape[0][1])/total_size)*0.5, (layers_in_shape[0][2] * layers_in_shape[0][1])/total_size, ((layers_in_shape[0][2] * layers_in_shape[0][1] * layers_in_shape[0][3])/total_size)*0.25]
    # gap_coarseout = [1]
    # coarsein_1 = [1/layers_in_shape[1][1], 0.25, 0.5, 1]
    # coarseout_1 = [1/layers_out_shape[1][1], 0.25, 0.5, 1]
    # coarsein_2 = [1/layers_in_shape[2][1], 0.25, 0.5, 1]
    # coarseout_2 = [1/layers_out_shape[2][1], 0.25, 0.5, 1]
    # mul_total_size = layers_out_shape[3][1] * layers_out_shape[3][2] * layers_out_shape[3][3] * layers_out_shape[3][4]
    # mul_coarseinout = [(layers_out_shape[3][1]/mul_total_size)*0.1, (layers_out_shape[3][1]/mul_total_size)*0.25, (layers_out_shape[3][1]/mul_total_size)*0.5, layers_out_shape[3][1]/mul_total_size, ((layers_out_shape[3][2] * layers_out_shape[3][1])/mul_total_size)*0.1, ((layers_out_shape[3][2] * layers_out_shape[3][1])/mul_total_size)*0.25, ((layers_out_shape[3][2] * layers_out_shape[3][1])/mul_total_size)*0.5, (layers_out_shape[3][2] * layers_out_shape[3][1])/mul_total_size, (layers_out_shape[3][3] * layers_out_shape[3][2] * layers_out_shape[3][1])/total_size]

    # fine_1 = [1]
    # fine_2 = [1]
    # # mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
    # mem_bw = [(10000000, 10000000)]

    # total = [gap_coarsein, gap_coarseout, fine_1, coarsein_1, coarseout_1, fine_2, coarsein_2, coarseout_2, mul_coarseinout, mem_bw]
    # combinations = itertools.product(*total)

    # print("Calculating {} design points for layer {}.".format(len(gap_coarsein)*len(gap_coarseout)*len(coarsein_1)*len(coarseout_1)*len(coarsein_2)*len(coarseout_2)*len(mul_coarseinout)*len(fine_1)*len(fine_2)*len(mem_bw), name))

    se_model_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', 'x3d_m_se_pareto.csv')

    config_points = {}
    for n, d in description['primitive_ops'].items():
        config_points[n] = utils.get_config_points(n, se_model_file)

    total = []
    for c in config_points.values():
        total.append(c)

    bw_in, bw_out = 10000000, 10000000
    combinations = itertools.product(*total)

    throughput_gops = []
    throughput_vols = []
    latency = []
    dsp_util = []
    bram_util = []

    min_latency = 1000000000000
    best = {}
    with open(model_file, mode='a') as layer_dp:
        csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not singlethreaded:
            processes_pool = Pool(10)
            input_vars = []
            # for (gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, (bw_in, bw_out)) in combinations:
            #     input_vars.append([gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, se.mem_words_per_cycle*bw_in, se.mem_words_per_cycle*bw_out])
            for (gap, conv1, relu, conv2, sigm, mul) in combinations:
                input_vars.append([gap[0], gap[1], conv1[0], conv1[1], conv1[2], relu[0], conv2[0], conv2[1], conv2[2], sigm[0], mul[0], mul[1], mul[2], se.mem_words_per_cycle*bw_in, se.mem_words_per_cycle*bw_out])

            results = multithreaded_modeling(se.get_design_point, input_vars, processes_pool)
            processes_pool.close()
            for r in results:
                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
        else:
            # for (gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, (bw_in, bw_out)) in combinations:
            #     r = se.get_design_point(gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, se.mem_words_per_cycle*bw_in, se.mem_words_per_cycle*bw_out)
            for (gap, conv1, relu, conv2, sigm, mul) in combinations:
                r = se.get_design_point(gap[0], gap[1], conv1[0], conv1[1], conv1[2], relu[0], conv2[0], conv2[1], conv2[2], sigm[0], mul[0], mul[1], mul[2], se.mem_words_per_cycle*bw_in, se.mem_words_per_cycle*bw_out)

                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r

    print("Latency: {}.\n(gapcin = {:.2f}, gapcout = {:.2f}, fine1={:.2f}, coarsein1={:.2f}, coarseout1={:.2f}, relucinout={:.2f}, fine2={:.2f}, coarsein2={:.2f}, coarseout2={:.2f}, sigmcinout={:.2f},  fmulcin1={:.6f}, fmulcin2={:.6f}, fmulcout={:.6f}, bwIn={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, RateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['latency(S)'], best['config'][0], best['config'][1], best['config'][2], best['config'][3], best['config'][4], best['config'][5], best['config'][6], best['config'][7], best['config'][8], best['config'][9], best['config'][10], best['config'][11], best['config'][12], best['config'][13], best['config'][14], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn'], best['memBoundedOut']))
    print("*"*40)

    return throughput_gops, throughput_vols, latency, dsp_util, bram_util
    
def elemwise_compose(name, description, model_file, singlethreaded):
    elem = ElementWiseLayer(description)

    coarse_inout = utils.get_factors(elem.channels_1) / np.int64(elem.channels_1)

    # mem_bw = [(0.1, 0.1, 0.8), (0.2, 0.2, 0.6), (0.3, 0.3, 0.4), (0.4, 0.4, 0.2), (0.3, 0.3, 0.4), (0.1, 0.4, 0.5), (0.4, 0.2, 0.4), (0.7, 0.2, 0.1), (0.8, 0.1, 0.1)]
    mem_bw = [(10000000, 10000000, 10000000)]

    total = [coarse_inout, mem_bw]
    combinations = itertools.product(*total)
    
    print("Calculating {} design points for layer {}.".format(len(coarse_inout)*len(mem_bw), name))

    throughput_gops = []
    throughput_vols = []
    latency = []
    dsp_util = []
    bram_util = []

    min_latency = 1000000000000
    best = {}

    with open(model_file, mode='a') as layer_dp:
        csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if not singlethreaded:
            processes_pool = Pool(10)
            input_vars = []
            for (cinout, (bw_in1, bw_in2, bw_out)) in combinations:
                input_vars.append([cinout, elem.mem_words_per_cycle*bw_in1, elem.mem_words_per_cycle*bw_in2, elem.mem_words_per_cycle*bw_out])
            results = multithreaded_modeling(elem.get_design_point, input_vars, processes_pool)
            processes_pool.close()
            for r in results:
                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
        else:
            for (cinout, (bw_in1, bw_in2, bw_out)) in combinations:
                r = elem.get_design_point(cinout, elem.mem_words_per_cycle*bw_in1, elem.mem_words_per_cycle*bw_in2, elem.mem_words_per_cycle*bw_out)

                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn'], r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn'], r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
    
    print("Latency: {}.\n(cinout={:.2f}, bwIn1={:.2f}, bwIn2={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={}({:.2f}), BRAM(%)={}({:.2f}), RateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['latency(S)'], best['config'][0], best['config'][1], best['config'][2], best['config'][3], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP_RAW'], best['DSP'], best['BRAM_RAW'], best['BRAM'], best['rateIn'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn'], best['memBoundedOut']))
    print("*"*40)
    print("Searching for optimal point with simulated annealing for layer {}.".format(name))
    graph = nx.DiGraph()
    graph.add_node(name, type=description['operation'], hw=elem)
    
    optimizer = SimulatedAnnealing(graph, branch_mem=0)
    optimizer.run_optimizer_layer(name)
    print("*"*40)

    return throughput_gops, throughput_vols, latency, dsp_util, bram_util

def fc_compose(name, description, model_file, singlethreaded):
    fc = FCLayer(description)
    
    graph = nx.DiGraph()
    graph.add_node(name, type=description['operation'], hw=fc)

    print("Searching for optimal point with simulated annealing for layer {}.".format(name))
    optimizer = SimulatedAnnealing(graph=graph, branch_mem=0)
    optimizer.run_optimizer_layer(name)
    print("*"*40)
    return [], [], [], [], []