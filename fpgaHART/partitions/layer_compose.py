from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.batchnorm_3d import BatchNorm3DLayer
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..layers.gap import GAPLayer
from ..layers.elemwise import ElementWiseLayer
from ..layers.activation import ActivationLayer
import itertools
from multiprocessing import Pool
import csv

def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results

def layer_compose(name, description, model_file, optimization, singlethreaded):
    if description['operation'] == 'Conv':
        return conv_compose(name, description, model_file, optimization, singlethreaded)
    elif description['operation'] == 'BatchNormalization':
        return batchnorm_compose(name, description, model_file, optimization, singlethreaded)
    elif description['operation'] == 'GlobalAveragePool':
        return gap_compose(name, description, model_file, optimization, singlethreaded)
    elif description['operation'] == 'Relu' or description['operation'] == 'Sigmoid' or description['operation'] == 'Swish':
        return activation_compose(name, description, model_file, optimization, singlethreaded)
    elif description['operation'] == 'SqueezeExcitation':
        return se_compose(name, description, model_file, optimization, singlethreaded)
    elif description['operation'] == 'Add' or description['operation'] == 'Mul':
        return elemwise_compose(name, description, model_file, optimization, singlethreaded)
    elif description['operation'] == 'Gemm' or description['operation'] == 'MatMul':
        return fc_compose(name, description, model_file, optimization, singlethreaded)
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

def conv_compose(name, description, model_file, optimization, singlethreaded):
    conv = Convolutional3DLayer(description, optimization)
    
    if conv.depthwise:
        convtype = 'DepthWise'
    elif conv.pointwise:
        convtype = 'PointWise'
    else:
        convtype = '3DConv'
    
    if optimization == 'brute_force':
        fine = [0.25, 0.5, 0.75, 1]
        coarsein = [1/conv.channels, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        coarseout = [1/conv.filters, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    else:
        fine = [1]
        coarsein = [1]
        coarseout = [1]
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
            processes_pool = Pool(8)
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

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

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

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
    
    print("(fine={:.2f}, cIn={:.2f}, cOut={:.2f}, bwIn={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn1={:.2f}, RateOut={:.2f}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['config'][0], best['config'][1], best['config'][2], best['config'][3], best['config'][4], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn1'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn1'], best['memBoundedOut']))
    print("*"*40)

    return throughput_gops, throughput_vols, latency, dsp_util, bram_util

def batchnorm_compose(name, description, model_file, optimization, singlethreaded):
    bn = BatchNorm3DLayer(description, optimization)

    if optimization == 'brute_force':
        total_size = bn.channels*bn.depth_in*bn.rows_in*bn.cols_in
        coarse_inout = [1/total_size, (bn.channels/total_size)*0.1, (bn.channels/total_size)*0.25, (bn.channels/total_size)*0.5, bn.channels/total_size, ((bn.depth_in * bn.channels)/total_size)*0.1, ((bn.depth_in * bn.channels)/total_size)*0.25, ((bn.depth_in * bn.channels)/total_size)*0.5, (bn.depth_in * bn.channels)/total_size, (bn.depth_in * bn.channels * bn.rows_in)/total_size]
    else:
        coarse_inout = [1]
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
            processes_pool = Pool(8)
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

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

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

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
    
    print("(cinout={:.2f}, bwIn={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn1={:.2f}, RateOut={:.2f}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['config'][0], best['config'][1], best['config'][2], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn1'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn1'], best['memBoundedOut']))
    print("*"*40)

    return throughput_gops, throughput_vols, latency, dsp_util, bram_util

def gap_compose(name, description, model_file, optimization, singlethreaded):
    gap = GAPLayer(description, optimization)

    if optimization == 'brute_force':
        total_size = gap.channels*gap.depth_in*gap.rows_in*gap.cols_in
        coarse_in = [1/total_size, (gap.channels/total_size)*0.1, (gap.channels/total_size)*0.25, (gap.channels/total_size)*0.5, gap.channels/total_size, ((gap.depth_in * gap.channels)/total_size)*0.1, ((gap.depth_in * gap.channels)/total_size)*0.25, ((gap.depth_in * gap.channels)/total_size)*0.5, (gap.depth_in * gap.channels)/total_size, ((gap.depth_in * gap.channels * gap.rows_in)/total_size)*0.25]
        coarse_out = [1]
    else:
        coarse_in = [1]
        coarse_out = [1]
    # mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
    mem_bw = [(10000000, 10000000)]

    total = [coarse_in, coarse_out, mem_bw]
    combinations = itertools.product(*total)
    
    print("Calculating {} design points for layer {}.".format(len(coarse_in)*len(coarse_out)*len(mem_bw), name))

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
            processes_pool = Pool(8)
            input_vars = []
            for (cin, cout, (bw_in, bw_out)) in combinations:
                input_vars.append([cin, cout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out])
            results = multithreaded_modeling(gap.get_design_point, input_vars, processes_pool)
            processes_pool.close()
            for r in results:
                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
        else:
            for (cin, cout, (bw_in, bw_out)) in combinations:
                r = gap.get_design_point(cin, cout, gap.mem_words_per_cycle*bw_in, gap.mem_words_per_cycle*bw_out)

                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
    
    print("(cin={:.2f}, cout={:.2f}, bwIn={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn1={:.2f}, RateOut={:.2f}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['config'][0], best['config'][1], best['config'][2], best['config'][3], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn1'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn1'], best['memBoundedOut']))
    print("*"*40)
    return throughput_gops, throughput_vols, latency, dsp_util, bram_util

def activation_compose(name, description, model_file, optimization, singlethreaded):
    activ = ActivationLayer(description, optimization)

    if optimization == 'brute_force':
        total_size = activ.channels*activ.depth_in*activ.rows_in*activ.cols_in
        coarse_inout = [1/total_size, (activ.channels/total_size)*0.1, (activ.channels/total_size)*0.25, (activ.channels/total_size)*0.5, activ.channels/total_size, ((activ.depth_in * activ.channels)/total_size)*0.1, ((activ.depth_in * activ.channels)/total_size)*0.25, ((activ.depth_in * activ.channels)/total_size)*0.5, (activ.depth_in * activ.channels)/total_size, (activ.depth_in * activ.channels * activ.rows_in)/total_size]
    else:
        coarse_inout = [1]
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
            processes_pool = Pool(8)
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

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

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

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
    
    print("(cinout={:.2f}, bwIn={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn1={:.2f}, RateOut={:.2f}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['config'][0], best['config'][1], best['config'][2], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn1'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn1'], best['memBoundedOut']))
    print("*"*40)

    return throughput_gops, throughput_vols, latency, dsp_util, bram_util

def se_compose(name, description, model_file, optimization, singlethreaded):
    se = SqueezeExcitationLayer(description, optimization)

    layers_in_shape = []
    layers_out_shape = []
    for layers in se.sequencial.values():
        if isinstance(layers, ElementWiseLayer):
            layers_in_shape.append(layers.input_shape_1)
        else:
            layers_in_shape.append(layers.input_shape)
        layers_out_shape.append(layers.output_shape)

    if optimization == 'brute_force':
        gap_total_size = layers_in_shape[0][1]*layers_in_shape[0][2]*layers_in_shape[0][3]*layers_in_shape[0][4]
        gap_coarsein = [1/gap_total_size, (layers_in_shape[0][1]/total_size)*0.1, (layers_in_shape[0][1]/total_size)*0.25, (layers_in_shape[0][1]/total_size)*0.5, layers_in_shape[0][1]/total_size, ((layers_in_shape[0][2] * layers_in_shape[0][1])/total_size)*0.1, ((layers_in_shape[0][2] * layers_in_shape[0][1])/total_size)*0.25, ((layers_in_shape[0][2] * layers_in_shape[0][1])/total_size)*0.5, (layers_in_shape[0][2] * layers_in_shape[0][1])/total_size, ((layers_in_shape[0][2] * layers_in_shape[0][1] * layers_in_shape[0][3])/total_size)*0.25]
        gap_coarseout = [1]
        coarsein_1 = [1/layers_in_shape[1][1], 0.25, 0.5, 1]
        coarseout_1 = [1/layers_out_shape[1][1], 0.25, 0.5, 1]
        coarsein_2 = [1/layers_in_shape[2][1], 0.25, 0.5, 1]
        coarseout_2 = [1/layers_out_shape[2][1], 0.25, 0.5, 1]
        mul_total_size = layers_out_shape[3][1] * layers_out_shape[3][2] * layers_out_shape[3][3] * layers_out_shape[3][4]
        mul_coarseinout = [(layers_out_shape[3][1]/mul_total_size)*0.1, (layers_out_shape[3][1]/mul_total_size)*0.25, (layers_out_shape[3][1]/mul_total_size)*0.5, layers_out_shape[3][1]/mul_total_size, ((layers_out_shape[3][2] * layers_out_shape[3][1])/mul_total_size)*0.1, ((layers_out_shape[3][2] * layers_out_shape[3][1])/mul_total_size)*0.25, ((layers_out_shape[3][2] * layers_out_shape[3][1])/mul_total_size)*0.5, (layers_out_shape[3][2] * layers_out_shape[3][1])/mul_total_size, (layers_out_shape[3][3] * layers_out_shape[3][2] * layers_out_shape[3][1])/total_size]
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
    # mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
    mem_bw = [(10000000, 10000000)]

    total = [gap_coarsein, gap_coarseout, fine_1, coarsein_1, coarseout_1, fine_2, coarsein_2, coarseout_2, mul_coarseinout, mem_bw]
    combinations = itertools.product(*total)

    print("Calculating {} design points for layer {}.".format(len(gap_coarsein)*len(gap_coarseout)*len(coarsein_1)*len(coarseout_1)*len(coarsein_2)*len(coarseout_2)*len(mul_coarseinout)*len(fine_1)*len(fine_2)*len(mem_bw), name))

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
            processes_pool = Pool(8)
            input_vars = []
            for (gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, (bw_in, bw_out)) in combinations:
                input_vars.append([gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, se.mem_words_per_cycle*bw_in, se.mem_words_per_cycle*bw_out])
            results = multithreaded_modeling(se.get_design_point, input_vars, processes_pool)
            processes_pool.close()
            for r in results:
                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
        else:
            for (gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, (bw_in, bw_out)) in combinations:

                r = se.get_design_point(gapcin, gapcout, f1, c11, c21, f2, c12, c22, mulcinout, se.mem_words_per_cycle*bw_in, se.mem_words_per_cycle*bw_out)

                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r

    print("(gapcin = {:.2f}, gapcout = {:.2f}, fine1={:.2f}, coarsein1={:.2f}, coarseout1={:.2f}, fine2={:.2f}, coarsein2={:.2f}, coarseout2={:.2f}, fmul={:.10f}, bwIn={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn1={:.2f}, RateOut={:.2f}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['config'][0], best['config'][1], best['config'][2], best['config'][3], best['config'][4], best['config'][5], best['config'][6], best['config'][7], best['config'][8], best['config'][9], best['config'][10], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn1'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn1'], best['memBoundedOut']))
    print("*"*40)
    return throughput_gops, throughput_vols, latency, dsp_util, bram_util
    
def elemwise_compose(name, description, model_file, optimization, singlethreaded):
    elem = ElementWiseLayer(description, optimization)
    is_broadcasting = elem.broadcasting

    if optimization == 'brute_force':
        total_size = elem.channels_1*elem.depth_in_1*elem.rows_in_1*elem.cols_in_1
        coarse_in1 = [1/total_size, (elem.channels_1/total_size)*0.1, (elem.channels_1/total_size)*0.25, (elem.channels_1/total_size)*0.5, elem.channels_1/total_size, ((elem.depth_in_1 * elem.channels_1)/total_size)*0.1, ((elem.depth_in_1 * elem.channels_1)/total_size)*0.25, ((elem.depth_in_1 * elem.channels_1)/total_size)*0.5, (elem.depth_in_1 * elem.channels_1)/total_size, ((elem.depth_in_1 * elem.channels_1 * elem.rows_in_1)/total_size)*0.5, (elem.depth_in_1 * elem.channels_1 * elem.rows_in_1)/total_size]

        total_size = elem.channels_2*elem.depth_in_2*elem.rows_in_2*elem.cols_in_2
        coarse_in2 = [1/total_size, (elem.channels_2/total_size)*0.1, (elem.channels_2/total_size)*0.25, (elem.channels_2/total_size)*0.5, elem.channels_2/total_size, ((elem.depth_in_2 * elem.channels_2)/total_size)*0.1, ((elem.depth_in_2 * elem.channels_2)/total_size)*0.25, ((elem.depth_in_2 * elem.channels_2)/total_size)*0.5, (elem.depth_in_2 * elem.channels_2)/total_size, ((elem.depth_in_2 * elem.channels_2 * elem.rows_in_2)/total_size)*0.5, (elem.depth_in_2 * elem.channels_2 * elem.rows_in_2)/total_size]
    else:
        coarse_in1 = [1]
        coarse_in2 = [1]
    coarse_out = coarse_in1
    # mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]
    mem_bw = [(10000000, 10000000, 10000000)]

    total = [coarse_in1, coarse_in2, coarse_out, mem_bw]
    combinations = itertools.product(*total)
    
    print("Calculating {} design points for layer {}.".format(len(coarse_in1)*len(coarse_in2)*len(coarse_out)*len(mem_bw), name))

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
            processes_pool = Pool(8)
            input_vars = []
            for (cin1, cin2, cout, (bw_in1, bw_in2, bw_out)) in combinations:
                input_vars.append([cin1, cin2, cout, elem.mem_words_per_cycle*bw_in1, elem.mem_words_per_cycle*bw_in2, elem.mem_words_per_cycle*bw_out])
            results = multithreaded_modeling(elem.get_design_point, input_vars, processes_pool)
            processes_pool.close()
            for r in results:
                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
        else:
            for (cin1, cin2, cout, (bw_in1, bw_in2, bw_out)) in combinations:
                r = elem.get_design_point(cin1, cin2, cout, elem.mem_words_per_cycle*bw_in1, elem.mem_words_per_cycle*bw_in2, elem.mem_words_per_cycle*bw_out)

                if r['config']:
                    throughput_gops.append(r['GOP/s'])
                    throughput_vols.append(r['vols/s'])
                    latency.append(r['latency(C)'])
                    dsp_util.append(r['DSP'])
                    bram_util.append(r['BRAM'])

                    csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

                    if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        min_latency = r['latency(C)']
                        best = r
                    elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                        if r['DSP'] < best['DSP']:
                            min_latency = r['latency(C)']
                            best = r
    
    print("(cin1={:.2f}, cin2={:.2f}, cout={:.2f}, bwIn1={:.2f}, bwIn2={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn1={:.2f}, RateOut={:.2f}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['config'][0], best['config'][1], best['config'][2], best['config'][3], best['config'][4], best['config'][5], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn1'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn1'], best['memBoundedOut']))
    print("*"*40)

    return throughput_gops, throughput_vols, latency, dsp_util, bram_util

def fc_compose(name, description, model_file, optimization, singlethreaded):
    return [], [], [], [], []