from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..layers.gap import GAPLayer
from ..primitive_blocks.elemwise import ElementWiseLayer
import itertools
from multiprocessing import Pool

def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results

def layer_compose(name, description, optimization, singlethreaded):
    if description['operation'] == 'Conv':
        return conv_compose(name, description, optimization, singlethreaded)
    elif description['operation'] == 'BatchNormalization':
        return batchnorm_compose(name, description, optimization, singlethreaded)
    elif description['operation'] == 'GlobalAveragePool':
        return gap_compose(name, description, optimization, singlethreaded)
    elif description['operation'] == 'Relu' or description['operation'] == 'Sigmoid' or description['operation'] == 'Swish':
        return activation_compose(name, description, optimization, singlethreaded)
    elif description['operation'] == 'SqueezeExcitation':
        return se_compose(name, description, optimization, singlethreaded)
    elif description['operation'] == 'Add' or description['operation'] == 'Mul':
        return elemwise_compose(name, description, optimization, singlethreaded)
    elif description['operation'] == 'Gemm' or description['operation'] == 'MatMul':
        return fc_compose(name, description, optimization, singlethreaded)
    else:
        assert False, "{} operation in layer {} is not supported".format(description['operation'], name)

def conv_compose(name, description, optimization, singlethreaded):
    conv = Convolutional3DLayer(description, optimization)
    
    if conv.depthwise:
        convtype = 'DepthWise'
    elif conv.pointwise:
        convtype = 'PointWise'
    else:
        convtype = '3DConv'
    
    if optimization == 'brute_force':
        fine = [0.25, 0.5, 0.75, 1]
        coarsein = [1/conv.channels, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
        coarseout = [1/conv.filters, 0.1, 0.3, 0.5, 0.7, 0.9, 0.1]
    else:
        fine = [1]
        coarsein = [1]
        coarseout = [1]
    mem_bw = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3), (0.8,0.2), (0.9,0.1)]

    if convtype == 'PointWise':
        fine = [1]
    total = [fine, coarsein, coarseout, mem_bw]
    combinations = itertools.product(*total)
    
    print("*"*40)
    print("Calculating {} design points for layer {} ({}).".format(len(fine)*len(coarsein)*len(coarseout)*len(mem_bw), name, convtype))

    throughput_gops = []
    throughput_vols = []
    latency = []
    dsp_util = []
    bram_util = []

    min_latency = 1000000000000
    best = {}
    if not singlethreaded:
        processes_pool = Pool(4)
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
                if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                    min_latency = r['latency(C)']
                    best = r
                elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
                    if r['DSP'] < best['DSP']:
                        min_latency = r['latency(C)']
                        best = r
    
    print("(fine={:.2f}, cIn={:.2f}, cOut={:.2f}, bwIn={:.2f}, bwOut={:.2f}) Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, RateIn={:.2f}, RateOut={:.2f}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['config'][0], best['config'][1], best['config'][2], best['config'][3], best['config'][4], best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn'], best['memBoundedOut']))
    print("*"*40)

    return throughput_gops, throughput_vols, latency, dsp_util, bram_util

def batchnorm_compose(name, description, optimization, singlethreaded):
    return [], [], [], [], []

def gap_compose(name, description, optimization, singlethreaded):
    return [], [], [], [], []

def activation_compose(name, description, optimization, singlethreaded):
    return [], [], [], [], []

def se_compose(name, description, optimization, singlethreaded):
    return [], [], [], [], []

def elemwise_compose(name, description, optimization, singlethreaded):
    return [], [], [], [], []

def fc_compose(name, description, optimization, singlethreaded):
    return [], [], [], [], []