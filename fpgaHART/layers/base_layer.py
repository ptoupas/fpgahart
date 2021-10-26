import configparser
import os
import math
import numpy as np

class BaseLayer():
    def __init__(self, data_format='NHWDC'):
        assert data_format=='NHWDC' or data_format=='NCHWD', "Wrong data format. Accepted formats are 'NHWDC' or 'NCHWD'"
        
        self.get_config()
        self.data_format = data_format
        self.word_bytes = self.word_length/8
        self.cycles_per_sec = self.clock_freq*1e6
        self.mem_bandwidth = self.mem_bw * 1e9
        self.mem_words_per_cycle = (self.mem_bandwidth / self.word_length) / self.cycles_per_sec

    def get_config(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), 'fpgaHART', 'config', 'config_fpga.ini'))

        self.word_length = int(config.get('FPGA Specifications', 'word_length'))
        self.clock_freq = int(config.get('FPGA Specifications', 'clock_freq'))
        self.bram = int(config.get('FPGA Specifications', 'bram'))
        self.bram_bytes = int(config.get('FPGA Specifications', 'bram_type')) / 8
        self.dsp = int(config.get('FPGA Specifications', 'dsp'))
        self.mem_bw = float(config.get('FPGA Specifications', 'mem_bw'))

    def get_dp_performance(self, workload_matrix, ii, muls, adds, mem, depth):
        mem_kb = (mem * self.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.bram_bytes)
        bram_util = (mem_bram / self.bram) * 100

        dsps_util = (muls/self.dsp)*100

        latency_cycles = np.max(np.abs(ii)) + depth
        latency_sec = latency_cycles/self.cycles_per_sec

        thr_in = workload_matrix[0,0]/latency_sec       # Input words per second
        thr_out = workload_matrix[-1,-1]/latency_sec    # Output words per second

        return latency_sec, latency_cycles, thr_in, thr_out, dsps_util, bram_util, mem_kb

    def balance_matrix(self, matrix):        
        rate_ratio = [abs(matrix[i,i]/matrix[i-1,i]) for i in range(1, matrix.shape[1]-1)]

        for i in range(1, matrix.shape[0] - 1):
            layer = matrix.shape[0] - i
            if abs(matrix[layer-1,layer]) > matrix[layer-1,layer-1]:
                # propogate forward
                for j in range(layer,matrix.shape[0]):
                    if abs(matrix[j-1,j]) <= matrix[j-1,j-1]:
                        break
                    matrix[j-1,j] = -matrix[j-1,j-1]
                    if j < matrix.shape[0]:
                        matrix[j,j] = matrix[j-1,j-1]*rate_ratio[j-1]
            elif abs(matrix[layer-1,layer]) < matrix[layer-1,layer-1]:
                # propogate backward
                for j in range(0,layer):
                    if(abs(matrix[layer-j-1,layer-j]) >= matrix[layer-j-1,layer-j-1]):
                        break
                    matrix[layer-j-1,layer-j-1]  = abs(matrix[layer-j-1,layer-j])
                    if layer-j-1 > 0:
                        matrix[layer-j-2,layer-j-1] = -matrix[layer-j-1,layer-j-1]/rate_ratio[layer-1-j-1]
        
        rate_ratio_new = [abs(matrix[i,i]/matrix[i-1,i]) for i in range(1, matrix.shape[1]-1)]
        assert np.allclose(rate_ratio, rate_ratio_new), "{} - {}".format(rate_ratio, rate_ratio_new)

        mem_bounded_in = False
        mem_bounded_out = False

        if matrix[0, 0] < abs(matrix[0, 1]):
            mem_bounded_in = True
            for i in range(0, matrix.shape[0]):
                layer = matrix.shape[0] - i
                if abs(matrix[layer-1,layer]) > matrix[layer-1,layer-1]:
                    # propogate forward
                    for j in range(layer,matrix.shape[0] + 1):
                        if abs(matrix[j-1,j]) <= matrix[j-1,j-1]:
                            break
                        matrix[j-1,j] = -matrix[j-1,j-1]
                        if j < matrix.shape[0]:
                            matrix[j,j] = matrix[j-1,j-1]*rate_ratio[j-1]
                elif abs(matrix[layer-1,layer]) < matrix[layer-1,layer-1]:
                    # propogate backward
                    for j in range(0,layer):
                        if(abs(matrix[layer-j-1,layer-j]) >= matrix[layer-j-1,layer-j-1]):
                            break
                        matrix[layer-j-1,layer-j-1]  = abs(matrix[layer-j-1,layer-j])
                        if layer-j-1 > 0:
                            matrix[layer-j-2,layer-j-1] = -matrix[layer-j-1,layer-j-1]/rate_ratio[layer-1-j-1]

            rate_ratio_new_2 = [abs(matrix[i,i]/matrix[i-1,i]) for i in range(1, matrix.shape[1]-1)]
            assert np.allclose(rate_ratio_new, rate_ratio_new_2), "{} - {}".format(rate_ratio_new, rate_ratio_new_2)

        if abs(matrix[-1, -1]) < matrix[-1, -2]:
            mem_bounded_out = True
            
            for i in range(0, matrix.shape[0]):
                layer = matrix.shape[0] - i
                if abs(matrix[layer-1,layer]) > matrix[layer-1,layer-1]:
                    # propogate forward
                    for j in range(layer,matrix.shape[0] + 1):
                        if abs(matrix[j-1,j]) <= matrix[j-1,j-1]:
                            break
                        matrix[j-1,j] = -matrix[j-1,j-1]
                        if j < matrix.shape[0]:
                            matrix[j,j] = matrix[j-1,j-1]*rate_ratio[j-1]
                elif abs(matrix[layer-1,layer]) < matrix[layer-1,layer-1]:
                    # propogate backward
                    for j in range(0,layer):
                        if(abs(matrix[layer-j-1,layer-j]) >= matrix[layer-j-1,layer-j-1]):
                            break
                        matrix[layer-j-1,layer-j-1]  = abs(matrix[layer-j-1,layer-j])
                        if layer-j-1 > 0:
                            matrix[layer-j-2,layer-j-1] = -matrix[layer-j-1,layer-j-1]/rate_ratio[layer-1-j-1]

            rate_ratio_new_2 = [abs(matrix[i,i]/matrix[i-1,i]) for i in range(1, matrix.shape[1]-1)]
            assert np.allclose(rate_ratio_new, rate_ratio_new_2), "{} - {}".format(rate_ratio_new, rate_ratio_new_2)

        if not mem_bounded_in and not mem_bounded_out:
            matrix[0, 0] = abs(matrix[0, 1])
            matrix[-1, -1] = -matrix[-1, -1 -1]

        return matrix, mem_bounded_in, mem_bounded_out

    def balance_matrix_elemwise(self, matrix, branch_node):
        mem_bounded_in_1 = False
        mem_bounded_in_2 = False
        mem_bounded_out = False

        branch_ratio = abs(matrix[0,branch_node])/abs(matrix[branch_node-1,branch_node])

        if abs(matrix[branch_node-1, branch_node]) > matrix[branch_node-1, branch_node-1]:
            mem_bounded_in_2 = True
            matrix[branch_node-1, branch_node] = -matrix[branch_node-1, branch_node-1]
        else:
            matrix[branch_node-1, branch_node-1] = abs(matrix[branch_node-1, branch_node])

        if abs(matrix[0, branch_node]) > matrix[0, 0]:
            mem_bounded_in_1 = True
            matrix[0, branch_node] = -matrix[0,0]
        else:
            matrix[0,0] = abs(matrix[0, branch_node])

        matrix[branch_node,branch_node] = min(abs(matrix[0, branch_node]), abs(matrix[branch_node-1, branch_node]), matrix[branch_node,branch_node])

        if abs(matrix[-1, -1]) < matrix[-1, -2]:
            mem_bounded_out = True
            matrix[branch_node,branch_node] = abs(matrix[-1, -1])
        else:
            matrix[-1, -1] = -matrix[branch_node,branch_node]

        if matrix[branch_node,branch_node] <= abs(matrix[branch_node-1,branch_node]):
            matrix[branch_node-1,branch_node] = -matrix[branch_node,branch_node]
            matrix[branch_node-1,branch_node-1] = matrix[branch_node,branch_node]
        else:
            assert False, "Failed to move backwards on Γ matrix for input 2"

        if matrix[branch_node,branch_node] <= abs(matrix[0,branch_node]):
            matrix[0,branch_node] = -matrix[branch_node,branch_node]
            matrix[0,0] = matrix[branch_node,branch_node]
        else:
            assert False, "Failed to move backwards on Γ matrix for input 1"

        assert abs(matrix[0,branch_node]) == abs(matrix[branch_node-1,branch_node]), "Problem with the graph balancing\n{}\n{}".format(origin_matrix, matrix)
        assert abs(matrix[0,branch_node]) == abs(matrix[branch_node,branch_node]), "Problem with the graph balancing\n{}\n{}".format(origin_matrix, matrix)

        return matrix, mem_bounded_in_1, mem_bounded_in_2, mem_bounded_out

    def balance_matrix_elemwise_broadcasting(self, matrix, branch_node, branch_ratio):
        mem_bounded_in_1 = False
        mem_bounded_in_2 = False
        mem_bounded_out = False
        
        curr_branch_ratio = 1/(abs(matrix[0, branch_node])/abs(matrix[branch_node-1, branch_node]))
        if curr_branch_ratio < branch_ratio:
            matrix[0, branch_node] = -abs(matrix[branch_node-1, branch_node]) / branch_ratio
        elif curr_branch_ratio > branch_ratio:
            matrix[branch_node-1, branch_node] = -abs(matrix[0, branch_node]) * branch_ratio

        if abs(matrix[branch_node-1, branch_node]) > matrix[branch_node-1, branch_node-1]:
            mem_bounded_in_2 = True
            matrix[branch_node-1, branch_node] = -matrix[branch_node-1, branch_node-1]
            matrix[0, branch_node] = -abs(matrix[branch_node-1, branch_node]) / branch_ratio
        else:
            matrix[branch_node-1, branch_node-1] = abs(matrix[branch_node-1, branch_node])

        if abs(matrix[0, branch_node]) > matrix[0, 0]:
            mem_bounded_in_1 = True
            matrix[0, branch_node] = -matrix[0,0]
            matrix[branch_node-1, branch_node] = -abs(matrix[0, branch_node]) * branch_ratio
        else:
            matrix[0,0] = abs(matrix[0, branch_node])

        matrix[branch_node,branch_node] = abs(matrix[0, branch_node])

        if abs(matrix[-1, -1]) < matrix[-1, -2]:
            mem_bounded_out = True
            matrix[branch_node,branch_node] = abs(matrix[-1, -1])
        else:
            matrix[-1, -1] = -matrix[branch_node,branch_node]

        curr_branch_ratio = 1/(abs(matrix[branch_node, branch_node])/abs(matrix[branch_node-1, branch_node]))
        if curr_branch_ratio > branch_ratio:
            matrix[branch_node-1, branch_node] = -abs(matrix[branch_node, branch_node]) * branch_ratio
            matrix[branch_node-1, branch_node-1] = abs(matrix[branch_node, branch_node]) * branch_ratio

            matrix[0,branch_node] = -matrix[branch_node,branch_node]
            matrix[0,0] = matrix[branch_node,branch_node]
        
        curr_branch_ratio = 1/(abs(matrix[0, branch_node])/abs(matrix[branch_node-1, branch_node]))
        assert branch_ratio == curr_branch_ratio, "Problem with the graph balancing"
        curr_branch_ratio = 1/(abs(matrix[branch_node, branch_node])/abs(matrix[branch_node-1, branch_node]))
        assert branch_ratio == curr_branch_ratio, "Problem with the graph balancing"

        return matrix, mem_bounded_in_1, mem_bounded_in_2, mem_bounded_out
