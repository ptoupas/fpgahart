import configparser
import os
import math
import numpy as np

class BaseLayer():
    def __init__(self, data_format='NDHWC'):
        assert data_format=='NDHWC' or data_format=='NCDHW', "Wrong data format. Accepted formats are 'NDHWC' or 'NCDHW'"
        
        self.get_config()
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

    def get_dp_performance(self, gamma, ii, muls, adds, mem, depth):
        mem_kb = (mem * self.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.bram_bytes)
        bram_util = (mem_bram / self.bram) * 100

        dsps_util = (muls/self.dsp)*100

        rin = gamma[0,0]
        rout = abs(gamma[-1,-1])
        thr_in = self.cycles_per_sec * rin      # Words per second
        thr_out = self.cycles_per_sec * rout    # Words per second

        latency = (depth + np.max(np.abs(ii)))/self.cycles_per_sec

        return latency, thr_in, thr_out, dsps_util, bram_util

    def balance_matrix(self, matrix):
        rate_ratio = [abs(matrix[i,i]/matrix[i-1,i]) for i in range(1, matrix.shape[1]-1)]

        for i in range(0, matrix.shape[0]):
            layer = matrix.shape[0] - i
            # print(layer, matrix[layer-1,layer], matrix[layer-1,layer-1])
            if abs(matrix[layer-1,layer]) > matrix[layer-1,layer-1]:
                # propogate forward
                # print("balancing needed (forward pass)")
                for j in range(layer,matrix.shape[0]+1):
                    if abs(matrix[j-1,j]) <= matrix[j-1,j-1]:
                        break
                    # print(layer, matrix[j-1,j], matrix[j-1,j-1])
                    matrix[j-1,j] = -matrix[j-1,j-1]
                    if j < matrix.shape[0]:
                        # print("ratio", j-1, rate_ratio[j-1])
                        matrix[j,j] = matrix[j-1,j-1]*rate_ratio[j-1]
                # print("finished balancing (forward pass)")
            elif abs(matrix[layer-1,layer]) < matrix[layer-1,layer-1]:
                # propogate backward
                # print("balancing needed (backward pass)")
                for j in range(0,layer):
                    if(abs(matrix[layer-j-1,layer-j]) >= matrix[layer-j-1,layer-j-1]):
                        break
                    # print(layer, matrix[layer-j-1,layer-j], matrix[layer-j-1,layer-j-1])
                    matrix[layer-j-1,layer-j-1]  = abs(matrix[layer-j-1,layer-j])
                    if layer-j-1 > 0:
                        # print("ratio", layer-1-j-1, rate_ratio[layer-1-j-1])
                        matrix[layer-j-2,layer-j-1] = -matrix[layer-j-1,layer-j-1]/rate_ratio[layer-1-j-1]
                # print("finished balancing (backward pass)")
        
        rate_ratio_new = [abs(matrix[i,i]/matrix[i-1,i]) for i in range(1, matrix.shape[1]-1)]
        assert np.allclose(rate_ratio, rate_ratio_new), "{} - {}".format(rate_ratio, rate_ratio_new)

        return matrix
        