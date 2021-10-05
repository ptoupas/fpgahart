import numpy as np
import math
from .base_layer import BaseLayer

class Convolutional3DLayer(BaseLayer):
    def __init__(self, description):
        self.input_shape = description['shape_in'][0]
        self.depth_in = self.input_shape[2]
        self.rows_in = self.input_shape[3]
        self.cols_in = self.input_shape[4]
        self.output_shape = description['shape_out']
        self.depth_out = self.output_shape[2]
        self.rows_out = self.output_shape[3]
        self.cols_out = self.output_shape[4]

        self.kernel_shape = description['kernel'][2:]
        self.kd = self.kernel_shape[0]
        self.kh = self.kernel_shape[1]
        self.kw = self.kernel_shape[2]
        self.bias_shape = description['bias']

        self.channels = self.input_shape[1]
        self.filters = self.output_shape[1]

        self.groups = description['groups']

        self.padding = description['padding']
        self.stride = description['stride']
        self.dilation = description['dilation']

        self.branching = description['branching']

        self.depthwise = False
        self.pointwise = False
        if self.groups == self.channels:
            self.depthwise = True
        elif np.prod(np.array(self.kernel_shape)) == 1:
            self.pointwise = True
            
    def get_design_point(self, f_fine, f_coarseIn, f_coarseOut):
        
        kernel_elems = int(np.prod(np.array(self.kernel_shape)))

        if self.pointwise:
            # Sliding Window Depth
            depth = 1
            # Convolution 3D Depth
            depth += 1
            # Sliding Window buffer in words/elements
            init_buffer = 1
        else:
            # Sliding Window Depth
            depth = self.depth_in * self.rows_in * max(self.kw - 1, 1) * self.channels + self.channels * self.kd * self.kh * max(self.kw - 1, 1)
            # Convolution 3D Depth
            depth += math.ceil(1/f_fine)
            # Sliding Window buffer in words/elements
            init_buffer = self.depth_in * self.rows_in * max(self.kw - 1, 1) * self.channels + self.channels * self.kd * self.kh * max(self.kw - 1, 1)

        max_parallel_muls = kernel_elems * self.channels * self.filters
        max_parallel_adds = (kernel_elems - 1) * self.channels * self.filters
        memory = init_buffer + kernel_elems * self.channels * self.filters

        if not self.depthwise:
            # Accumulation Depth
            depth += math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
            # Accumulation Additions
            max_parallel_adds += (self.channels - 1) * self.filters
            # Accumulation Buffer
            memory += self.channels
        
        max_parallel_muls *= f_fine * f_coarseIn * f_coarseOut
        max_parallel_adds *= f_fine * f_coarseIn * f_coarseOut


        rin_sw = 1                                                                                          # 1 * 
        rout_sw = (self.depth_out*self.rows_out*self.cols_out)/(self.depth_in*self.rows_in*self.cols_in)    # Kd X Kw X Kh

        rin_pop = 1
        rout_pop = 1
        
        rin_conv = f_fine           # Kd X Kw X Kh
        rout_conv = f_fine          # 1

        rin_accum = 1                   
        rout_accum = 1/self.channels

        thr_in, thr_out, dsps_util, bram_util = self.get_dp_performance(1, 1, max_parallel_muls, max_parallel_adds, memory)
        print("DSP % = {}, BRAM % = {}".format(dsps_util, bram_util))