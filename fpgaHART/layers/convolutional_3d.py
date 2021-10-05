import numpy as np
import math

class Convolutional3DLayer():
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
        
        if self.pointwise:
            # Sliding Window Depth
            depth = 1 * self.channels
            # Convolution 3D Depth
            depth += 1
            # Memory in words/elements
            memory = 1 * self.channels
        else:
            # Sliding Window Depth
            depth = self.depth_in * self.rows_in * max(self.kw - 1, 1) * self.channels + self.channels * self.kd * self.kh * max(self.kw - 1, 1)
            # Convolution 3D Depth
            depth += math.ceil(1/f_fine)
            # Memory in words/elements
            memory = self.depth_in * self.rows_in * max(self.kw - 1, 1) * self.channels + self.channels * self.kd * self.kh * max(self.kw - 1, 1)

        parallel_muls = int(np.prod(np.array(self.kernel_shape))) * f_fine
        parallel_adds = (int(np.prod(np.array(self.kernel_shape))) - 1) * f_fine

        if not self.depthwise:
            # Accumulation Depth
            depth += self.channels * self.filters
            # Accumulation Additions
            parallel_adds += int(np.prod(np.array(self.output_shape))) * (self.channels - 1)