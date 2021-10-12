import numpy as np
import math
import scipy.optimize as optimize
from scipy.optimize import NonlinearConstraint, Bounds
from .base_layer import BaseLayer
from .convolutional_3d import Convolutional3DLayer
np.set_printoptions(precision=5, suppress=True, linewidth=150)

DEBUG=False

class SqueezeExcitationLayer(BaseLayer):
    def __init__(self, description, optimization):
        super().__init__()
        self.optimization = optimization

        self.sequencial = {}
        for n_se, l_se in description['primitive_ops'].items():
          if l_se['operation'] == 'GlobalAveragePool':
            continue
            self.sequencial[n_se] = GAPLayer(l_se, optimization)
          elif l_se['operation'] == 'Conv':
            self.sequencial[n_se] = Convolutional3DLayer(l_se, optimization)
          elif l_se['operation'] == 'Relu' or l_se['operation'] == 'Sigmoid':
            continue
            self.sequencial[n_se] = ActivationLayer(l_se, optimization)
          elif l_se['operation'] == 'Mul':
            continue
            self.sequencial[n_se] = ElementWiseLayer(l_se, optimization)
        
        self.num_layers = len(self.sequencial) + 2
        self.gamma_matrix = np.zeros( shape=(self.num_layers-1, self.num_layers) , dtype=float )
        for l in self.sequencial.keys():
          print(l, self.sequencial[l].input_shape, self.sequencial[l].output_shape, self.sequencial[l].kernel_shape)

    def get_design_point(self, f_fine_1, f_coarseIn_1, f_coarseOut_1, f_fine_2, f_coarseIn_2, f_coarseOut_2, mem_bw_in, mem_bw_out):
        self.gamma_matrix[0, 0] = mem_bw_in
        self.gamma_matrix[-1, -1] = -mem_bw_out

        total_muls = 0
        total_memory = 0
        total_depth = 0
        #TODO: Find a way to discriminate between different layers to pass them the correct configuration
        prev_layer_rate = mem_bw_in
        for n, l in enumerate(self.sequencial.keys()):
            if n == len(self.sequencial) - 1:
                curr_layer_rate = mem_bw_out
            else:
                curr_layer_rate = 10000000
            print("A", prev_layer_rate, curr_layer_rate)
            self.sequencial[l].get_design_point(f_fine_1, f_coarseIn_1, f_coarseOut_1, prev_layer_rate, curr_layer_rate)
            full_rate_in, full_rate_out, muls, memory, depth = self.sequencial[l].get_dp_info()
            prev_layer_rate = full_rate_out
            print(full_rate_in, full_rate_out, muls, memory, depth)
            self.gamma_matrix[n, n+1] = full_rate_in
            self.gamma_matrix[n+1, n+1] = full_rate_out
            total_muls += muls
            total_memory += memory
            total_depth += depth

        gamma_matrix_balanced, mem_bounded_in, mem_bounded_out = self.balance_matrix(self.gamma_matrix.copy())
        print(gamma_matrix_balanced)
        print("Bounded in = {}, Bounded out = {}".format(mem_bounded_in, mem_bounded_out))