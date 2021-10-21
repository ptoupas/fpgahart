import numpy as np
import math
import scipy.optimize as optimize
from scipy.optimize import NonlinearConstraint, Bounds
from .base_layer import BaseLayer
from .convolutional_3d import Convolutional3DLayer
from .gap import GAPLayer
from ..primitive_blocks.elemwise import ElementWiseLayer
np.set_printoptions(precision=5, suppress=True, linewidth=150)
np.seterr(divide='ignore', invalid='ignore')

DEBUG=False

class SqueezeExcitationLayer(BaseLayer):
    def __init__(self, description, optimization):
        super().__init__()
        self.optimization = optimization

        self.branching = description['branching']
        if self.branching:
            self.branch_shape = description['shape_branch']
        self.input_shape = description['shape_in'][0]
        self.output_shape = description['shape_out']

        self.sequencial = {}
        for n_se, l_se in description['primitive_ops'].items():
            if l_se['operation'] == 'GlobalAveragePool':
                self.sequencial[n_se] = GAPLayer(l_se, optimization)
            elif l_se['operation'] == 'Conv':
                self.sequencial[n_se] = Convolutional3DLayer(l_se, optimization)
            elif l_se['operation'] == 'Relu' or l_se['operation'] == 'Sigmoid':
                continue
                self.sequencial[n_se] = ActivationLayer(l_se, optimization)
            elif l_se['operation'] == 'Mul':
                self.sequencial[n_se] = ElementWiseLayer(l_se, optimization)
        self.num_layers = len(self.sequencial) + 2

    def get_design_point(self, f_gap_coarsein, f_gap_coarseout, f_fine_1, f_coarseIn_1, f_coarseOut_1, f_fine_2, f_coarseIn_2, f_coarseOut_2, f_mul_coarseinout, mem_bw_in, mem_bw_out):

        #TODO: Add an extra connection to the graph for the 2nd input of MUL operation
        gamma_matrix = np.zeros( shape=(self.num_layers-1, self.num_layers) , dtype=float )
        gamma_matrix[0, 0] = mem_bw_in
        gamma_matrix[-1, -1] = -mem_bw_out

        total_muls = 0
        total_adds = 0
        total_memory = 0
        total_depth = 0
        first_layer_bw_in = False
        prev_layer_rate = mem_bw_in

        #TODO: Find a way to discriminate between different layers to pass them the correct configuration 
        for n, l in enumerate(self.sequencial.keys()):
            if n == len(self.sequencial) - 1:
                curr_layer_rate = mem_bw_out
            else:
                curr_layer_rate = 10000000
            
            if n==0:
                self.sequencial[l].get_design_point(f_gap_coarsein, f_gap_coarseout, prev_layer_rate, curr_layer_rate)
            elif n==1:
                self.sequencial[l].get_design_point(f_fine_1, f_coarseIn_1, f_coarseOut_1, prev_layer_rate, curr_layer_rate)
            elif n==2:
                self.sequencial[l].get_design_point(f_fine_2, f_coarseIn_2, f_coarseOut_2, prev_layer_rate, curr_layer_rate)
            elif n==3:
                self.sequencial[l].get_design_point(f_mul_coarseinout, mem_bw_in, prev_layer_rate, curr_layer_rate)

            if n==3:
                full_rate_in_1, full_rate_in_2, full_rate_out, muls, adds, memory, depth, mem_bd_in_1, mem_bd_in_2, mem_bd_out = self.sequencial[l].get_dp_info()
                # gamma_matrix[0, n+1] = -full_rate_in_1
                gamma_matrix[n, n+1] = -full_rate_in_2
                gamma_matrix[n+1, n+1] = full_rate_out
            else:
                full_rate_in, full_rate_out, muls, adds, memory, depth, mem_bd_in, mem_bd_out = self.sequencial[l].get_dp_info()
                gamma_matrix[n, n+1] = -full_rate_in
                gamma_matrix[n+1, n+1] = full_rate_out

            if n==0:
               first_layer_bw_in = mem_bd_in
            prev_layer_rate = full_rate_out

            total_muls += muls
            total_adds += adds
            total_memory += memory
            total_depth += depth

        if DEBUG:
            print("Γ:\n{}".format(gamma_matrix))
        gamma_matrix_balanced, mem_bounded_in, mem_bounded_out = self.balance_matrix(gamma_matrix.copy())
        if DEBUG:
            print("Γ Balanced:\n{}".format(gamma_matrix_balanced))
        workload_matrix = self.get_workload_matrix()
        if DEBUG:
            print("WL:\n{}".format(workload_matrix))
        ii_matrix = np.nan_to_num(workload_matrix/gamma_matrix_balanced)
        if DEBUG:
            print("II:\n{}".format(ii_matrix))

        mem_bounded_in = mem_bounded_in or first_layer_bw_in
        latency_sec, latency_cycles, thr_in, thr_out, dsps_util, bram_util = self.get_dp_performance(workload_matrix, ii_matrix, total_muls, total_adds, total_memory, total_depth)
        total_ops = self.get_total_workload()
        throughput_ops = total_ops/latency_sec
        thr_in /= workload_matrix[0,0]              # Volumes per second
        thr_out /= workload_matrix[-1,-1]           # Volumes per second
        assert math.isclose(thr_in, thr_out), "Thoughputs missmatch. IN = {}, OUT = {}.".format(thr_in, thr_out)

        if DEBUG:
            if dsps_util < 90. and bram_util < 90.:
                print("GOPs/s={:.2f}, DSPS={:.2f}, BRAM={:.2f}, depth={}, latency(s)={:.2f}, latency(c)={:.2f}, mem bounded in = {}, mem bounded out = {}".format(throughput_ops*1e-9, dsps_util, bram_util, total_depth, latency_sec, latency_cycles, mem_bounded_in, mem_bounded_out))
            else:
                print("Discarding design point.")

        return f_gap_coarsein, f_gap_coarseout, f_fine_1, f_coarseIn_1, f_coarseOut_1, f_fine_2, f_coarseIn_2, f_coarseOut_2, f_mul_coarseinout, mem_bw_in, mem_bw_out, dsps_util, bram_util, latency_sec, int(latency_cycles), throughput_ops, thr_in, thr_out, total_depth, total_ops, mem_bounded_in, mem_bounded_out
        
    def get_total_workload(self):
        total_wl = 0
        for n, l in enumerate(self.sequencial.keys()):
            total_wl += self.sequencial[l].get_total_workload()

        return total_wl

    def get_workload_matrix(self):
        #TODO: Add an extra connection to the graph for the 2nd input of MUL operation
        workload_matrix = np.zeros( shape=(self.num_layers-1, self.num_layers) , dtype=float )

        for n, l in enumerate(self.sequencial.keys()):
            if n == 0:
                workload_matrix[0, 0] = np.prod(np.array(self.sequencial[l].input_shape[1:]))
            if n == len(self.sequencial) - 1:
                workload_matrix[-1, -1] = np.prod(np.array(self.sequencial[l].output_shape[1:]))
            
            if isinstance(self.sequencial[l], ElementWiseLayer):
                # workload_matrix[0, n+1] = np.prod(np.array(self.sequencial[l].input_shape_1[1:]))
                workload_matrix[n, n+1] = np.prod(np.array(self.sequencial[l].input_shape_2[1:]))
            else:
                workload_matrix[n, n+1] = np.prod(np.array(self.sequencial[l].input_shape[1:]))
            workload_matrix[n+1, n+1] = np.prod(np.array(self.sequencial[l].output_shape[1:]))
        return workload_matrix