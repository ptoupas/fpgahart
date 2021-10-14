import numpy as np
import math
import scipy.optimize as optimize
from scipy.optimize import NonlinearConstraint, Bounds
from .base_layer import BaseLayer
np.set_printoptions(precision=5, suppress=True, linewidth=150)
np.seterr(divide='ignore', invalid='ignore')

DEBUG=False

class GAPLayer(BaseLayer):
    def __init__(self, description, optimization):
        super().__init__()

        self.optimization = optimization
        self.input_shape = description['shape_in'][0]
        self.depth_in = self.input_shape[2]
        self.rows_in = self.input_shape[3]
        self.cols_in = self.input_shape[4]
        self.output_shape = description['shape_out']
        self.depth_out = self.output_shape[2]
        self.rows_out = self.output_shape[3]
        self.cols_out = self.output_shape[4]

        self.channels = self.input_shape[1]
        self.filters = self.output_shape[1]

    def update_layer(self):
        self.full_rate_in = 0
        self.full_rate_out = 0
        self.max_parallel_muls = 0
        self.max_parallel_adds = 0
        self.memory = 0
        self.depth = 0
        self.mem_bd_in = False
        self.mem_bd_out = False

    def get_total_workload(self):
        return self.channels * 2

    def get_dp_info(self):
        return self.full_rate_in, self.full_rate_out, self.max_parallel_muls, self.max_parallel_adds, self.memory, self.depth, self.mem_bd_in, self.mem_bd_out

    def get_design_point(self, coarse_in_out, mem_bw_in, mem_bw_out):
        self.update_layer()

        gamma_matrix = self.get_rate_matrix() * self.get_stream_matrix(coarse_in_out) * self.get_data_matrix(mem_bw_in, mem_bw_out)
        if DEBUG:
            print("Γ:\n{}".format(gamma_matrix))
        gamma_matrix_balanced, mem_bounded_in, mem_bounded_out = self.balance_matrix(gamma_matrix.copy())
        if DEBUG:
            print("Γ Balanced:\n{}".format(gamma_matrix_balanced))
        workload_matrix = self.get_workload_matrix()
        ii_matrix = np.nan_to_num(workload_matrix/gamma_matrix_balanced)
        if DEBUG:
            print("II:\n{}".format(ii_matrix))

        if self.data_format == 'NCHWD':
            max_parallel_muls = 2
            max_parallel_adds = math.ceil(self.depth_in  * coarse_in_out)
            memory = 1
            depth = (self.depth_in * self.rows_in * self.cols_in)/math.ceil(self.depth_in  * coarse_in_out)
        else:
            max_parallel_muls = (self.channels * coarse_in_out) * 2
            max_parallel_adds = math.ceil(self.channels * coarse_in_out)
            memory = self.channels
            depth = (self.depth_in * self.rows_in * self.cols_in * self.channels)/math.ceil(self.channels * coarse_in_out)

        latency_sec, latency_cycles, thr_in, thr_out, dsps_util, bram_util = self.get_dp_performance(workload_matrix, ii_matrix, max_parallel_muls, max_parallel_adds, memory, depth)
        total_ops = self.get_total_workload()
        throughput_ops = total_ops/latency_sec
        thr_in /= (np.prod(np.array(self.input_shape[2:])) * self.channels)         # Volumes per second
        thr_out /= (np.prod(np.array(self.output_shape[2:])) * self.filters)        # Volumes per second
        assert math.isclose(thr_in, thr_out), "Thoughputs missmatch. IN = {}, OUT = {}.".format(thr_in, thr_out)

        self.full_rate_in = gamma_matrix_balanced[0, 0]
        self.full_rate_out = abs(gamma_matrix_balanced[-1, -1])
        self.max_parallel_muls = max_parallel_muls
        self.max_parallel_adds = max_parallel_adds
        self.memory = memory
        self.depth = depth
        self.mem_bd_in = mem_bounded_in
        self.mem_bd_out = mem_bounded_out
        if DEBUG:
            print("*"*40, coarse_in_out, latency_cycles, depth)
        return coarse_in_out, mem_bw_in, mem_bw_out, dsps_util, max_parallel_muls, bram_util, latency_sec, int(latency_cycles), throughput_ops, thr_in, thr_out, depth, total_ops, mem_bounded_in, mem_bounded_out

    def get_rate_matrix(self):
        rate_matrix = np.zeros( shape=(2,3) , dtype=float )

        rate_matrix[0, 0] = 1
        
        rate_matrix[0, 1] = 1
        if self.data_format == 'NCHWD':
           rate_matrix[1, 1] = 1/(self.depth_in * self.rows_in * self.cols_in)
        else:
            rate_matrix[1, 1] = 1

        rate_matrix[1, 2] = 1

        if DEBUG:
            print("R:\n{}".format(rate_matrix))
        return rate_matrix

    def get_stream_matrix(self, coarse_in_out):
        stream_matrix = np.zeros( shape=(2,3) , dtype=float )

        stream_matrix[0, 0] = 1
        
        if self.data_format == 'NCHWD':
            stream_matrix[0, 1] = math.ceil(self.depth_in  * coarse_in_out)
            stream_matrix[1, 1] = math.ceil(self.depth_in  * coarse_in_out)
        else:
            stream_matrix[0, 1] = math.ceil(self.channels * coarse_in_out)
            stream_matrix[1, 1] = math.ceil(self.channels * coarse_in_out)

        stream_matrix[1, 2] = 1

        if DEBUG:
            print("S:\n{}".format(stream_matrix))
        return stream_matrix

    def get_data_matrix(self, mem_bw_in, mem_bw_out):
        data_matrix = np.zeros( shape=(2,3) , dtype=float )

        data_matrix[0, 0] = mem_bw_in
        
        data_matrix[0, 1] = -(self.depth_in * self.rows_in * self.cols_in)
        data_matrix[1, 1] = 1

        data_matrix[1, 2] = -mem_bw_out

        if DEBUG:
            print("D:\n{}".format(data_matrix))
        return data_matrix
        
    def get_workload_matrix(self):
        in_volume = self.depth_in * self.rows_in * self.cols_in * self.channels
        out_volume = self.depth_out * self.rows_out * self.cols_out * self.filters

        workload_matrix = np.zeros( shape=(2,3) , dtype=float )

        workload_matrix[0, 0] = in_volume
        
        workload_matrix[0, 1] = in_volume
        workload_matrix[1, 1] = out_volume

        workload_matrix[1, 2] = out_volume

        if DEBUG:
            print("WL:\n{}".format(workload_matrix))
        return workload_matrix
