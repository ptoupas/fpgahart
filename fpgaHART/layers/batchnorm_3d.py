import numpy as np
import math
import scipy.optimize as optimize
from scipy.optimize import NonlinearConstraint, Bounds
from .base_layer import BaseLayer
np.set_printoptions(precision=5, suppress=True, linewidth=150)
np.seterr(divide='ignore', invalid='ignore')

DEBUG=False

class BatchNorm3DLayer(BaseLayer):
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
        self.full_rate_in = []
        self.full_rate_out = []
        self.max_parallel_muls = 0
        self.max_parallel_adds = 0
        self.memory = 0
        self.memoryKB = 0
        self.depth = 0
        self.mem_bd_in = []
        self.mem_bd_out = []
        self.config = []
        self.dsps_util = 0
        self.bram_util = 0
        self.latency_sec = 0
        self.latency_cycles = 0
        self.throughput_ops = 0
        self.throughput_vols = 0

    def get_total_workload(self):
        return int(np.prod(np.array(self.output_shape[1:])))

    def get_dp_info(self):
        dp_info = {}

        dp_info['latency(C)'] = self.latency_cycles
        dp_info['latency(S)'] = self.latency_sec
        dp_info['GOP/s'] = self.throughput_ops*1e-9
        dp_info['vols/s'] = self.throughput_vols
        dp_info['DSP'] = self.dsps_util
        dp_info['BRAM'] = self.bram_util
        dp_info['rateIn'] = self.full_rate_in
        dp_info['rateOut'] = self.full_rate_out
        dp_info['depth'] = self.depth
        dp_info['muls'] = self.max_parallel_muls
        dp_info['adds'] = self.max_parallel_adds
        dp_info['memWords'] = self.memory
        dp_info['memKBs'] = self.memoryKB
        dp_info['memBoundedIn'] = self.mem_bd_in
        dp_info['memBoundedOut'] = self.mem_bd_out
        dp_info['config'] = self.config
        
        return dp_info

    def get_num_streams(self):
        self.max_streams_in = self.channels
        self.max_streams_out = self.filters
        return self.max_streams_in, self.max_streams_out

    def get_design_point(self, coarse_inout, mem_bw_in, mem_bw_out):
        self.update_layer()

        if self.optimization == 'Powell':
            initial_guess = [0.5, mem_bw_in, mem_bw_out]
            lower_bound = 1/(self.channels*self.depth_in*self.rows_in*self.cols_in)
            bnds = ((lower_bound, 1.0), (lower_bound, 1.0), (0.001, mem_bw_in), (0.001, mem_bw_out))   
            result = optimize.minimize(self.get_latency, initial_guess, method=self.optimization, bounds=bnds)
            if result.success:
                coarse_inout, mem_bw_in, mem_bw_out = result.x
            else:
                print("Failed to optimize. Skipping...")
                return

        gamma_matrix = self.get_rate_matrix() * self.get_stream_matrix(coarse_inout) * self.get_data_matrix(mem_bw_in, mem_bw_out)
        if DEBUG:
            print("Γ:\n{}".format(gamma_matrix))
        gamma_matrix_balanced, mem_bounded_in, mem_bounded_out = self.balance_matrix(gamma_matrix.copy())
        if DEBUG:
            print("Γ Balanced:\n{}".format(gamma_matrix_balanced))
        workload_matrix = self.get_workload_matrix()
        ii_matrix = np.nan_to_num(workload_matrix/gamma_matrix_balanced)
        if DEBUG:
            print("II:\n{}".format(ii_matrix))

        max_parallel_muls = math.ceil(self.channels * coarse_inout)
        max_parallel_adds = math.ceil(self.channels * coarse_inout)
        memory = 1
        depth = 1

        latency_sec, latency_cycles, thr_in, thr_out, dsps_util, bram_util, memKBs = self.get_dp_performance(workload_matrix, ii_matrix, max_parallel_muls, max_parallel_adds, memory, depth)
        total_ops = self.get_total_workload()
        throughput_ops = total_ops/latency_sec
        thr_in /= workload_matrix[0, 0]             # Volumes per second
        thr_out /= workload_matrix[-1, -1]          # Volumes per second
        assert math.isclose(thr_in, thr_out), "Thoughputs missmatch. IN = {}, OUT = {}.".format(thr_in, thr_out)

        if dsps_util < 90. and bram_util < 90.:
            self.full_rate_in = [gamma_matrix_balanced[0, 0]]
            self.full_rate_out = [abs(gamma_matrix_balanced[-1, -1])]
            self.max_parallel_muls = max_parallel_muls
            self.max_parallel_adds = max_parallel_adds
            self.memory = memory
            self.depth = depth
            self.mem_bd_in = [mem_bounded_in]
            self.mem_bd_out = [mem_bounded_out]

            config = [coarse_inout, mem_bw_in, mem_bw_out]
            self.config = config
            self.memoryKB = memKBs
            self.dsps_util = dsps_util
            self.bram_util = bram_util
            self.latency_sec = latency_sec
            self.latency_cycles = int(latency_cycles)
            self.throughput_ops = throughput_ops
            self.throughput_vols = thr_out

            if DEBUG:
                print("*"*40, "inout factor={} latency={} depth={}, max_parallel_muls={}".format(coarse_inout, int(latency_cycles), depth, dsps_util))
        else:
            self.update_layer()
            if DEBUG:
                print("Discarding design point.")

        return self.get_dp_info()

    def get_latency(self, params):
        coarse_inout, mem_bw_in, mem_bw_out = params
        if not (coarse_inout>0 and mem_bw_in>0 and mem_bw_out>0):
            return 1000000000000
        if (coarse_inout>1):
            return 1000000000000

        gamma_matrix = self.get_rate_matrix() * self.get_stream_matrix(coarse_inout) * self.get_data_matrix(mem_bw_in, mem_bw_out)
        if DEBUG:
            print("Γ:\n{}".format(gamma_matrix))
        gamma_matrix_balanced, mem_bounded_in, mem_bounded_out = self.balance_matrix(gamma_matrix.copy())
        if DEBUG:
            print("Γ Balanced:\n{}".format(gamma_matrix_balanced))
        workload_matrix = self.get_workload_matrix()
        ii_matrix = np.nan_to_num(workload_matrix/gamma_matrix_balanced)
        if DEBUG:
            print("II:\n{}".format(ii_matrix))

        max_parallel_muls = math.ceil(self.channels * coarse_inout)
        max_parallel_adds = math.ceil(self.channels * coarse_inout)
        memory = 1
        depth = 1

        latency_sec, latency_cycles, thr_in, thr_out, dsps_util, bram_util, memKBs = self.get_dp_performance(workload_matrix, ii_matrix, max_parallel_muls, max_parallel_adds, memory, depth)

        latency_cycles = np.max(np.abs(ii_matrix)) + depth
        penalize_factor = (bram_util/100 + dsps_util/100) * (np.max(np.abs(ii_matrix)) + depth)

        if self.optimization == 'Powell':
            optimization_score = latency_cycles + penalize_factor
        elif self.optimization == 'trust-constr':
            optimization_score = latency_cycles

        return optimization_score


    def get_rate_matrix(self):
        rate_matrix = np.zeros( shape=(2,3) , dtype=float )

        rate_matrix[0, 0] = 1
        
        rate_matrix[0, 1] = 1
        rate_matrix[1, 1] = 1

        rate_matrix[1, 2] = 1

        assert np.max(rate_matrix) <= 1 and np.min(rate_matrix[np.nonzero(rate_matrix)]) > 0, "Rate matrix issue"

        if DEBUG:
            print("R:\n{}".format(rate_matrix))
        return rate_matrix

    def get_stream_matrix(self, coarse_inout):
        stream_matrix = np.zeros( shape=(2,3) , dtype=float )

        stream_matrix[0, 0] = 1
    
        stream_matrix[0, 1] = math.ceil(self.channels * coarse_inout)
        stream_matrix[1, 1] = math.ceil(self.channels * coarse_inout)
        stream_matrix[1, 2] = 1

        if DEBUG:
            print("S:\n{}".format(stream_matrix))
        return stream_matrix

    def get_data_matrix(self, mem_bw_in, mem_bw_out):
        data_matrix = np.zeros( shape=(2,3) , dtype=float )

        data_matrix[0, 0] = mem_bw_in
        
        data_matrix[0, 1] = -1
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