import numpy as np
import math
import scipy.optimize as optimize
from scipy.optimize import NonlinearConstraint, Bounds
from .base_layer import BaseLayer
np.set_printoptions(precision=5, suppress=True, linewidth=150)
np.seterr(divide='ignore', invalid='ignore')

DEBUG=False

class Convolutional3DLayer(BaseLayer):
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

    def update_layer(self):
        self.channels = self.input_shape[1]
        self.full_rate_in = 0
        self.full_rate_out = 0
        self.max_parallel_muls = 0
        self.max_parallel_adds = 0
        self.memory = 0
        self.memoryKB = 0
        self.depth = 0
        self.mem_bd_in = False
        self.mem_bd_out = False
        self.config = []
        self.dsps_util = 0
        self.bram_util = 0
        self.latency_sec = 0
        self.latency_cycles = 0
        self.throughput_ops = 0
        self.throughput_vols = 0

    def get_total_workload(self):
        return self.depth_out*self.rows_out*self.cols_out*self.kd*self.kw*self.kh*self.channels*self.filters

    def get_dp_info(self):
        dp_info = {}

        dp_info['latency(C)'] = self.latency_cycles
        dp_info['latency(S)'] = self.latency_sec
        dp_info['GOP/s'] = self.throughput_ops*1e-9
        dp_info['vols/s'] = self.throughput_vols
        dp_info['DSP'] = self.dsps_util
        dp_info['BRAM'] = self.bram_util
        dp_info['rateIn1'] = self.full_rate_in
        dp_info['rateIn2'] = -1
        dp_info['rateOut'] = self.full_rate_out
        dp_info['depth'] = self.depth
        dp_info['muls'] = self.max_parallel_muls
        dp_info['adds'] = self.max_parallel_adds
        dp_info['memWords'] = self.memory
        dp_info['memKBs'] = self.memoryKB
        dp_info['memBoundedIn1'] = self.mem_bd_in
        dp_info['memBoundedIn2'] = -1
        dp_info['memBoundedOut'] = self.mem_bd_out
        dp_info['config'] = self.config
        
        return dp_info

    def get_design_point(self, f_fine, f_coarseIn, f_coarseOut, mem_bw_in, mem_bw_out):
        self.update_layer()

        kernel_elems = int(np.prod(np.array(self.kernel_shape)))

        if self.depthwise:
            self.channels = self.channels//self.groups

        if self.optimization == 'Powell':
            initial_guess = [0.5, 0.5, 0.5, mem_bw_in, mem_bw_out]
            bnds = ((0.01, 1.0), (0.01, 1.0), (0.01, 1.0), (0.01, mem_bw_in), (0.01, mem_bw_out))   
            result = optimize.minimize(self.get_latency, initial_guess, method=self.optimization, bounds=bnds)
            if result.success:
                f_fine, f_coarseIn, f_coarseOut, mem_bw_in, mem_bw_out = result.x
            else:
                print("Failed to optimize. Skipping...")
                return
        elif self.optimization == 'trust-constr':
            initial_guess = [0.5, 0.5, 0.5, mem_bw_in, mem_bw_out]
            bnds = Bounds([0.001, 0.001, 0.001, 0.001, 0.001], [1.0, 1.0, 1.0, mem_bw_in, mem_bw_out])  
            cons_f = lambda x: np.array([((math.ceil(kernel_elems * x[0]) * math.ceil(self.channels * x[1]) * math.ceil(self.filters * x[2]))/self.dsp) * 100 + 0*x[3] + 0*x[4]])
            cons_J = lambda x: np.array([[((kernel_elems * math.ceil(self.channels * x[1]) * math.ceil(self.filters * x[2]))/8)*100, ((math.ceil(kernel_elems * x[0]) * self.channels * math.ceil(self.filters * x[2]))/8)*100, ((math.ceil(kernel_elems * x[0]) * math.ceil(self.channels * x[1]) * self.filters)/8)*100]])
            non_linear_constraint = NonlinearConstraint(cons_f, 0.001, 90.)
            result = optimize.minimize(self.get_latency, initial_guess, method=self.optimization, constraints=[non_linear_constraint], bounds=bnds)
            if result.success:
                f_fine, f_coarseIn, f_coarseOut, mem_bw_in, mem_bw_out = result.x
            else:
                print("Failed to optimize. Skipping...")
                return

        if self.pointwise:
            # Sliding Window Depth
            depth = 1
            # Convolution 3D Depth
            depth += 1
            # Sliding Window buffer in words/elements
            init_buffer = 1
        else:
            # Sliding Window Depth
            depth = self.depth_in * self.rows_in * max(self.kw - 1, 1) * math.ceil(self.channels * f_coarseIn) + math.ceil(self.channels * f_coarseIn) * self.kd * self.kh * max(self.kw - 1, 1)
            # Convolution 3D Depth
            depth += math.ceil(1/f_fine)
            # Sliding Window buffer in words/elements
            init_buffer = self.depth_in * self.rows_in * max(self.kw - 1, 1) * self.channels + self.channels * self.kd * self.kh * max(self.kw - 1, 1)

        max_parallel_muls = math.ceil(kernel_elems * f_fine) * math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
        max_parallel_adds = math.ceil((kernel_elems - 1) * f_fine) * math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
        memory = init_buffer + kernel_elems * self.channels * self.filters

        if not self.depthwise:
            # Accumulation Depth
            depth += (math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut))
            # Accumulation Additions
            max_parallel_adds += ((math.ceil(self.channels * f_coarseIn) - 1) * math.ceil(self.filters * f_coarseOut))
            # Accumulation Buffer
            memory += self.channels

        gamma_matrix = self.get_rate_matrix(f_fine) * self.get_stream_matrix(f_coarseIn, f_coarseOut) * self.get_data_matrix(mem_bw_in, mem_bw_out)
        if DEBUG:
            print("Γ:\n{}".format(gamma_matrix))
        gamma_matrix_balanced, mem_bounded_in, mem_bounded_out = self.balance_matrix(gamma_matrix.copy())
        if DEBUG:
            print("Γ Balanced:\n{}".format(gamma_matrix_balanced))
        workload_matrix = self.get_workload_matrix()
        ii_matrix = np.nan_to_num(workload_matrix/gamma_matrix_balanced)
        if DEBUG:
            print("II:\n{}".format(ii_matrix))

        latency_sec, latency_cycles, thr_in, thr_out, dsps_util, bram_util, memKBs = self.get_dp_performance(workload_matrix, ii_matrix, max_parallel_muls, max_parallel_adds, memory, depth)

        total_ops = self.get_total_workload()
        throughput_ops = total_ops/latency_sec
        thr_in /= (np.prod(np.array(self.input_shape[2:])) * self.channels)         # Volumes per second
        thr_out /= (np.prod(np.array(self.output_shape[2:])) * self.filters)        # Volumes per second
        assert math.isclose(thr_in, thr_out), "Thoughputs missmatch. IN = {}, OUT = {}.".format(thr_in, thr_out)

        if dsps_util < 90. and bram_util < 90.:
            
            self.full_rate_in = gamma_matrix_balanced[0, 0]
            self.full_rate_out = abs(gamma_matrix_balanced[-1, -1])
            self.max_parallel_muls = max_parallel_muls
            self.max_parallel_adds = max_parallel_adds
            self.memory = memory
            self.depth = depth
            self.mem_bd_in = mem_bounded_in
            self.mem_bd_out = mem_bounded_out

            config = [f_fine, f_coarseIn, f_coarseOut, mem_bw_in, mem_bw_out]
            self.config = config
            self.memoryKB = memKBs
            self.dsps_util = dsps_util
            self.bram_util = bram_util
            self.latency_sec = latency_sec
            self.latency_cycles = int(latency_cycles)
            self.throughput_ops = throughput_ops
            self.throughput_vols = thr_out

            if DEBUG:
                print("(fine={:.2f}({}), cIn={:.2f}({}), cOut={:.2f}({}), bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f} ({}), BRAM % = {:.2f}, latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(f_fine, math.ceil(f_fine*kernel_elems), f_coarseIn, math.ceil(self.channels * f_coarseIn), f_coarseOut, math.ceil(self.filters * f_coarseOut), mem_bw_in, mem_bw_out, dsps_util, max_parallel_muls, bram_util, latency_sec, int(latency_cycles), throughput_ops, thr_in, thr_out, depth, total_ops, mem_bounded_in, mem_bounded_out))
        else:
            self.update_layer()
            if DEBUG:
                print("Discarding design point.")

        return self.get_dp_info()

    def get_latency(self, params):
        f_fine, f_coarseIn, f_coarseOut, mem_bw_in, mem_bw_out = params
        if not (f_fine>0 and f_coarseIn>0 and f_coarseOut>0 and mem_bw_in>0 and mem_bw_out>0):
            return 1000000000000
        if (f_fine>1 or f_coarseIn>1 or f_coarseOut>1):
            return 1000000000000

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
            depth = self.depth_in * self.rows_in * max(self.kw - 1, 1) * math.ceil(self.channels * f_coarseIn) + math.ceil(self.channels * f_coarseIn) * self.kd * self.kh * max(self.kw - 1, 1)
            # Convolution 3D Depth
            depth += math.ceil(1/f_fine)
            # Sliding Window buffer in words/elements
            init_buffer = self.depth_in * self.rows_in * max(self.kw - 1, 1) * self.channels + self.channels * self.kd * self.kh * max(self.kw - 1, 1)

        max_parallel_muls = math.ceil(kernel_elems * f_fine) * math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
        max_parallel_adds = math.ceil((kernel_elems - 1) * f_fine) * math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
        memory = init_buffer + kernel_elems * self.channels * self.filters

        if not self.depthwise:
            # Accumulation Depth
            depth += (math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut))
            # Accumulation Additions
            max_parallel_adds += ((math.ceil(self.channels * f_coarseIn) - 1) * math.ceil(self.filters * f_coarseOut))
            # Accumulation Buffer
            memory += self.channels

        mem_kb = (memory * self.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.bram_bytes)
        bram_util = (mem_bram / self.bram) * 100
        dsps_util = (max_parallel_muls/self.dsp)*100

        gamma_matrix = self.get_rate_matrix(f_fine) * self.get_stream_matrix(f_coarseIn, f_coarseOut) * self.get_data_matrix(mem_bw_in, mem_bw_out)
        gamma_matrix_balanced, mem_bounded_in, mem_bounded_out = self.balance_matrix(gamma_matrix.copy())
        workload_matrix = self.get_workload_matrix()
        ii_matrix = np.nan_to_num(workload_matrix/gamma_matrix_balanced)
        
        latency_cycles = np.max(np.abs(ii_matrix)) + depth
        penalize_factor = (bram_util/100 + dsps_util/100) * (np.max(np.abs(ii_matrix)) + depth)

        if self.optimization == 'Powell':
            optimization_score = latency_cycles + penalize_factor
        elif self.optimization == 'trust-constr':
            optimization_score = latency_cycles

        return optimization_score

    def get_rate_matrix(self, f_fine):
        if not self.depthwise:
            rate_matrix = np.zeros( shape=(5,6) , dtype=float )
        else:
            rate_matrix = np.zeros( shape=(4,5) , dtype=float )

        rate_matrix[0, 0] = 1
        
        rate_matrix[0, 1] = 1
        rate_matrix[1, 1] = (self.depth_out*self.rows_out*self.cols_out)/(self.depth_in*self.rows_in*self.cols_in)

        rate_matrix[1, 2] = 1
        rate_matrix[2, 2] = 1

        rate_matrix[2, 3] = f_fine
        rate_matrix[3, 3] = f_fine

        if not self.depthwise:
            rate_matrix[3, 4] = 1
            rate_matrix[4, 4] = 1

            rate_matrix[4, 5] = 1
        else:
            rate_matrix[3, 4] = 1

        assert np.max(rate_matrix) <= 1 and np.min(rate_matrix[np.nonzero(rate_matrix)]) > 0, "Rate matrix issue"
        if DEBUG:
            print("R:\n{}".format(rate_matrix))
        return rate_matrix

    def get_stream_matrix(self, f_coarseIn, f_coarseOut):
        if not self.depthwise:
            stream_matrix = np.zeros( shape=(5,6) , dtype=float )
        else:
            stream_matrix = np.zeros( shape=(4,5) , dtype=float )

        stream_matrix[0, 0] = 1
        
        stream_matrix[0, 1] = math.ceil(self.channels * f_coarseIn)
        stream_matrix[1, 1] = math.ceil(self.channels * f_coarseIn)

        stream_matrix[1, 2] = math.ceil(self.channels * f_coarseIn)
        stream_matrix[2, 2] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)

        stream_matrix[2, 3] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
        stream_matrix[3, 3] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)

        if not self.depthwise:
            stream_matrix[3, 4] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
            stream_matrix[4, 4] = math.ceil(self.filters * f_coarseOut)

            stream_matrix[4, 5] = 1
        else:
            stream_matrix[3, 4] = 1
        if DEBUG:
            print("S:\n{}".format(stream_matrix))
        return stream_matrix

    def get_data_matrix(self, mem_bw_in, mem_bw_out):
        if not self.depthwise:
            data_matrix = np.zeros( shape=(5,6) , dtype=float )
        else:
            data_matrix = np.zeros( shape=(4,5) , dtype=float )

        data_matrix[0, 0] = mem_bw_in
        
        data_matrix[0, 1] = -1
        data_matrix[1, 1] = self.kd * self.kw * self.kh

        data_matrix[1, 2] = -self.kd * self.kw * self.kh
        data_matrix[2, 2] = self.kd * self.kw * self.kh

        data_matrix[2, 3] = -self.kd * self.kw * self.kh
        data_matrix[3, 3] = 1

        if not self.depthwise:
            data_matrix[3, 4] = -1
            data_matrix[4, 4] = 1

            data_matrix[4, 5] = -mem_bw_out
        else:
            data_matrix[3, 4] = -mem_bw_out

        if DEBUG:
            print("D:\n{}".format(data_matrix))
        return data_matrix

    def get_workload_matrix(self):
        in_volume = self.depth_in * self.rows_in * self.cols_in
        out_volume = self.depth_out * self.rows_out * self.cols_out
        kernel_volume = self.kd * self.kw * self.kh

        if not self.depthwise:
            workload_matrix = np.zeros( shape=(5,6) , dtype=float )
        else:
            workload_matrix = np.zeros( shape=(4,5) , dtype=float )

        workload_matrix[0, 0] = in_volume * self.channels
        
        workload_matrix[0, 1] = in_volume * self.channels
        workload_matrix[1, 1] = out_volume * kernel_volume * self.channels

        workload_matrix[1, 2] = out_volume * kernel_volume * self.channels
        workload_matrix[2, 2] = out_volume * kernel_volume * self.channels * self.filters

        workload_matrix[2, 3] = out_volume * kernel_volume * self.channels * self.filters
        workload_matrix[3, 3] = out_volume * self.channels * self.filters

        if not self.depthwise:
            workload_matrix[3, 4] = out_volume * self.channels * self.filters
            workload_matrix[4, 4] = out_volume * self.filters

            workload_matrix[4, 5] = out_volume * self.filters
        else:
            workload_matrix[3, 4] = out_volume * self.filters     

        if DEBUG:
            print("WL:\n{}".format(workload_matrix))
        return workload_matrix