from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.batchnorm_3d import BatchNorm3DLayer
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..layers.gap import GAPLayer
from ..layers.elemwise import ElementWiseLayer
from ..layers.activation import ActivationLayer
from ..layers.fully_connected import FCLayer
from ..layers.base_layer import BaseLayer
import numpy as np
import math

np.set_printoptions(precision=5, suppress=True, linewidth=150)
np.seterr(divide='ignore', invalid='ignore')

DEBUG=False
class PartitionComposer(BaseLayer):
    def __init__(self):
        super().__init__()

    def update_layer(self):
        self.full_rate_in_1 = 0
        self.full_rate_in_2 = 0
        self.full_rate_out = 0
        self.max_parallel_muls = 0
        self.max_parallel_adds = 0
        self.memory = 0
        self.memoryKB = 0
        self.depth = 0
        self.mem_bd_in_1 = False
        self.mem_bd_in_2 = False
        self.mem_bd_out = False
        self.config = []
        self.dsps_util = 0
        self.bram_util = 0
        self.latency_sec = 0
        self.latency_cycles = 0
        self.throughput_ops = 0
        self.throughput_vols = 0

    def get_total_workload(self, sequencial):
        total_wl = 0
        for n, l in enumerate(sequencial.keys()):
            total_wl += sequencial[l].get_total_workload()

        return total_wl

    def get_dp_info(self):
        dp_info = {}

        dp_info['latency(C)'] = self.latency_cycles
        dp_info['latency(S)'] = self.latency_sec
        dp_info['GOP/s'] = self.throughput_ops*1e-9
        dp_info['vols/s'] = self.throughput_vols
        dp_info['DSP'] = self.dsps_util
        dp_info['BRAM'] = self.bram_util
        dp_info['rateIn1'] = self.full_rate_in_1
        dp_info['rateIn2'] = self.full_rate_in_2
        dp_info['rateOut'] = self.full_rate_out
        dp_info['depth'] = self.depth
        dp_info['muls'] = self.max_parallel_muls
        dp_info['adds'] = self.max_parallel_adds
        dp_info['memWords'] = self.memory
        dp_info['memKBs'] = self.memoryKB
        dp_info['memBoundedIn1'] = self.mem_bd_in_1
        dp_info['memBoundedIn2'] = self.mem_bd_in_2
        dp_info['memBoundedOut'] = self.mem_bd_out
        dp_info['config'] = self.config
        
        return dp_info

    def get_design_point(self, sequencial, comb, mem_bw_in_1, mem_bw_in_2, mem_bw_out):
        self.update_layer()

        mem_bw_in_1, mem_bw_in_2, mem_bw_out = mem_bw_in_1*self.mem_words_per_cycle, mem_bw_in_2*self.mem_words_per_cycle, mem_bw_out*self.mem_words_per_cycle
        num_layers = len(sequencial) + 2

        gamma_matrix = np.zeros( shape=(num_layers-1, num_layers) , dtype=float )
        gamma_matrix[0, 0] = mem_bw_in_1
        gamma_matrix[-1, -1] = -mem_bw_out

        total_muls = 0
        total_adds = 0
        total_memory = 0
        total_depth = 0
        first_layer_bw_in = False
        prev_layer_rate = mem_bw_in_1

        for n, (l, c) in enumerate(zip(sequencial.values(), comb)):
            if n == len(sequencial) - 1:
                curr_layer_rate = mem_bw_out
            else:
                curr_layer_rate = 10000000
            
            if isinstance(l, GAPLayer):
                dp_info = l.get_design_point(c[0], c[1], prev_layer_rate, curr_layer_rate)
            elif isinstance(l, Convolutional3DLayer):
                dp_info = l.get_design_point(c[0], c[1], c[2], prev_layer_rate, curr_layer_rate)
            elif isinstance(l, ActivationLayer):
                dp_info = l.get_design_point(c[0], prev_layer_rate, curr_layer_rate)
            elif isinstance(l, ElementWiseLayer):
                #TODO: Check this how to deal when the input comes from another layer and not from off-chip mem
                dp_info = l.get_design_point(c[0], c[1], c[2], mem_bw_in_2, prev_layer_rate, curr_layer_rate)
            elif isinstance(l, BatchNorm3DLayer):
                dp_info = l.get_design_point(c[0], prev_layer_rate, curr_layer_rate)
            elif isinstance(l, SqueezeExcitationLayer):
                dp_info = l.get_design_point(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], prev_layer_rate, curr_layer_rate)
            elif isinstance(l, FCLayer):
                pass
                # dp_info = l.get_design_point(f_mul_coarsein1, f_mul_coarsein2, f_mul_coarseout, mem_bw_in_1, prev_layer_rate, curr_layer_rate)
            else:
                assert False, "Not supported layer"
            

            if isinstance(l, ElementWiseLayer):
                full_rate_in_1, full_rate_in_2, full_rate_out, muls, adds, memory, depth, mem_bd_in_1, mem_bd_in_2, mem_bd_out = dp_info['rateIn1'], dp_info['rateIn2'], dp_info['rateOut'], dp_info['muls'], dp_info['adds'], dp_info['memWords'], dp_info['depth'], dp_info['memBoundedIn1'], dp_info['memBoundedIn2'], dp_info['memBoundedOut']
                # gamma_matrix[0, n+1] = -full_rate_in_1
                gamma_matrix[n, n+1] = -full_rate_in_2
                gamma_matrix[n+1, n+1] = full_rate_out
            else:
                full_rate_in, full_rate_out, muls, adds, memory, depth, mem_bd_in, mem_bd_out = dp_info['rateIn1'], dp_info['rateOut'], dp_info['muls'], dp_info['adds'], dp_info['memWords'], dp_info['depth'], dp_info['memBoundedIn1'], dp_info['memBoundedOut']
                gamma_matrix[n, n+1] = -full_rate_in
                gamma_matrix[n+1, n+1] = full_rate_out

            prev_layer_rate = full_rate_out

            total_muls += muls
            total_adds += adds
            total_memory += memory
            total_depth += depth

            mem_kb = (total_memory * self.word_bytes) / 1e3
            mem_bram = math.ceil(mem_kb / self.bram_bytes)
            curr_bram_util = (mem_bram / self.bram) * 100
            curr_dsps_util = (total_muls/self.dsp)*100

            if not dp_info['config'] or curr_dsps_util >= 90. or curr_bram_util >= 90.:
                self.update_layer()
                if DEBUG:
                    print("Discarding design point.")
                return self.get_dp_info()

        if DEBUG:
            print("Γ:\n{}".format(gamma_matrix))
        gamma_matrix_balanced, mem_bounded_in, mem_bounded_out = self.balance_matrix(gamma_matrix.copy())
        if DEBUG:
            print("Γ Balanced:\n{}".format(gamma_matrix_balanced))
        workload_matrix = self.get_workload_matrix(sequencial, num_layers)
        if DEBUG:
            print("WL:\n{}".format(workload_matrix))
        ii_matrix = np.nan_to_num(workload_matrix/gamma_matrix_balanced)
        if DEBUG:
            print("II:\n{}".format(ii_matrix))

        mem_bounded_in = mem_bounded_in or first_layer_bw_in
        latency_sec, latency_cycles, thr_in, thr_out, dsps_util, bram_util, memKBs = self.get_performance(workload_matrix, ii_matrix, total_muls, total_adds, total_memory, total_depth)
        total_ops = self.get_total_workload(sequencial)
        throughput_ops = total_ops/latency_sec
        thr_in /= workload_matrix[0,0]              # Volumes per second
        thr_out /= workload_matrix[-1,-1]           # Volumes per second
        assert math.isclose(thr_in, thr_out), "Thoughputs missmatch. IN = {}, OUT = {}.".format(thr_in, thr_out)

        if dsps_util < 90. and bram_util < 90.:
            #TODO: Add 2nd input
            self.full_rate_in_1 = gamma_matrix_balanced[0, 0]
            self.full_rate_in_2 = gamma_matrix_balanced[1, 1]
            self.full_rate_out = abs(gamma_matrix_balanced[-1, -1])
            self.max_parallel_muls = total_muls
            self.max_parallel_adds = total_adds
            self.memory = total_memory
            self.depth = total_depth
            #TODO: Add 2nd input
            self.mem_bd_in_1 = mem_bounded_in
            self.mem_bd_in_2 = mem_bounded_in
            self.mem_bd_out = mem_bounded_out

            #TODO: Add 2nd input
            config = comb
            self.config = config
            self.memoryKB = memKBs
            self.dsps_util = dsps_util
            self.bram_util = bram_util
            self.latency_sec = latency_sec
            self.latency_cycles = int(latency_cycles)
            self.throughput_ops = throughput_ops
            self.throughput_vols = thr_out

            if DEBUG:
                print("GOPs/s={:.2f}, DSPS={:.2f}, BRAM={:.2f}, depth={}, latency(s)={:.2f}, latency(c)={:.2f}, mem bounded in = {}, mem bounded out = {}".format(throughput_ops*1e-9, dsps_util, bram_util, total_depth, latency_sec, int(latency_cycles), mem_bounded_in, mem_bounded_out))
        else:
            self.update_layer()
            if DEBUG:
                print("Discarding design point.")

        return self.get_dp_info()

    def get_workload_matrix(self, sequencial, num_layers):
        #TODO: Add an extra connection to the graph for the 2nd input of MUL operation
        workload_matrix = np.zeros( shape=(num_layers-1, num_layers) , dtype=float )

        for n, l in enumerate(sequencial.keys()):

            if n == 0:
                workload_matrix[0, 0] = np.prod(np.array(sequencial[l].input_shape[1:]))
            if n == len(sequencial) - 1:
                workload_matrix[-1, -1] = np.prod(np.array(sequencial[l].output_shape[1:]))
            
            if isinstance(sequencial[l], ElementWiseLayer):
                # workload_matrix[0, n+1] = np.prod(np.array(self.sequencial[l].input_shape_1[1:]))
                workload_matrix[n, n+1] = np.prod(np.array(sequencial[l].input_shape_2[1:]))
            else:
                workload_matrix[n, n+1] = np.prod(np.array(sequencial[l].input_shape[1:]))
            workload_matrix[n+1, n+1] = np.prod(np.array(sequencial[l].output_shape[1:]))
        return workload_matrix

    def get_performance(self, workload_matrix, ii, muls, adds, mem, depth):
        mem_kb = (mem * self.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.bram_bytes)
        bram_util = (mem_bram / self.bram) * 100

        dsps_util = (muls/self.dsp)*100

        latency_cycles = np.max(np.abs(ii)) + depth
        latency_sec = latency_cycles/self.cycles_per_sec

        thr_in = workload_matrix[0,0]/latency_sec       # Input words per second
        thr_out = workload_matrix[-1,-1]/latency_sec    # Output words per second

        return latency_sec, latency_cycles, thr_in, thr_out, dsps_util, bram_util, mem_kb