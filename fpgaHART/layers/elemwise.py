import numpy as np
import math
import scipy.optimize as optimize
from scipy.optimize import NonlinearConstraint, Bounds
from ..layers.base_layer import BaseLayer
np.set_printoptions(precision=5, suppress=True, linewidth=150)
np.seterr(divide='ignore', invalid='ignore')

DEBUG=False

class ElementWiseLayer(BaseLayer):
    def __init__(self, description):
        super().__init__()

        # Available options 'C' channel parallelism, 'DC' channel AND depth parallelism
        self.parrallel_dims = 'C'
        self.type = description['operation']
        self.input_shape_1 = description['shape_in'][0]
        self.depth_in_1 = self.input_shape_1[2]
        self.rows_in_1 = self.input_shape_1[3]
        self.cols_in_1 = self.input_shape_1[4]
        self.input_shape_2 = description['shape_in'][1]
        self.depth_in_2 = self.input_shape_2[2]
        self.rows_in_2 = self.input_shape_2[3]
        self.cols_in_2 = self.input_shape_2[4]

        self.output_shape = description['shape_out']
        self.depth_out = self.output_shape[2]
        self.rows_out = self.output_shape[3]
        self.cols_out = self.output_shape[4]

        self.channels_1 = self.input_shape_1[1]
        self.channels_2 = self.input_shape_2[1]
        self.filters = self.output_shape[1]

        self.broadcasting = False if self.input_shape_1 == self.input_shape_2 else True
        if self.broadcasting:
            """
                Works only in case where the broadcasting is done in the channels dimension i.e. multiplication between 2 tensors of shape c X h X w X d * c X 1 X 1 X 1 resulting in a tensor of shape c X h X w X d
            """
            self.full_shape = self.input_shape_1 if int(np.prod(np.array(self.input_shape_1[1:]))) > int(np.prod(np.array(self.input_shape_2[1:]))) else self.input_shape_2
            self.reduced_shape = self.input_shape_1 if int(np.prod(np.array(self.input_shape_1[1:]))) < int(np.prod(np.array(self.input_shape_2[1:]))) else self.input_shape_2
            assert self.output_shape == self.full_shape, 'Elementwise layer ({}) input {} and output {} shapes does not match.'.format(self.type, self.full_shape, self.output_shape)
        else:
            self.full_shape = self.input_shape_1
            self.reduced_shape = self.input_shape_2
            assert self.output_shape == self.input_shape_1 and self.output_shape == self.input_shape_2, 'Elementwise layer ({}) input 1 {}, input 2 {} and output {} shapes does not match.'.format(self.type, self.input_shape_1, self.input_shape_2, self.output_shape)

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
        self.dsp_raw = 0
        self.bram_util = 0
        self.bram_raw = 0
        self.latency_sec = 0
        self.latency_cycles = 0
        self.throughput_ops = 0
        self.throughput_vols = 0

    def get_total_workload(self):
        if self.type == 'Add':
            return 1
        elif self.type == 'Mul':
            return int(np.prod(np.array(self.output_shape[1:])))
        elif self.type == 'Div':
            return int(np.prod(np.array(self.output_shape[1:]))) * 2

    def get_dp_info(self):
        dp_info = {}

        dp_info['latency(C)'] = self.latency_cycles
        dp_info['latency(S)'] = self.latency_sec
        dp_info['GOP/s'] = self.throughput_ops*1e-9
        dp_info['vols/s'] = self.throughput_vols
        dp_info['DSP'] = self.dsps_util
        dp_info['DSP_RAW'] = self.dsp_raw
        dp_info['BRAM'] = self.bram_util
        dp_info['BRAM_RAW'] = self.bram_raw
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
        if self.broadcasting:
            full_shape = self.full_shape
            reduced_shape = self.reduced_shape
        else:
            full_shape = self.input_shape_1
            reduced_shape = self.input_shape_1
        _, channels_1, depth_1, rows_1, cols_1 = full_shape
        _, channels_2, depth_2, rows_2, cols_2 = reduced_shape

        if self.parrallel_dims == 'C':
            self.max_streams_in_1 = channels_1
            self.max_streams_in_2 = channels_2
            self.max_streams_out = self.filters
        elif self.parrallel_dims == 'DC':
            self.max_streams_in_1 = channels_1 * depth_1
            self.max_streams_in_2 = channels_2 * depth_2
            self.max_streams_out = self.filters * self.depth_out
        return self.max_streams_in_1, self.max_streams_in_2, self.max_streams_out

    def get_design_point(self, coarse_inout, mem_bw_in_1, mem_bw_in_2, mem_bw_out):
        self.update_layer()

        gamma_matrix = self.get_rate_matrix() * self.get_stream_matrix(coarse_inout) * self.get_data_matrix(mem_bw_in_1, mem_bw_in_2, mem_bw_out)
        if DEBUG:
            print("Γ:\n{}".format(gamma_matrix))
        if self.broadcasting:
            # branch_ratio = 1/(int(np.prod(np.array(self.full_shape[2:]))))
            # gamma_matrix_balanced, mem_bounded_in_1, mem_bounded_in_2, mem_bounded_out = self.balance_matrix_elemwise_broadcasting(gamma_matrix.copy(), 2, branch_ratio)
            gamma_matrix_balanced, mem_bounded_in_1, mem_bounded_out = self.balance_matrix(gamma_matrix.copy())
            storage_latency, rate2_broadcast, mem_bounded_in_2 = self.get_storage_latency(coarse_inout, mem_bw_in_2)
        else:
            gamma_matrix_balanced, mem_bounded_in_1, mem_bounded_in_2, mem_bounded_out = self.balance_matrix_elemwise(gamma_matrix.copy(), 2)
        if DEBUG:
            print("Γ Balanced:\n{}".format(gamma_matrix_balanced))

        workload_matrix = self.get_workload_matrix()
        ii_matrix = np.nan_to_num(workload_matrix/gamma_matrix_balanced)
        if DEBUG:
            print("II:\n{}".format(ii_matrix))

        layer_fifos_arrays = {
            'elemwise_bc': 0
        }
        if self.broadcasting:
            depth = 2
            # layer_fifos_arrays['elemwise_bc'] = math.ceil(1/coarse_inout) + 1
            final_channel = self.full_shape[1]
            final_depth = self.full_shape[2]
            final_columns = self.full_shape[3]
            final_rows = self.full_shape[4]
        else:
            depth = 2
            final_channel = self.channels_1
            final_depth = self.depth_in_1
            final_columns = self.cols_in_1
            final_rows = self.rows_in_1
            
        if self.type == 'Add':
            if self.parrallel_dims == 'C':
                max_parallel_adds = math.ceil(final_channel  * coarse_inout)
                max_parallel_muls = 0
            elif self.parrallel_dims == 'DC':
                max_parallel_adds = math.ceil(final_channel * final_depth * coarse_inout)
                max_parallel_muls = 0
        elif self.type == 'Mul':
            if self.parrallel_dims == 'C':
                max_parallel_adds = 0
                max_parallel_muls = math.ceil(final_channel * coarse_inout)
            elif self.parrallel_dims == 'DC':
                max_parallel_adds = 0
                max_parallel_muls = math.ceil(final_channel * final_depth * coarse_inout)
        elif self.type == 'Div':
            if self.parrallel_dims == 'C':
                max_parallel_adds = 0
                max_parallel_muls = math.ceil(final_channel * coarse_inout) * 2
            elif self.parrallel_dims == 'DC':
                max_parallel_adds = 0
                max_parallel_muls = math.ceil(final_channel * final_depth * coarse_inout) * 2

        latency_sec, latency_cycles, thr_in, thr_out, dsps_util, dsp_raw, bram_util, bram_raw, memKBs = self.get_dp_performance(workload_matrix, ii_matrix, max_parallel_muls, max_parallel_adds, layer_fifos_arrays, depth, coarse_inout=math.ceil(final_channel  * coarse_inout))
        total_ops = self.get_total_workload()
        throughput_ops = total_ops/latency_sec
        thr_in /= workload_matrix[0,0]              # Volumes per second
        thr_out /= workload_matrix[-1,-1]           # Volumes per second
        assert math.isclose(thr_in, thr_out), "Thoughputs missmatch. IN = {}, OUT = {}.".format(thr_in, thr_out)

        if dsps_util < 90. and bram_util < 95.:

            self.full_rate_in.append(gamma_matrix_balanced[0, 0])
            if self.broadcasting:
                self.full_rate_in.append(rate2_broadcast)
            else:
                self.full_rate_in.append(gamma_matrix_balanced[1, 1])
            self.full_rate_out = [abs(gamma_matrix_balanced[-1, -1])]
            self.max_parallel_muls = max_parallel_muls
            self.max_parallel_adds = max_parallel_adds
            self.depth = depth
            self.mem_bd_in.append(mem_bounded_in_1)
            self.mem_bd_in.append(mem_bounded_in_2)
            self.mem_bd_out = [mem_bounded_out]

            config = [coarse_inout, mem_bw_in_1, mem_bw_in_2, mem_bw_out]
            self.config = config
            self.memoryKB = memKBs
            self.dsps_util = dsps_util
            self.dsp_raw = dsp_raw
            self.bram_util = bram_util
            self.bram_raw = bram_raw
            self.latency_sec = latency_sec
            self.latency_cycles = int(latency_cycles)
            self.throughput_ops = throughput_ops
            self.throughput_vols = thr_out

            if DEBUG:
                print("coarse_inout={}, bwin1={}, bwin2={}, bwout={}, latency={}, muls={}, dsps={}, boundin1={}, boundin2={}, boundout={}".format(coarse_inout, mem_bw_in_1, mem_bw_in_2, mem_bw_out, int(latency_cycles), max_parallel_muls, dsps_util, mem_bounded_in_1, mem_bounded_in_2, mem_bounded_out))
                print("*"*40)
        else:
            self.update_layer()
            if DEBUG:
                print("Discarding design point.")
        
        return self.get_dp_info()

    def get_rate_matrix(self):
        if self.broadcasting:
            rate_matrix = np.zeros( shape=(2,3) , dtype=float )

            rate_matrix[0, 0] = 1
            
            rate_matrix[0, 1] = 1
            rate_matrix[1, 1] = 1

            rate_matrix[1, 2] = 1
        else:
            rate_matrix = np.zeros( shape=(3,4) , dtype=float )

            rate_matrix[0, 0] = 1
            rate_matrix[0, 2] = 1
            
            rate_matrix[1, 1] = 1
            rate_matrix[1, 2] = 1

            if self.type == 'Add' or self.type == 'Mul':
                rate_matrix[2, 2] = 1
            elif self.type == 'Div':
                assert False, 'Currently not supporting ElementWise Division layer'
                exit()

            rate_matrix[2, 3] = 1

        assert np.max(rate_matrix) <= 1 and np.min(rate_matrix[np.nonzero(rate_matrix)]) > 0, "Rate matrix issue"
        if DEBUG:
            print("R:\n{}".format(rate_matrix))
        return rate_matrix

    def get_stream_matrix(self, coarse_inout):
        if self.broadcasting:
            stream_matrix = np.zeros( shape=(2,3) , dtype=float )

            stream_matrix[0, 0] = 1

            if self.parrallel_dims == 'C':
                stream_matrix[0, 1] = math.ceil(self.full_shape[1] * coarse_inout)
                stream_matrix[1, 1] = math.ceil(self.filters * coarse_inout)
            elif self.parrallel_dims == 'DC':
                stream_matrix[0, 1] = math.ceil(self.full_shape[1] * self.full_shape[2] * coarse_inout)
                stream_matrix[1, 1] = math.ceil(self.filters * self.depth_out * coarse_inout)
            stream_matrix[1, 2] = 1
        else:
            stream_matrix = np.zeros( shape=(3,4) , dtype=float )

            _, channels_1, depth_1, rows_1, cols_1 = self.input_shape_1
            _, channels_2, depth_2, rows_2, cols_2 = self.input_shape_2

            
            stream_matrix[0, 0] = 1
            if self.parrallel_dims == 'C':
                stream_matrix[0, 2] = math.ceil(channels_1 * coarse_inout)
            elif self.parrallel_dims == 'DC':
                stream_matrix[0, 2] = math.ceil(channels_1 * depth_1 * coarse_inout)

            stream_matrix[1, 1] = 1
            if self.parrallel_dims == 'C':
                stream_matrix[1, 2] = math.ceil(channels_2 * coarse_inout)
            elif self.parrallel_dims == 'DC':
                stream_matrix[1, 2] = math.ceil(channels_2 * depth_2 * coarse_inout)

            if self.parrallel_dims == 'C':
                stream_matrix[2, 2] = math.ceil(self.filters * coarse_inout)
            elif self.parrallel_dims == 'DC':
                stream_matrix[2, 2] = math.ceil(self.filters * self.depth_out * coarse_inout)

            stream_matrix[2, 3] = 1

        if DEBUG:
            print("S:\n{}".format(stream_matrix))
        return stream_matrix

    def get_data_matrix(self, mem_bw_in_1, mem_bw_in_2, mem_bw_out):
        if self.broadcasting:
            data_matrix = np.zeros( shape=(2,3) , dtype=float )

            data_matrix[0, 0] = mem_bw_in_1
            
            data_matrix[0, 1] = -1
            data_matrix[1, 1] = 1

            data_matrix[1, 2] = -mem_bw_out
        else:
            data_matrix = np.zeros( shape=(3,4) , dtype=float )

            data_matrix[0, 0] = mem_bw_in_1
            data_matrix[0, 2] = -1
            
            data_matrix[1, 1] = mem_bw_in_2
            data_matrix[1, 2] = -1

            data_matrix[2, 2] = 1
            data_matrix[2, 3] = -mem_bw_out

        if DEBUG:
            print("D:\n{}".format(data_matrix))
        return data_matrix
        
    def get_workload_matrix(self):
        in_volume_1 = self.depth_in_1 * self.rows_in_1 * self.cols_in_1 * self.channels_1
        in_volume_2 = self.depth_in_2 * self.rows_in_2 * self.cols_in_2 * self.channels_2
        out_volume = self.depth_out * self.rows_out * self.cols_out * self.filters

        if self.broadcasting:
            workload_matrix = np.zeros( shape=(2,3) , dtype=float )

            workload_matrix[0, 0] = self.full_shape[1] * self.full_shape[2] * self.full_shape[3] * self.full_shape[4]
            
            workload_matrix[0, 1] = self.full_shape[1] * self.full_shape[2] * self.full_shape[3] * self.full_shape[4]
            workload_matrix[1, 1] = out_volume

            workload_matrix[1, 2] = out_volume
        else:
            workload_matrix = np.zeros( shape=(3,4) , dtype=float )

            workload_matrix[0, 0] = in_volume_1
            workload_matrix[0, 2] = in_volume_1

            workload_matrix[1, 1] = in_volume_2
            workload_matrix[1, 2] = in_volume_2

            workload_matrix[2, 2] = out_volume
            workload_matrix[2, 3] = out_volume

        if DEBUG:
            print("WL:\n{}".format(workload_matrix))
        return workload_matrix

    def get_storage_latency(self, coarse_inout, mem_bw_in_2):
        gamma_matrix = np.zeros( shape=(2,3) , dtype=float )

        gamma_matrix[0, 0] = mem_bw_in_2
        
        gamma_matrix[0, 1] = -math.ceil(self.reduced_shape[1] * coarse_inout)
        gamma_matrix[1, 1] = math.ceil(self.reduced_shape[1] * coarse_inout)

        gamma_matrix[1, 2] = -1000000

        gamma_matrix_balanced, mem_bounded_in_2, mem_bounded_out = self.balance_matrix(gamma_matrix.copy())

        workload_matrix = np.zeros( shape=(2,3) , dtype=float )

        workload_matrix[0, 0] = self.reduced_shape[1]
        
        workload_matrix[0, 1] = self.reduced_shape[1]
        workload_matrix[1, 1] = self.reduced_shape[1]

        workload_matrix[1, 2] = self.reduced_shape[1]
        
        ii_matrix = np.nan_to_num(workload_matrix/gamma_matrix_balanced)

        latency_cycles = np.max(np.abs(ii_matrix))

        return int(latency_cycles), gamma_matrix_balanced[0,0], mem_bounded_in_2