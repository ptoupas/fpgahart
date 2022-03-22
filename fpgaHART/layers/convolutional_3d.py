import numpy as np
import math
import scipy.optimize as optimize
from scipy.optimize import NonlinearConstraint, Bounds
from .base_layer import BaseLayer
np.set_printoptions(precision=5, suppress=True, linewidth=150)
np.seterr(divide='ignore', invalid='ignore')

DEBUG=False

class Convolutional3DLayer(BaseLayer):
    def __init__(self, description):
        super().__init__()

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
        if self.groups == self.channels and self.channels == self.filters:
            self.depthwise = True
        elif np.prod(np.array(self.kernel_shape)) == 1:
            self.pointwise = True

    def update_layer(self):
        self.channels = self.input_shape[1]
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
        if not self.depthwise:
            return self.depth_out*self.rows_out*self.cols_out*self.kd*self.kw*self.kh*self.channels*self.filters
        else:
            return self.depth_out*self.rows_out*self.cols_out*self.kd*self.kh*self.kw*self.channels

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
        self.max_streams_in = self.channels
        self.max_streams_out = self.filters
        return self.max_streams_in, self.max_streams_out

    def get_design_point(self, f_fine, f_coarseIn, f_coarseOut, mem_bw_in, mem_bw_out):
        self.update_layer()

        if self.depthwise:
            assert self.channels == self.filters and self.channels == self.groups, 'Depthwise convolutional layer must have groups equal to channels and filters'
            # self.channels = self.channels//self.groups

        kernel_elems = int(np.prod(np.array(self.kernel_shape)))

        layer_fifos_arrays = {
            'sw_lb_3d': 0,
            'sw_lb_2d': 0,
            'sw_wb_3d': 0,
            'acc_fifo': 0,
            'acc_array': 0,
        }

        depth = 2
        if self.pointwise:
            # Sliding Window Module (SWM) Depth and Memory
            depth += 1

            # Fork Module (FM) Depth and Memory
            
            # Convolution Module (CM) Depth and Memory
            depth += math.ceil(1/f_fine) + 1

            # Accumulator Module (AM) Depth and Memory

            # Glue Module (GM) Depth and Memory
        else:
            # Sliding Window Module (SWM) Depth and Memory
            first_time_read_input = self.padding[1] * math.ceil(1/f_coarseIn) * (self.cols_in+2*self.padding[2]) * (self.depth_in+2*self.padding[0]) +\
                                    self.padding[2] * math.ceil(1/f_coarseIn) * (self.depth_in+2*self.padding[0]) +\
                                    self.padding[0] * math.ceil(1/f_coarseIn) + 1    

            depth_line_buffer_3d = math.ceil(1/f_coarseIn) * (self.depth_in+2*self.padding[0]) + 1
            layer_fifos_arrays['sw_lb_3d'] = depth_line_buffer_3d

            depth_line_buffer_2d = math.ceil(1/f_coarseIn) * ((self.depth_in+2*self.padding[0])*(self.cols_in+2*self.padding[2]) - (self.kw-1)*self.depth_in - (self.kd-1)) + 1
            layer_fifos_arrays['sw_lb_2d'] = depth_line_buffer_2d

            depth_window_buffer_3d = math.ceil(1/f_coarseIn) + 1
            layer_fifos_arrays['sw_wb_3d'] = depth_window_buffer_3d

            # DEPTH V1
            depth += math.ceil(1/f_coarseIn) * (self.cols_in + 2*self.padding[2]) * (self.depth_in + 2*self.padding[0]) * (self.kh - 1) +\
                    math.ceil(1/f_coarseIn) * (self.depth_in + 2*self.padding[0]) * (self.kw - 1) +\
                    math.ceil(1/f_coarseIn) * (self.kd - 1)
            depth += math.ceil(1/f_coarseIn) * ( (self.kh - 1)*self.kw*self.kd + (self.kw - 1)*self.kd + (self.kd - 1) )

            # DEPTH V2
            # depth += self.kh*(self.kw-1) * depth_line_buffer_3d + (self.kh-1) * depth_line_buffer_2d + self.kh*self.kw*(self.kd-1) * depth_window_buffer_3d
            
            # DEPTH V3
            # depth_ = math.ceil(1/f_coarseIn) * (self.cols_in + 2*self.padding[2]) * (self.depth_in + 2*self.padding[0]) * (self.kh - 1) +\
            #         math.ceil(1/f_coarseIn) * (self.depth_in + 2*self.padding[0]) * (self.kw - 1) +\
            #         math.ceil(1/f_coarseIn) * (self.kd - 1)
            # depth += depth_ - first_time_read_input - (self.cols_in - 1) * math.ceil(1/f_coarseIn) * (self.kd - 1) - math.ceil(1/f_coarseIn) * (self.depth_in + 2*self.padding[0]) * (self.kw - 1)

            # Fork Module (FM) Depth and Memory

            # Convolution Module (CM) Depth and Memory
            depth += math.ceil(1/f_fine) + 1
            
            # Accumulator Module (AM) Depth and Memory

            # Glue Module (GM) Depth and Memory

        if not self.depthwise:
            # Accumulator Module (AM) Depth and Memory
            depth_accumulator = math.ceil(1/f_coarseOut) + 1
            layer_fifos_arrays['acc_fifo'] = depth_accumulator
            
            # Accumulation Buffer
            array_accumulator = math.ceil(1/f_coarseOut)
            layer_fifos_arrays['acc_array'] = array_accumulator

            # DEPTH V1
            # depth += depth_accumulator
            
            # DEPTH V2
            depth += math.ceil(1/f_coarseOut) + 1
            
            max_parallel_muls = math.ceil(kernel_elems * f_fine) * math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
            max_parallel_adds = math.ceil((kernel_elems - 1) * f_fine) * math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)

        else:
            max_parallel_muls = math.ceil(kernel_elems * f_fine) * math.ceil(self.channels * f_coarseIn)
            max_parallel_adds = math.ceil((kernel_elems - 1) * f_fine) * math.ceil(self.channels * f_coarseIn)


        gamma_matrix = self.get_rate_matrix(f_fine) * self.get_stream_matrix(f_coarseIn, f_coarseOut) * self.get_data_matrix(mem_bw_in, mem_bw_out, f_fine)
        if DEBUG:
            print("Γ:\n{}".format(gamma_matrix))
        gamma_matrix_balanced, mem_bounded_in, mem_bounded_out = self.balance_matrix(gamma_matrix.copy())
        if DEBUG:
            print("Γ Balanced:\n{}".format(gamma_matrix_balanced))
        workload_matrix = self.get_workload_matrix()
        #TODO: Investigate whether we need to use the balancing of the matrix or not
        ii_matrix = np.nan_to_num(workload_matrix/gamma_matrix_balanced)
        if DEBUG:
            print("II:\n{}".format(ii_matrix))

        if not self.depthwise:
            latency_sec, latency_cycles, thr_in, thr_out, dsps_util, dsp_raw, bram_util, bram_raw, memKBs = self.get_dp_performance(workload_matrix, ii_matrix, max_parallel_muls, max_parallel_adds, layer_fifos_arrays, depth, kernel_shape=[self.filters, self.channels, self.kd, self.kh, self.kw], coarse_in=math.ceil(self.channels * f_coarseIn), coarse_out=math.ceil(self.filters * f_coarseOut), fine=math.ceil(kernel_elems * f_fine))
        else:
            latency_sec, latency_cycles, thr_in, thr_out, dsps_util, dsp_raw, bram_util, bram_raw, memKBs = self.get_dp_performance(workload_matrix, ii_matrix, max_parallel_muls, max_parallel_adds, layer_fifos_arrays, depth, kernel_shape=[self.filters/self.groups, self.channels, self.kd, self.kh, self.kw], coarse_in=math.ceil(self.channels * f_coarseIn), coarse_out=1, fine=math.ceil(kernel_elems * f_fine))
        
        total_ops = self.get_total_workload()
        throughput_ops = total_ops/latency_sec
        thr_in /= (np.prod(np.array(self.input_shape[2:])) * self.channels)         # Volumes per second
        thr_out /= (np.prod(np.array(self.output_shape[2:])) * self.filters)        # Volumes per second
        assert math.isclose(thr_in, thr_out), "Thoughputs missmatch. IN = {}, OUT = {}.".format(thr_in, thr_out)
        
        if DEBUG:
            print(f"Fine: {f_fine:.3f} ({f_fine*np.prod(np.array(self.kernel_shape))}), CoarseIn: {f_coarseIn:.3f} ({int(f_coarseIn*self.channels)}), CoarseOut: {f_coarseOut:.3f} ({int(f_coarseOut*self.filters)}), Shape in: {self.input_shape}, Shape out: {self.output_shape}, Kernel: {self.kernel_shape}")
        if dsps_util < 90. and bram_util < 95.:
            
            self.full_rate_in = [gamma_matrix_balanced[0, 0]]
            self.full_rate_out = [abs(gamma_matrix_balanced[-1, -1])]
            self.max_parallel_muls = max_parallel_muls
            self.max_parallel_adds = max_parallel_adds
            self.depth = depth
            self.mem_bd_in = [mem_bounded_in]
            self.mem_bd_out = [mem_bounded_out]

            config = [f_fine, f_coarseIn, f_coarseOut, mem_bw_in, mem_bw_out, f_fine*kernel_elems, f_coarseIn*self.channels, f_coarseOut*self.filters]
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
                print("(fine={:.2f}({}), cIn={:.2f}({}), cOut={:.2f}({}), bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f} ({}), BRAM % = {:.2f} ({}), latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(f_fine, math.ceil(f_fine*kernel_elems), f_coarseIn, math.ceil(self.channels * f_coarseIn), f_coarseOut, math.ceil(self.filters * f_coarseOut), mem_bw_in, mem_bw_out, dsps_util, dsp_raw, bram_util, bram_raw, latency_sec, int(latency_cycles), throughput_ops, thr_in, thr_out, depth, total_ops, mem_bounded_in, mem_bounded_out))
        else:
            self.update_layer()
            if DEBUG:
                print("Discarding design point. DSP = {:.2f} - BRAM = {:.2f}".format(dsps_util, bram_util))

        return self.get_dp_info()

    def get_rate_matrix(self, f_fine):
        if self.depthwise:
            rate_matrix = np.zeros( shape=(5,6) , dtype=float )
            
            rate_matrix[0, 0] = 1

            # Sliding Window
            rate_matrix[0, 1] = 1
            rate_matrix[1, 1] = (self.depth_out*self.rows_out*self.cols_out)/(self.depth_in*self.rows_in*self.cols_in)

            # Fork
            rate_matrix[1, 2] = 1
            rate_matrix[2, 2] = 1

            # Convolution 3D
            rate_matrix[2, 3] = f_fine #f_fine #*self.groups / math.ceil(1/f_coarseOut)
            rate_matrix[3, 3] = f_fine

            # Accumulation

            # Glue
            rate_matrix[3, 4] = 1
            rate_matrix[4, 4] = 1

            # Concatenation
            rate_matrix[4, 5] = 1
        elif self.pointwise:
            rate_matrix = np.zeros( shape=(5,6) , dtype=float )
            
            rate_matrix[0, 0] = 1

            # Sliding Window

            # Fork
            rate_matrix[0, 1] = 1
            rate_matrix[1, 1] = 1

            # Convolution 3D
            rate_matrix[1, 2] = f_fine
            rate_matrix[2, 2] = f_fine

            # Accumulation
            rate_matrix[2, 3] = 1
            rate_matrix[3, 3] = 1

            # Glue
            rate_matrix[3, 4] = 1
            rate_matrix[4, 4] = 1

            # Concatenation
            rate_matrix[4, 5] = 1
        else:
            rate_matrix = np.zeros( shape=(6,7) , dtype=float )

            rate_matrix[0, 0] = 1
        
            # Sliding Window
            rate_matrix[0, 1] = 1
            rate_matrix[1, 1] = (self.depth_out*self.rows_out*self.cols_out)/(self.depth_in*self.rows_in*self.cols_in)

            # Fork
            rate_matrix[1, 2] = 1
            rate_matrix[2, 2] = 1

            # Convolution 3D
            rate_matrix[2, 3] = f_fine
            rate_matrix[3, 3] = f_fine

            # Accumulation
            rate_matrix[3, 4] = 1
            rate_matrix[4, 4] = 1

            # Glue
            rate_matrix[4, 5] = 1
            rate_matrix[5, 5] = 1

            # Concatenation
            rate_matrix[5, 6] = 1

        assert np.max(rate_matrix) <= 1 and np.min(rate_matrix[np.nonzero(rate_matrix)]) > 0, "Rate matrix issue"
        if DEBUG:
            print("R:\n{}".format(rate_matrix))
        return rate_matrix

    def get_stream_matrix(self, f_coarseIn, f_coarseOut):
        if self.depthwise:
            stream_matrix = np.zeros( shape=(5,6) , dtype=float )

            stream_matrix[0, 0] = 1
        
            # Sliding Window
            stream_matrix[0, 1] = math.ceil(self.channels * f_coarseIn)
            stream_matrix[1, 1] = math.ceil(self.channels * f_coarseIn)

            # Fork
            stream_matrix[1, 2] = math.ceil(self.channels * f_coarseIn)
            stream_matrix[2, 2] = math.ceil(self.channels * f_coarseIn)

            # Convolution 3D
            stream_matrix[2, 3] = math.ceil(self.channels * f_coarseIn)
            stream_matrix[3, 3] = math.ceil(self.channels * f_coarseIn)

            # Accumulation

            # Glue
            stream_matrix[3, 4] = math.ceil(self.channels * f_coarseIn)
            stream_matrix[4, 4] = math.ceil(self.channels * f_coarseIn)

            # Concatenation
            stream_matrix[4, 5] = 1
        elif self.pointwise:
            stream_matrix = np.zeros( shape=(5,6) , dtype=float )
            
            stream_matrix[0, 0] = 1

            # Sliding Window

            # Fork
            stream_matrix[0, 1] = math.ceil(self.channels * f_coarseIn)
            stream_matrix[1, 1] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)

            # Convolution 3D
            stream_matrix[1, 2] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
            stream_matrix[2, 2] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)

            # Accumulation
            stream_matrix[2, 3] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
            stream_matrix[3, 3] = math.ceil(self.filters * f_coarseOut)

            # Glue
            stream_matrix[3, 4] = math.ceil(self.filters * f_coarseOut)
            stream_matrix[4, 4] = math.ceil(self.filters * f_coarseOut)

            # Concatenation
            stream_matrix[4, 5] = 1
        else:
            stream_matrix = np.zeros( shape=(6,7) , dtype=float )

            stream_matrix[0, 0] = 1
            
            # Sliding Window
            stream_matrix[0, 1] = math.ceil(self.channels * f_coarseIn)
            stream_matrix[1, 1] = math.ceil(self.channels * f_coarseIn)

            # Fork
            stream_matrix[1, 2] = math.ceil(self.channels * f_coarseIn)
            stream_matrix[2, 2] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)

            # Convolution 3D
            stream_matrix[2, 3] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
            stream_matrix[3, 3] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)

            # Accumulation
            stream_matrix[3, 4] = math.ceil(self.channels * f_coarseIn) * math.ceil(self.filters * f_coarseOut)
            stream_matrix[4, 4] = math.ceil(self.filters * f_coarseOut)

            # Glue
            stream_matrix[4, 5] = math.ceil(self.filters * f_coarseOut)
            stream_matrix[5, 5] = math.ceil(self.filters * f_coarseOut)

            # Concatenation
            stream_matrix[5, 6] = 1

        if DEBUG:
            print("S:\n{}".format(stream_matrix))
        return stream_matrix

    def get_data_matrix(self, mem_bw_in, mem_bw_out, f_fine):
        if self.depthwise:
            data_matrix = np.zeros( shape=(5,6) , dtype=float )

            data_matrix[0, 0] = mem_bw_in

            # Sliding Window
            data_matrix[0, 1] = -math.ceil(1/f_fine)
            data_matrix[1, 1] = self.kd * self.kw * self.kh

            # Fork
            data_matrix[1, 2] = -self.kd * self.kw * self.kh
            data_matrix[2, 2] = self.kd * self.kw * self.kh

            # Convolution 3D
            data_matrix[2, 3] = -self.kd * self.kw * self.kh
            data_matrix[3, 3] = 1

            # Accumulation

            # Glue
            data_matrix[3, 4] = -1
            data_matrix[4, 4] = 1

            # Concatenation
            data_matrix[4, 5] = -mem_bw_out
        elif self.pointwise:
            data_matrix = np.zeros( shape=(5,6) , dtype=float )
            
            data_matrix[0, 0] = mem_bw_in

            # Sliding Window

            # Fork
            data_matrix[0, 1] = -self.kd * self.kw * self.kh
            data_matrix[1, 1] = self.kd * self.kw * self.kh

            # Convolution 3D
            data_matrix[1, 2] = -self.kd * self.kw * self.kh
            data_matrix[2, 2] = 1

            # Accumulation
            data_matrix[2, 3] = -1
            data_matrix[3, 3] = 1

            # Glue
            data_matrix[3, 4] = -1
            data_matrix[4, 4] = 1

            # Concatenation
            data_matrix[4, 5] = -mem_bw_out
        else:
            data_matrix = np.zeros( shape=(6,7) , dtype=float )
            
            data_matrix[0, 0] = mem_bw_in
        
            # Sliding Window
            data_matrix[0, 1] = -1
            data_matrix[1, 1] = self.kd * self.kw * self.kh

            # Fork
            data_matrix[1, 2] = -self.kd * self.kw * self.kh
            data_matrix[2, 2] = self.kd * self.kw * self.kh

            # Convolution 3D
            data_matrix[2, 3] = -self.kd * self.kw * self.kh
            data_matrix[3, 3] = 1

            # Accumulation
            data_matrix[3, 4] = -1
            data_matrix[4, 4] = 1

            # Glue
            data_matrix[4, 5] = -1
            data_matrix[5, 5] = 1

            # Concatenation
            data_matrix[5, 6] = -mem_bw_out

        if DEBUG:
            print("D:\n{}".format(data_matrix))
        return data_matrix

    def get_workload_matrix(self):
        in_volume = self.depth_in * self.rows_in * self.cols_in
        in_volume_pad = (self.depth_in + 2*self.padding[0]) * (self.rows_in + 2*self.padding[1]) * (self.cols_in + 2*self.padding[2])
        out_volume = self.depth_out * self.rows_out * self.cols_out
        kernel_volume = self.kd * self.kw * self.kh

        if self.depthwise:
            workload_matrix = np.zeros( shape=(5,6) , dtype=float )

            workload_matrix[0, 0] = in_volume * self.channels
            
            # Sliding Window
            workload_matrix[0, 1] = in_volume_pad * self.channels
            workload_matrix[1, 1] = out_volume * kernel_volume * self.channels

            # Fork
            workload_matrix[1, 2] = out_volume * kernel_volume * self.channels
            workload_matrix[2, 2] = out_volume * kernel_volume * self.channels

            # Convolution 3D
            workload_matrix[2, 3] = out_volume * kernel_volume * self.channels
            workload_matrix[3, 3] = out_volume * self.channels

            # Accumulation

            # Glue
            workload_matrix[3, 4] = out_volume * self.filters
            workload_matrix[4, 4] = out_volume * self.filters

            # Concatenation
            workload_matrix[4, 5] = out_volume * self.filters
        elif self.pointwise:
            workload_matrix = np.zeros( shape=(5,6) , dtype=float )
            
            workload_matrix[0, 0] = in_volume * self.channels

            # Sliding Window

            # Fork
            workload_matrix[0, 1] = out_volume * kernel_volume * self.channels
            workload_matrix[1, 1] = out_volume * kernel_volume * self.channels * self.filters

            # Convolution 3D
            workload_matrix[1, 2] = out_volume * kernel_volume * self.channels * self.filters
            workload_matrix[2, 2] = out_volume * self.channels * self.filters

            # Accumulation
            workload_matrix[2, 3] = out_volume * self.channels * self.filters
            workload_matrix[3, 3] = out_volume * self.filters

            # Glue
            workload_matrix[3, 4] = out_volume * self.filters
            workload_matrix[4, 4] = out_volume * self.filters

            # Concatenation
            workload_matrix[4, 5] = out_volume * self.filters
        else:
            workload_matrix = np.zeros( shape=(6,7) , dtype=float )
            
            workload_matrix[0, 0] = in_volume * self.channels
            
            # Sliding Window
            workload_matrix[0, 1] = in_volume_pad * self.channels
            workload_matrix[1, 1] = out_volume * kernel_volume * self.channels

            # Fork
            workload_matrix[1, 2] = out_volume * kernel_volume * self.channels
            workload_matrix[2, 2] = out_volume * kernel_volume * self.channels * self.filters

            # Convolution 3D
            workload_matrix[2, 3] = out_volume * kernel_volume * self.channels * self.filters
            workload_matrix[3, 3] = out_volume * self.channels * self.filters

            # Accumulation
            workload_matrix[3, 4] = out_volume * self.channels * self.filters
            workload_matrix[4, 4] = out_volume * self.filters

            # Glue
            workload_matrix[4, 5] = out_volume * self.filters
            workload_matrix[5, 5] = out_volume * self.filters

            # Concatenation
            workload_matrix[5, 6] = out_volume * self.filters

        if DEBUG:
            print("WL:\n{}".format(workload_matrix))
        return workload_matrix