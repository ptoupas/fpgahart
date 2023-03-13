import math

import numpy as np

from fpga_hart.layers.base_layer_3d import BaseLayer3D

np.set_printoptions(precision=5, suppress=True, linewidth=150)
np.seterr(divide="ignore", invalid="ignore")

DEBUG = False


class BatchNorm3DLayer(BaseLayer3D):
    def __init__(self, max_DSP_util, max_BRAM_util, description):
        super().__init__(max_DSP_util=max_DSP_util, max_BRAM_util=max_BRAM_util)

        self.input_shape = description["shape_in"][0]
        self.depth_in = self.input_shape[2]
        self.rows_in = self.input_shape[3]
        self.cols_in = self.input_shape[4]
        self.output_shape = description["shape_out"]
        self.depth_out = self.output_shape[2]
        self.rows_out = self.output_shape[3]
        self.cols_out = self.output_shape[4]

        self.channels = self.input_shape[1]
        self.filters = self.output_shape[1]

        self.data_size_in = np.prod(np.array(self.input_shape[1:]))
        self.data_size_out = np.prod(np.array(self.output_shape[1:]))

    def update_shapes(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.channels = self.input_shape[1]
        self.depth_in = self.input_shape[2]
        self.rows_in = self.input_shape[3]
        self.cols_in = self.input_shape[4]
        self.data_size_in = np.prod(np.array(self.input_shape[1:]))

        self.output_shape = output_shape
        self.filters = self.output_shape[1]
        self.depth_out = self.output_shape[2]
        self.rows_out = self.output_shape[3]
        self.cols_out = self.output_shape[4]
        self.data_size_out = np.prod(np.array(self.output_shape[1:]))

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
        self.total_bw_util = 0
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
        return int(np.prod(np.array(self.output_shape[1:])))

    def get_dp_info(self):
        dp_info = {}

        dp_info["latency(C)"] = self.latency_cycles
        dp_info["latency(S)"] = self.latency_sec
        dp_info["GOP/s"] = self.throughput_ops * 1e-9
        dp_info["GOPs"] = self.get_total_workload() * 1e-9
        dp_info["vols/s"] = self.throughput_vols
        dp_info["DSP"] = self.dsps_util
        dp_info["DSP_RAW"] = self.dsp_raw
        dp_info["BRAM"] = self.bram_util
        dp_info["BRAM_RAW"] = self.bram_raw
        dp_info["rateIn"] = self.full_rate_in
        dp_info["rateOut"] = self.full_rate_out
        dp_info["depth"] = self.depth
        dp_info["branch_depth"] = 0
        dp_info["muls"] = self.max_parallel_muls
        dp_info["adds"] = self.max_parallel_adds
        dp_info["memWords"] = self.memory
        dp_info["memKBs"] = self.memoryKB
        dp_info["dataSizeIn"] = (self.data_size_in * self.word_bytes) / 1e6
        dp_info["dataSizeOut"] = (self.data_size_out * self.word_bytes) / 1e6
        dp_info["memBoundedIn"] = self.mem_bd_in
        dp_info["memBoundedOut"] = self.mem_bd_out
        dp_info["memBwUtil"] = self.total_bw_util
        dp_info["config"] = self.config

        return dp_info

    def get_num_streams(self):
        self.max_streams_in = self.channels
        self.max_streams_out = self.filters
        return self.max_streams_in, self.max_streams_out

    def get_design_point(self, coarse_inout, mem_bw_in, mem_bw_out, ignore_bw_util=False):
        self.update_layer()

        gamma_matrix = (
            self.get_rate_matrix()
            * self.get_stream_matrix(coarse_inout)
            * self.get_data_matrix(mem_bw_in, mem_bw_out)
        )
        if DEBUG:
            print("Γ:\n{}".format(gamma_matrix))
        gamma_matrix_balanced, mem_bounded_in, mem_bounded_out = self.balance_matrix(
            gamma_matrix.copy()
        )
        if DEBUG:
            print("Γ Balanced:\n{}".format(gamma_matrix_balanced))

        layer_mem_bw_in = (
            abs(gamma_matrix_balanced[0, 0]) * self.cycles_per_sec * self.word_length
        )
        layer_mem_bw_out = (
            abs(gamma_matrix_balanced[-1, -1]) * self.cycles_per_sec * self.word_length
        )
        total_bw_util = (
            (layer_mem_bw_in + layer_mem_bw_out) / self.mem_bandwidth
        ) * 100
        assert total_bw_util <= 100 + 1e-6 or ignore_bw_util, f"Total BW utilization ({total_bw_util:.2f}) is greater than 100%"

        workload_matrix = self.get_workload_matrix()
        ii_matrix = np.nan_to_num(workload_matrix / gamma_matrix_balanced)
        if DEBUG:
            print("II:\n{}".format(ii_matrix))

        max_parallel_muls = math.ceil(self.channels * coarse_inout)
        max_parallel_adds = math.ceil(self.channels * coarse_inout)
        layer_fifos_arrays = {}
        depth = 2

        (
            latency_sec,
            latency_cycles,
            thr_in,
            thr_out,
            dsps_util,
            dsp_raw,
            bram_util,
            bram_raw,
            memKBs,
        ) = self.get_dp_performance(
            workload_matrix,
            ii_matrix,
            max_parallel_muls,
            max_parallel_adds,
            layer_fifos_arrays,
            depth,
            coarse_inout=coarse_inout,
        )
        total_ops = self.get_total_workload()
        throughput_ops = total_ops / latency_sec
        thr_in /= workload_matrix[0, 0]  # Volumes per second
        thr_out /= workload_matrix[-1, -1]  # Volumes per second
        assert math.isclose(
            thr_in, thr_out
        ), "Thoughputs missmatch. IN = {}, OUT = {}.".format(thr_in, thr_out)

        if dsps_util < self.max_DSP_util and bram_util < self.max_BRAM_util:
            self.full_rate_in = [gamma_matrix_balanced[0, 0]]
            self.full_rate_out = [abs(gamma_matrix_balanced[-1, -1])]
            self.max_parallel_muls = max_parallel_muls
            self.max_parallel_adds = max_parallel_adds
            self.depth = depth
            self.mem_bd_in = [mem_bounded_in]
            self.mem_bd_out = [mem_bounded_out]
            self.total_bw_util = total_bw_util

            config = [coarse_inout]
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
                print(
                    "*" * 40,
                    "inout factor={} latency={} depth={}, max_parallel_muls={}".format(
                        coarse_inout, int(latency_cycles), depth, dsps_util
                    ),
                )
        else:
            self.update_layer()
            if DEBUG:
                print("Discarding design point.")

        return self.get_dp_info()

    def get_rate_matrix(self):
        rate_matrix = np.zeros(shape=(2, 3), dtype=float)

        rate_matrix[0, 0] = 1

        rate_matrix[0, 1] = 1
        rate_matrix[1, 1] = 1

        rate_matrix[1, 2] = 1

        assert (
            np.max(rate_matrix) <= 1
            and np.min(rate_matrix[np.nonzero(rate_matrix)]) > 0
        ), "Rate matrix issue"

        if DEBUG:
            print("R:\n{}".format(rate_matrix))
        return rate_matrix

    def get_stream_matrix(self, coarse_inout):
        stream_matrix = np.zeros(shape=(2, 3), dtype=float)

        stream_matrix[0, 0] = 1

        stream_matrix[0, 1] = math.ceil(self.channels * coarse_inout)
        stream_matrix[1, 1] = math.ceil(self.channels * coarse_inout)
        stream_matrix[1, 2] = 1

        if DEBUG:
            print("S:\n{}".format(stream_matrix))
        return stream_matrix

    def get_data_matrix(self, mem_bw_in, mem_bw_out):
        data_matrix = np.zeros(shape=(2, 3), dtype=float)

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

        workload_matrix = np.zeros(shape=(2, 3), dtype=float)

        workload_matrix[0, 0] = in_volume

        workload_matrix[0, 1] = in_volume
        workload_matrix[1, 1] = out_volume

        workload_matrix[1, 2] = out_volume

        if DEBUG:
            print("WL:\n{}".format(workload_matrix))
        return workload_matrix
