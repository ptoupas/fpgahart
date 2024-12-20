import logging
import math
from typing import Tuple

import numpy as np

from fpga_hart import _logger
from fpga_hart.layers.base_layer_3d import BaseLayer3D

np.set_printoptions(precision=5, suppress=True, linewidth=150)
np.seterr(divide="ignore", invalid="ignore")


class Pooling3DLayer(BaseLayer3D):
    def __init__(self, max_DSP_util, max_BRAM_util, description, platform):
        super().__init__(max_DSP_util=max_DSP_util, max_BRAM_util=max_BRAM_util, platform=platform)
        # _logger.setLevel(level=logging.DEBUG)

        self.op_type = "max" if "Max" in description["operation"] else "avg"

        self.input_shape = description["shape_in"][0]
        self.depth_in = self.input_shape[2]
        self.rows_in = self.input_shape[3]
        self.cols_in = self.input_shape[4]
        self.output_shape = description["shape_out"]
        self.depth_out = self.output_shape[2]
        self.rows_out = self.output_shape[3]
        self.cols_out = self.output_shape[4]

        self.kernel_shape = description["kernel"]
        self.kd = self.kernel_shape[0]
        self.kh = self.kernel_shape[1]
        self.kw = self.kernel_shape[2]

        self.channels = self.input_shape[1]
        self.filters = self.input_shape[1]

        self.padding = description["padding"]
        self.stride = description["stride"]

        self.branching = description["branching"]

        self.data_size_in = np.prod(np.array(self.input_shape[1:]))
        self.data_size_out = np.prod(np.array(self.output_shape[1:]))

    def update_shapes(self, input_shape, output_shape, padding=None, stride=None):
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

        if padding is not None:
            self.padding = padding
        if stride is not None:
            self.stride = stride

    def update_layer(self):
        self.channels = self.input_shape[1]
        self.full_rate_in = []
        self.full_rate_out = []
        self.max_parallel_muls = 0
        self.max_parallel_adds = 0
        self.memory = 0
        self.memoryKB = 0
        self.depth = 0
        self.mem_bw_in = []
        self.mem_bw_out = []
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
        return (
            self.depth_out
            * self.rows_out
            * self.cols_out
            * self.kd
            * self.kh
            * self.kw
            * self.channels
        )

    def get_dp_info(self) -> dict:
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
        dp_info["memBoundedIn"] = self.mem_bw_in
        dp_info["memBoundedOut"] = self.mem_bw_out
        dp_info["memBwUtil"] = self.total_bw_util
        dp_info["config"] = self.config

        return dp_info

    def get_num_streams(self):
        self.max_streams_in = self.channels
        self.max_streams_out = self.channels
        return self.max_streams_in, self.max_streams_out

    def get_resource_util(
        self,
        f_fine: np.float64,
        f_coarse_inout: np.float64,
    ) -> Tuple[float, float]:

        pipeline_depth = 2
        pipeline_depth += (
            math.ceil(1 / f_coarse_inout)
            * (self.cols_in + 2 * self.padding[2])
            * (self.depth_in + 2 * self.padding[0])
            * (self.kh - 1)
            + math.ceil(1 / f_coarse_inout)
            * (self.depth_in + 2 * self.padding[0])
            * (self.kw - 1)
            + math.ceil(1 / f_coarse_inout) * (self.kd - 1)
        )
        pipeline_depth += math.ceil(1 / f_coarse_inout) * (
            (self.kh - 1) * self.kw * self.kd + (self.kw - 1) * self.kd + (self.kd - 1)
        )
        pipeline_depth += math.ceil(1 / f_fine) + 1

        kernel_elems = int(np.prod(np.array(self.kernel_shape)))

        layer_fifos_arrays = {
            "pool_sw_lb_3d": 0,
            "pool_sw_lb_2d": 0,
            "pool_sw_wb_3d": 0,
        }

        depth_line_buffer_3d = (
            math.ceil(1 / f_coarse_inout) * (self.depth_in + 2 * self.padding[0]) + 1
        )
        layer_fifos_arrays["pool_sw_lb_3d"] = depth_line_buffer_3d

        depth_line_buffer_2d = (
            math.ceil(1 / f_coarse_inout)
            * (
                (self.depth_in + 2 * self.padding[0])
                * (self.cols_in + 2 * self.padding[2])
                - (self.kw - 1) * self.depth_in
                - (self.kd - 1)
            )
            + 1
        )
        layer_fifos_arrays["pool_sw_lb_2d"] = depth_line_buffer_2d

        depth_window_buffer_3d = math.ceil(1 / f_coarse_inout) + 1
        layer_fifos_arrays["pool_sw_wb_3d"] = depth_window_buffer_3d

        max_parallel_muls = 0 if self.op_type == "max" else 1
        max_parallel_adds = (
            math.ceil((kernel_elems - 1) * f_fine)
            * math.ceil(self.channels * f_coarse_inout)
            * math.ceil(self.channels * f_coarse_inout)
        )

        (
            _,
            _,
            _,
            _,
            dsps_util,
            dsp_raw,
            bram_util,
            bram_raw,
            _,
        ) = self.get_dp_performance(
            np.zeros(shape=(3, 4), dtype=float),
            np.zeros(shape=(3, 4), dtype=float),
            max_parallel_muls,
            max_parallel_adds,
            layer_fifos_arrays,
            0,
            kernel_shape=[self.channels, self.channels, self.kd, self.kh, self.kw],
            coarse_inout=math.ceil(self.channels * f_coarse_inout),
            fine=math.ceil(kernel_elems * f_fine),
        )

        return dsps_util, bram_util, pipeline_depth

    def get_design_point(
        self,
        f_fine: np.float64,
        f_coarse_inout: np.float64,
        mem_bw_in: int,
        mem_bw_out: int,
        ignore_bw_util: bool = False
    ) -> dict:
        self.update_layer()

        kernel_elems = int(np.prod(np.array(self.kernel_shape)))

        layer_fifos_arrays = {
            "pool_sw_lb_3d": 0,
            "pool_sw_lb_2d": 0,
            "pool_sw_wb_3d": 0,
        }

        depth = 2

        depth_line_buffer_3d = (
            math.ceil(1 / f_coarse_inout) * (self.depth_in + 2 * self.padding[0]) + 1
        )
        layer_fifos_arrays["pool_sw_lb_3d"] = depth_line_buffer_3d

        depth_line_buffer_2d = (
            math.ceil(1 / f_coarse_inout)
            * (
                (self.depth_in + 2 * self.padding[0])
                * (self.cols_in + 2 * self.padding[2])
                - (self.kw - 1) * self.depth_in
                - (self.kd - 1)
            )
            + 1
        )
        layer_fifos_arrays["pool_sw_lb_2d"] = depth_line_buffer_2d

        depth_window_buffer_3d = math.ceil(1 / f_coarse_inout) + 1
        layer_fifos_arrays["pool_sw_wb_3d"] = depth_window_buffer_3d

        # DEPTH V1
        depth += (
            math.ceil(1 / f_coarse_inout)
            * (self.cols_in + 2 * self.padding[2])
            * (self.depth_in + 2 * self.padding[0])
            * (self.kh - 1)
            + math.ceil(1 / f_coarse_inout)
            * (self.depth_in + 2 * self.padding[0])
            * (self.kw - 1)
            + math.ceil(1 / f_coarse_inout) * (self.kd - 1)
        )
        depth += math.ceil(1 / f_coarse_inout) * (
            (self.kh - 1) * self.kw * self.kd + (self.kw - 1) * self.kd + (self.kd - 1)
        )

        # Convolution Module (CM) Depth and Memory
        depth += math.ceil(1 / f_fine) + 1

        max_parallel_muls = 0 if self.op_type == "max" else 1
        max_parallel_adds = (
            math.ceil((kernel_elems - 1) * f_fine)
            * math.ceil(self.channels * f_coarse_inout)
            * math.ceil(self.channels * f_coarse_inout)
        )

        rate_matrix_balanced, rate_matrix = self.get_rate_matrix(f_fine, f_coarse_inout)
        stream_matrix = self.get_stream_matrix(f_coarse_inout)
        data_matrix = self.get_data_matrix(mem_bw_in, mem_bw_out, f_fine)

        gamma_matrix = rate_matrix_balanced * stream_matrix * data_matrix
        _logger.debug("Γ:\n{}".format(gamma_matrix))
        (
            gamma_matrix,
            mem_bounded_in,
            mem_bounded_out,
        ) = self.balance_matrix(gamma_matrix.copy())
        _logger.debug("Γ Balanced:\n{}".format(gamma_matrix))

        layer_mem_bw_in = (
            abs(gamma_matrix[0, 0]) * self.cycles_per_sec * self.word_length
        )
        layer_mem_bw_out = (
            abs(gamma_matrix[-1, -1]) * self.cycles_per_sec * self.word_length
        )
        total_bw_util = (
            (layer_mem_bw_in + layer_mem_bw_out) / self.mem_bandwidth
        ) * 100
        assert total_bw_util <= 100 + 1e-6 or ignore_bw_util, f"Total BW utilization ({total_bw_util:.2f}) is greater than 100%"

        workload_matrix = self.get_workload_matrix()
        ii_matrix = np.nan_to_num(workload_matrix / gamma_matrix)
        _logger.debug("II:\n{}".format(ii_matrix))

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
            kernel_shape=[self.channels, self.channels, self.kd, self.kh, self.kw],
            coarse_inout=math.ceil(self.channels * f_coarse_inout),
            fine=math.ceil(kernel_elems * f_fine),
        )

        total_ops = self.get_total_workload()
        throughput_ops = total_ops / latency_sec
        thr_in /= (
            np.prod(np.array(self.input_shape[2:])) * self.channels
        )  # Volumes per second
        thr_out /= (
            np.prod(np.array(self.output_shape[2:])) * self.channels
        )  # Volumes per second
        assert math.isclose(
            thr_in, thr_out
        ), "Thoughputs missmatch. IN = {}, OUT = {}.".format(thr_in, thr_out)

        _logger.debug(
            f"Fine: {f_fine:.3f} ({f_fine*np.prod(np.array(self.kernel_shape))}), CoarseInOut: {f_coarse_inout:.3f} ({int(f_coarse_inout*self.channels)}), Shape in: {self.input_shape}, Shape out: {self.output_shape}, Kernel: {self.kernel_shape}"
        )
        if dsps_util < self.max_DSP_util and bram_util < self.max_BRAM_util:

            self.full_rate_in = [gamma_matrix[0, 0]]
            self.full_rate_out = [abs(gamma_matrix[-1, -1])]
            self.max_parallel_muls = max_parallel_muls
            self.max_parallel_adds = max_parallel_adds
            self.depth = depth
            self.mem_bw_in = [mem_bounded_in]
            self.mem_bw_out = [mem_bounded_out]
            self.total_bw_util = total_bw_util

            config = [
                f_fine,
                f_coarse_inout
            ]
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

            _logger.debug(
                "(fine={:.2f}({}), cInOut={:.2f}({}), bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f} ({}), BRAM % = {:.2f} ({}), latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(
                    f_fine,
                    math.ceil(f_fine * kernel_elems),
                    f_coarse_inout,
                    math.ceil(self.channels * f_coarse_inout),
                    mem_bw_in,
                    mem_bw_out,
                    dsps_util,
                    dsp_raw,
                    bram_util,
                    bram_raw,
                    latency_sec,
                    int(latency_cycles),
                    throughput_ops * 1e-9,
                    thr_in,
                    thr_out,
                    depth,
                    total_ops * 1e-9,
                    mem_bounded_in,
                    mem_bounded_out,
                )
            )
        else:
            self.update_layer()
            _logger.debug(
                "Discarding design point. DSP = {:.2f} - BRAM = {:.2f}".format(
                    dsps_util, bram_util
                )
            )

        return self.get_dp_info()

    def get_rate_matrix(self, f_fine, f_coarse_inout):
        in_volume_pad = (
            (self.depth_in + 2 * self.padding[0])
            * (self.rows_in + 2 * self.padding[1])
            * (self.cols_in + 2 * self.padding[2])
        )

        rate_matrix = np.zeros(shape=(3, 4), dtype=float)

        rate_matrix[0, 0] = 1

        # Sliding Window
        rate_matrix[0, 1] = 1
        rate_matrix[1, 1] = (
            self.depth_out * self.rows_out * self.cols_out
        ) / in_volume_pad

        # Pool
        rate_matrix[1, 2] = 1
        rate_matrix[2, 2] = 1

        # Concatenation
        rate_matrix[2, 3] = 1

        assert (
            np.max(rate_matrix) <= 1
            and np.min(rate_matrix[np.nonzero(rate_matrix)]) > 0
        ), f"Rate matrix issue {rate_matrix}"
        rate_matrix_balanced, _, _ = self.balance_matrix(rate_matrix.copy())
        rate_matrix_balanced[0, 0] = 1
        rate_matrix_balanced[-1, -1] = 1
        _logger.debug("R:\n{}".format(rate_matrix))
        _logger.debug("R (balanced):\n{}".format(rate_matrix_balanced))
        return rate_matrix_balanced, rate_matrix

    def get_stream_matrix(self, f_coarse_inout):

        stream_matrix = np.zeros(shape=(3, 4), dtype=float)

        stream_matrix[0, 0] = 1

        # Sliding Window
        stream_matrix[0, 1] = math.ceil(self.channels * f_coarse_inout)
        stream_matrix[1, 1] = (
            math.ceil(self.channels * f_coarse_inout) * self.kd * self.kw * self.kh
        )

        # Pool
        stream_matrix[1, 2] = (
            math.ceil(self.channels * f_coarse_inout) * self.kd * self.kw * self.kh
        )
        stream_matrix[2, 2] = math.ceil(self.channels * f_coarse_inout)

        # Concatenation
        stream_matrix[2, 3] = 1

        _logger.debug("S:\n{}".format(stream_matrix))
        return stream_matrix

    def get_data_matrix(self, mem_bw_in, mem_bw_out, f_fine):

        data_matrix = np.zeros(shape=(3, 4), dtype=float)

        data_matrix[0, 0] = mem_bw_in

        # Sliding Window
        data_matrix[0, 1] = -1
        data_matrix[1, 1] = 1

        # Pool
        data_matrix[1, 2] = -1
        data_matrix[2, 2] = 1

        # Concatenation
        data_matrix[2, 3] = -mem_bw_out

        _logger.debug("D:\n{}".format(data_matrix))
        return data_matrix

    def get_workload_matrix(self):
        in_volume = self.depth_in * self.rows_in * self.cols_in
        in_volume_pad = (
            (self.depth_in + 2 * self.padding[0])
            * (self.rows_in + 2 * self.padding[1])
            * (self.cols_in + 2 * self.padding[2])
        )
        out_volume = self.depth_out * self.rows_out * self.cols_out
        kernel_volume = self.kd * self.kw * self.kh

        workload_matrix = np.zeros(shape=(3, 4), dtype=float)

        workload_matrix[0, 0] = in_volume * self.channels

        # Sliding Window
        workload_matrix[0, 1] = in_volume_pad * self.channels
        workload_matrix[1, 1] = out_volume * kernel_volume * self.channels

        # Pool
        workload_matrix[1, 2] = out_volume * kernel_volume * self.channels
        workload_matrix[2, 2] = out_volume * self.channels

        # Concatenation
        workload_matrix[2, 3] = out_volume * self.channels

        _logger.debug("WL:\n{}".format(workload_matrix))
        return workload_matrix
