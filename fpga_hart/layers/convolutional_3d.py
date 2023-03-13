import logging
import math
from typing import Tuple

import numpy as np

from fpga_hart import _logger
from fpga_hart.layers.base_layer_3d import BaseLayer3D

np.set_printoptions(precision=5, suppress=True, linewidth=150)
np.seterr(divide="ignore", invalid="ignore")


class Convolutional3DLayer(BaseLayer3D):
    def __init__(self, max_DSP_util, max_BRAM_util, description):
        super().__init__(max_DSP_util=max_DSP_util, max_BRAM_util=max_BRAM_util)
        # _logger.setLevel(level=logging.DEBUG)

        self.input_shape = description["shape_in"][0]
        self.depth_in = self.input_shape[2]
        self.rows_in = self.input_shape[3]
        self.cols_in = self.input_shape[4]
        self.output_shape = description["shape_out"]
        self.depth_out = self.output_shape[2]
        self.rows_out = self.output_shape[3]
        self.cols_out = self.output_shape[4]

        self.kernel_shape = description["kernel"][2:]
        self.kd = self.kernel_shape[0]
        self.kh = self.kernel_shape[1]
        self.kw = self.kernel_shape[2]
        self.bias_shape = description["bias"]

        self.channels = self.input_shape[1]
        self.filters = self.output_shape[1]

        self.groups = description["groups"]

        self.padding = description["padding"]
        self.stride = description["stride"]
        self.dilation = description["dilation"]

        self.branching = description["branching"]

        self.data_size_in = np.prod(np.array(self.input_shape[1:]))
        self.data_size_out = np.prod(np.array(self.output_shape[1:]))

        self.depthwise = False
        self.pointwise = False
        if self.groups == self.channels and self.channels == self.filters:
            self.depthwise = True
        elif np.prod(np.array(self.kernel_shape)) == 1:
            self.pointwise = True

        self.spatial = False
        self.temporal = False
        if self.kd > 1 and self.kh == 1 and self.kw == 1:
            self.temporal = True
        if self.kd == 1 and self.kh > 1 and self.kw > 1:
            self.spatial = True

    def update_shapes(
        self, input_shape, output_shape, groups=None, padding=None, stride=None
    ):
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

        if self.bias_shape:
            self.bias_shape = [self.filters]
        if self.depthwise:
            self.input_shape[1] = self.filters
            self.channels = self.filters
            self.groups = self.filters

        if groups is not None:
            self.groups = groups
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
        self.wr_factor = 1
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
            return (
                self.depth_out
                * self.rows_out
                * self.cols_out
                * self.kd
                * self.kw
                * self.kh
                * self.channels
                * self.filters
            )
        else:
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
        dp_info["wr_factor"] = self.wr_factor

        return dp_info

    def get_num_streams(self):
        self.max_streams_in = self.channels
        self.max_streams_out = self.filters
        return self.max_streams_in, self.max_streams_out

    def get_resource_util(
        self,
        f_fine: np.float64,
        f_coarseIn: np.float64,
        f_coarseOut: np.float64,
    ) -> Tuple[float, float]:

        kernel_elems = int(np.prod(np.array(self.kernel_shape)))

        layer_fifos_arrays = {
            "sw_lb_3d": 0,
            "sw_lb_2d": 0,
            "sw_wb_3d": 0,
            "acc_fifo": 0,
            "acc_array": 0,
        }
        if not self.pointwise:
            if not self.temporal and not self.spatial:
                depth_line_buffer_3d = (
                    math.ceil(1 / f_coarseIn) * (self.depth_in + 2 * self.padding[0]) + 1
                )
                layer_fifos_arrays["sw_lb_3d"] = depth_line_buffer_3d

                depth_line_buffer_2d = (
                    math.ceil(1 / f_coarseIn)
                    * (
                        (self.depth_in + 2 * self.padding[0])
                        * (self.cols_in + 2 * self.padding[2])
                        - (self.kw - 1) * self.depth_in
                        - (self.kd - 1)
                    )
                    + 1
                )
                layer_fifos_arrays["sw_lb_2d"] = depth_line_buffer_2d

                depth_window_buffer_3d = math.ceil(1 / f_coarseIn) + 1
                layer_fifos_arrays["sw_wb_3d"] = depth_window_buffer_3d
            elif self.spatial and not self.temporal:
                depth_line_buffer_3d = (
                    math.ceil(1 / f_coarseIn) * (self.depth_in + 2 * self.padding[0]) + 1
                )
                layer_fifos_arrays["sw_lb_3d"] = depth_line_buffer_3d

                depth_line_buffer_2d = (
                    math.ceil(1 / f_coarseIn)
                    * (
                        (self.depth_in + 2 * self.padding[0])
                        * (self.cols_in + 2 * self.padding[2])
                        - (self.kw - 1) * self.depth_in
                    )
                    + 1
                )
                layer_fifos_arrays["sw_lb_2d"] = depth_line_buffer_2d
            elif self.temporal and not self.spatial:
                depth_line_buffer_3d = (
                    math.ceil(1 / f_coarseIn) * (self.depth_in + 2 * self.padding[0]) + 1
                )
                layer_fifos_arrays["sw_lb_3d"] = depth_line_buffer_3d

                depth_window_buffer_3d = math.ceil(1 / f_coarseIn) + 1
                layer_fifos_arrays["sw_wb_3d"] = depth_window_buffer_3d

        if not self.depthwise:
            # Accumulator Module (AM) Depth and Memory
            depth_accumulator = math.ceil(1 / f_coarseOut) + 1
            layer_fifos_arrays["acc_fifo"] = depth_accumulator

            # Accumulation Buffer
            array_accumulator = math.ceil(1 / f_coarseOut)
            layer_fifos_arrays["acc_array"] = array_accumulator

            max_parallel_muls = (
                math.ceil(kernel_elems * f_fine)
                * math.ceil(self.channels * f_coarseIn)
                * math.ceil(self.filters * f_coarseOut)
            )
            max_parallel_adds = (
                math.ceil((kernel_elems - 1) * f_fine)
                * math.ceil(self.channels * f_coarseIn)
                * math.ceil(self.filters * f_coarseOut)
            )

        else:
            max_parallel_muls = math.ceil(kernel_elems * f_fine) * math.ceil(
                self.channels * f_coarseIn
            )
            max_parallel_adds = math.ceil((kernel_elems - 1) * f_fine) * math.ceil(
                self.channels * f_coarseIn
            )

        if not self.depthwise:
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
                np.zeros(shape=(2, 3), dtype=float),
                np.zeros(shape=(2, 3), dtype=float),
                max_parallel_muls,
                max_parallel_adds,
                layer_fifos_arrays,
                0,
                kernel_shape=[self.filters, self.channels, self.kd, self.kh, self.kw],
                coarse_in=math.ceil(self.channels * f_coarseIn),
                coarse_out=math.ceil(self.filters * f_coarseOut),
                fine=math.ceil(kernel_elems * f_fine),
            )
        else:
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
                np.zeros(shape=(2, 3), dtype=float),
                np.zeros(shape=(2, 3), dtype=float),
                max_parallel_muls,
                max_parallel_adds,
                layer_fifos_arrays,
                0,
                kernel_shape=[
                    self.filters / self.groups,
                    self.channels,
                    self.kd,
                    self.kh,
                    self.kw,
                ],
                coarse_in=math.ceil(self.channels * f_coarseIn),
                coarse_out=1,
                fine=math.ceil(kernel_elems * f_fine),
            )

        return dsps_util, bram_util

    def get_design_point(
        self,
        f_fine: np.float64,
        f_coarseIn: np.float64,
        f_coarseOut: np.float64,
        mem_bw_in: int,
        mem_bw_out: int,
        wr_factor: int = 1,
        ignore_bw_util: bool = False
    ) -> dict:
        self.update_layer()

        if self.depthwise:
            assert (
                self.channels == self.filters and self.channels == self.groups
            ), "Depthwise convolutional layer must have groups equal to channels and filters"
            # self.channels = self.channels//self.groups

        kernel_elems = int(np.prod(np.array(self.kernel_shape)))

        layer_fifos_arrays = {
            "sw_lb_3d": 0,
            "sw_lb_2d": 0,
            "sw_wb_3d": 0,
            "acc_fifo": 0,
            "acc_array": 0,
        }

        depth = 2
        if self.pointwise:
            # Sliding Window Module (SWM) Depth and Memory
            depth += 1

            # Fork Module (FM) Depth and Memory

            # Convolution Module (CM) Depth and Memory
            depth += math.ceil(1 / f_fine) + 1

            # Accumulator Module (AM) Depth and Memory

            # Glue Module (GM) Depth and Memory
        else:
            # Sliding Window Module (SWM) Depth and Memory
            if not self.temporal and not self.spatial:
                depth_line_buffer_3d = (
                    math.ceil(1 / f_coarseIn) * (self.depth_in + 2 * self.padding[0]) + 1
                )
                layer_fifos_arrays["sw_lb_3d"] = depth_line_buffer_3d

                depth_line_buffer_2d = (
                    math.ceil(1 / f_coarseIn)
                    * (
                        (self.depth_in + 2 * self.padding[0])
                        * (self.cols_in + 2 * self.padding[2])
                        - (self.kw - 1) * self.depth_in
                        - (self.kd - 1)
                    )
                    + 1
                )
                layer_fifos_arrays["sw_lb_2d"] = depth_line_buffer_2d

                depth_window_buffer_3d = math.ceil(1 / f_coarseIn) + 1
                layer_fifos_arrays["sw_wb_3d"] = depth_window_buffer_3d

                # DEPTH
                depth += (
                    math.ceil(1 / f_coarseIn)
                    * (self.cols_in + 2 * self.padding[2])
                    * (self.depth_in + 2 * self.padding[0])
                    * (self.kh - 1)
                    + math.ceil(1 / f_coarseIn)
                    * (self.depth_in + 2 * self.padding[0])
                    * (self.kw - 1)
                    + math.ceil(1 / f_coarseIn) * (self.kd - 1)
                )
            elif self.spatial and not self.temporal:
                depth_line_buffer_3d = (
                    math.ceil(1 / f_coarseIn) * (self.depth_in + 2 * self.padding[0]) + 1
                )
                layer_fifos_arrays["sw_lb_3d"] = depth_line_buffer_3d

                depth_line_buffer_2d = (
                    math.ceil(1 / f_coarseIn)
                    * (
                        (self.depth_in + 2 * self.padding[0])
                        * (self.cols_in + 2 * self.padding[2])
                        - (self.kw - 1) * self.depth_in
                    )
                    + 1
                )
                layer_fifos_arrays["sw_lb_2d"] = depth_line_buffer_2d

                # DEPTH
                depth += (
                    math.ceil(1 / f_coarseIn)
                    * (self.cols_in + 2 * self.padding[2])
                    * (self.depth_in + 2 * self.padding[0])
                    * (self.kh - 1)
                    + math.ceil(1 / f_coarseIn)
                    * (self.depth_in + 2 * self.padding[0])
                    * (self.kw - 1)
                )
            elif self.temporal and not self.spatial:
                depth_line_buffer_3d = (
                    math.ceil(1 / f_coarseIn) * (self.depth_in + 2 * self.padding[0]) + 1
                )
                layer_fifos_arrays["sw_lb_3d"] = depth_line_buffer_3d

                depth_window_buffer_3d = math.ceil(1 / f_coarseIn) + 1
                layer_fifos_arrays["sw_wb_3d"] = depth_window_buffer_3d

                # DEPTH
                depth += (
                    math.ceil(1 / f_coarseIn)
                    * (self.depth_in + 2 * self.padding[0])
                    * (self.kw - 1)
                    + math.ceil(1 / f_coarseIn) * (self.kd - 1)
                )

            depth += math.ceil(1 / f_coarseIn) * (
                (self.kh - 1) * self.kw * self.kd
                + (self.kw - 1) * self.kd
                + (self.kd - 1)
            )

            # Fork Module (FM) Depth and Memory

            # Convolution Module (CM) Depth and Memory
            depth += math.ceil(1 / f_fine) + 1

            # Accumulator Module (AM) Depth and Memory

            # Glue Module (GM) Depth and Memory

        if not self.depthwise:
            # Accumulator Module (AM) Depth and Memory
            depth_accumulator = math.ceil(1 / f_coarseOut) + 1
            layer_fifos_arrays["acc_fifo"] = depth_accumulator

            # Accumulation Buffer
            array_accumulator = math.ceil(1 / f_coarseOut)
            layer_fifos_arrays["acc_array"] = array_accumulator

            # DEPTH
            depth += math.ceil(1 / f_coarseOut) + 1

            max_parallel_muls = (
                math.ceil(kernel_elems * f_fine)
                * math.ceil(self.channels * f_coarseIn)
                * math.ceil(self.filters * f_coarseOut)
            )
            max_parallel_adds = (
                math.ceil((kernel_elems - 1) * f_fine)
                * math.ceil(self.channels * f_coarseIn)
                * math.ceil(self.filters * f_coarseOut)
            )

        else:
            max_parallel_muls = math.ceil(kernel_elems * f_fine) * math.ceil(
                self.channels * f_coarseIn
            )
            max_parallel_adds = math.ceil((kernel_elems - 1) * f_fine) * math.ceil(
                self.channels * f_coarseIn
            )

        rate_matrix_balanced, rate_matrix = self.get_rate_matrix(
            f_fine, f_coarseIn, f_coarseOut
        )
        stream_matrix = self.get_stream_matrix(f_coarseIn, f_coarseOut)
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

        if not self.depthwise:
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
                kernel_shape=[self.filters, self.channels, self.kd, self.kh, self.kw],
                coarse_in=math.ceil(self.channels * f_coarseIn),
                coarse_out=math.ceil(self.filters * f_coarseOut),
                fine=math.ceil(kernel_elems * f_fine),
                wr_factor=wr_factor
            )
        else:
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
                kernel_shape=[
                    self.filters / self.groups,
                    self.channels,
                    self.kd,
                    self.kh,
                    self.kw,
                ],
                coarse_in=math.ceil(self.channels * f_coarseIn),
                coarse_out=1,
                fine=math.ceil(kernel_elems * f_fine),
                wr_factor=wr_factor
            )

        total_ops = self.get_total_workload()
        throughput_ops = total_ops / latency_sec
        thr_in /= (
            np.prod(np.array(self.input_shape[2:])) * self.channels
        )  # Volumes per second
        thr_out /= (
            np.prod(np.array(self.output_shape[2:])) * self.filters
        )  # Volumes per second
        assert math.isclose(
            thr_in, thr_out
        ), "Thoughputs missmatch. IN = {}, OUT = {}.".format(thr_in, thr_out)

        _logger.debug(
            f"Fine: {f_fine:.3f} ({f_fine*np.prod(np.array(self.kernel_shape))}), CoarseIn: {f_coarseIn:.3f} ({int(f_coarseIn*self.channels)}), CoarseOut: {f_coarseOut if not self.depthwise else f_coarseIn:.3f} ({int(f_coarseOut*self.filters) if not self.depthwise else int(f_coarseIn*self.channels)}), Shape in: {self.input_shape}, Shape out: {self.output_shape}, Kernel: {self.kernel_shape}"
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
                f_coarseIn,
                f_coarseOut,
                # wr_factor,
            ]
            self.config = config
            self.wr_factor = wr_factor
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
                "(fine={:.2f}({}), cIn={:.2f}({}), cOut={:.2f}({}), bwIn={:.2f}, bwOut={:.2f}) DSP % = {:.2f} ({}), BRAM % = {:.2f} ({}), latency = {:.5f}({}), GOPs/s = {:.5f}, In Volumes/s = {:.5f}, Out Volumes/s = {:.5f}, depth = {}, workload(G) = {:.5f}, Mem Bound In={}, Mem Bound Out={}".format(
                    f_fine,
                    math.ceil(f_fine * kernel_elems),
                    f_coarseIn,
                    math.ceil(self.channels * f_coarseIn),
                    f_coarseOut if not self.depthwise else f_coarseIn,
                    math.ceil(self.filters * f_coarseOut)
                    if not self.depthwise
                    else math.ceil(self.channels * f_coarseIn),
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

    def get_rate_matrix(self, f_fine, f_coarseIn, f_coarseOut):
        in_volume_pad = (
            (self.depth_in + 2 * self.padding[0])
            * (self.rows_in + 2 * self.padding[1])
            * (self.cols_in + 2 * self.padding[2])
        )
        if self.depthwise:
            rate_matrix = np.zeros(shape=(5, 6), dtype=float)

            rate_matrix[0, 0] = 1

            # Sliding Window
            rate_matrix[0, 1] = 1
            rate_matrix[1, 1] = (
                self.depth_out * self.rows_out * self.cols_out
            ) / in_volume_pad

            # Fork
            rate_matrix[1, 2] = 1
            rate_matrix[2, 2] = 1

            # Convolution 3D
            rate_matrix[2, 3] = f_fine  # / math.ceil(1 / f_coarseOut)
            rate_matrix[3, 3] = f_fine

            # Accumulation

            # Glue
            rate_matrix[3, 4] = 1
            rate_matrix[4, 4] = 1

            # Concatenation
            rate_matrix[4, 5] = 1
        elif self.pointwise:
            rate_matrix = np.zeros(shape=(5, 6), dtype=float)

            rate_matrix[0, 0] = 1

            # Sliding Window

            # Fork
            rate_matrix[0, 1] = 1
            rate_matrix[1, 1] = 1

            # Convolution 3D
            rate_matrix[1, 2] = f_fine / math.ceil(1 / f_coarseOut)
            rate_matrix[2, 2] = f_fine

            # Accumulation
            rate_matrix[2, 3] = 1
            rate_matrix[3, 3] = 1 / math.ceil(1 / f_coarseIn)

            # Glue
            rate_matrix[3, 4] = 1
            rate_matrix[4, 4] = 1

            # Concatenation
            rate_matrix[4, 5] = 1
        else:
            rate_matrix = np.zeros(shape=(6, 7), dtype=float)

            rate_matrix[0, 0] = 1

            # Sliding Window
            rate_matrix[0, 1] = 1
            rate_matrix[1, 1] = (
                self.depth_out * self.rows_out * self.cols_out
            ) / in_volume_pad

            # Fork
            rate_matrix[1, 2] = 1
            rate_matrix[2, 2] = 1

            # Convolution 3D
            rate_matrix[2, 3] = f_fine / math.ceil(1 / f_coarseOut)
            rate_matrix[3, 3] = f_fine

            # Accumulation
            rate_matrix[3, 4] = 1
            rate_matrix[4, 4] = 1 / math.ceil(1 / f_coarseIn)

            # Glue
            rate_matrix[4, 5] = 1
            rate_matrix[5, 5] = 1

            # Concatenation
            rate_matrix[5, 6] = 1

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

    def get_stream_matrix(self, f_coarseIn, f_coarseOut):
        if self.depthwise:
            stream_matrix = np.zeros(shape=(5, 6), dtype=float)

            stream_matrix[0, 0] = 1

            # Sliding Window
            stream_matrix[0, 1] = math.ceil(self.channels * f_coarseIn)
            stream_matrix[1, 1] = (
                math.ceil(self.channels * f_coarseIn) * self.kd * self.kw * self.kh
            )

            # Fork
            stream_matrix[1, 2] = (
                math.ceil(self.channels * f_coarseIn) * self.kd * self.kw * self.kh
            )
            stream_matrix[2, 2] = (
                math.ceil(self.channels * f_coarseIn) * self.kd * self.kw * self.kh
            )

            # Convolution 3D
            stream_matrix[2, 3] = (
                math.ceil(self.channels * f_coarseIn) * self.kd * self.kw * self.kh
            )
            stream_matrix[3, 3] = math.ceil(self.channels * f_coarseIn)

            # Accumulation

            # Glue
            stream_matrix[3, 4] = math.ceil(self.channels * f_coarseIn)
            stream_matrix[4, 4] = math.ceil(self.channels * f_coarseIn)

            # Concatenation
            stream_matrix[4, 5] = 1
        elif self.pointwise:
            stream_matrix = np.zeros(shape=(5, 6), dtype=float)

            stream_matrix[0, 0] = 1

            # Sliding Window

            # Fork
            stream_matrix[0, 1] = (
                math.ceil(self.channels * f_coarseIn) * self.kd * self.kw * self.kh
            )
            stream_matrix[1, 1] = (
                math.ceil(self.channels * f_coarseIn)
                * math.ceil(self.filters * f_coarseOut)
                * self.kd
                * self.kw
                * self.kh
            )

            # Convolution 3D
            stream_matrix[1, 2] = (
                math.ceil(self.channels * f_coarseIn)
                * math.ceil(self.filters * f_coarseOut)
                * self.kd
                * self.kw
                * self.kh
            )
            stream_matrix[2, 2] = math.ceil(self.channels * f_coarseIn) * math.ceil(
                self.filters * f_coarseOut
            )

            # Accumulation
            stream_matrix[2, 3] = math.ceil(self.channels * f_coarseIn) * math.ceil(
                self.filters * f_coarseOut
            )
            stream_matrix[3, 3] = math.ceil(self.channels * f_coarseIn) * math.ceil(
                self.filters * f_coarseOut
            )

            # Glue
            stream_matrix[3, 4] = math.ceil(self.channels * f_coarseIn) * math.ceil(
                self.filters * f_coarseOut
            )
            stream_matrix[4, 4] = math.ceil(self.filters * f_coarseOut)

            # Concatenation
            stream_matrix[4, 5] = 1
        else:
            stream_matrix = np.zeros(shape=(6, 7), dtype=float)

            stream_matrix[0, 0] = 1

            # Sliding Window
            stream_matrix[0, 1] = math.ceil(self.channels * f_coarseIn)
            stream_matrix[1, 1] = (
                math.ceil(self.channels * f_coarseIn) * self.kd * self.kw * self.kh
            )

            # Fork
            stream_matrix[1, 2] = (
                math.ceil(self.channels * f_coarseIn) * self.kd * self.kw * self.kh
            )
            stream_matrix[2, 2] = (
                math.ceil(self.channels * f_coarseIn)
                * math.ceil(self.filters * f_coarseOut)
                * self.kd
                * self.kw
                * self.kh
            )

            # Convolution 3D
            stream_matrix[2, 3] = (
                math.ceil(self.channels * f_coarseIn)
                * math.ceil(self.filters * f_coarseOut)
                * self.kd
                * self.kw
                * self.kh
            )
            stream_matrix[3, 3] = math.ceil(self.channels * f_coarseIn) * math.ceil(
                self.filters * f_coarseOut
            )

            # Accumulation
            stream_matrix[3, 4] = math.ceil(self.channels * f_coarseIn) * math.ceil(
                self.filters * f_coarseOut
            )
            stream_matrix[4, 4] = math.ceil(self.channels * f_coarseIn) * math.ceil(
                self.filters * f_coarseOut
            )

            # Glue
            stream_matrix[4, 5] = math.ceil(self.channels * f_coarseIn) * math.ceil(
                self.filters * f_coarseOut
            )
            stream_matrix[5, 5] = math.ceil(self.filters * f_coarseOut)

            # Concatenation
            stream_matrix[5, 6] = 1

        _logger.debug("S:\n{}".format(stream_matrix))
        return stream_matrix

    def get_data_matrix(self, mem_bw_in, mem_bw_out, f_fine):
        if self.depthwise:
            data_matrix = np.zeros(shape=(5, 6), dtype=float)

            data_matrix[0, 0] = mem_bw_in

            # Sliding Window
            data_matrix[0, 1] = -1  # math.ceil(1 / f_fine)
            data_matrix[1, 1] = 1

            # Fork
            data_matrix[1, 2] = -1
            data_matrix[2, 2] = 1

            # Convolution 3D
            data_matrix[2, 3] = -1
            data_matrix[3, 3] = 1

            # Accumulation

            # Glue
            data_matrix[3, 4] = -1
            data_matrix[4, 4] = 1

            # Concatenation
            data_matrix[4, 5] = -mem_bw_out
        elif self.pointwise:
            data_matrix = np.zeros(shape=(5, 6), dtype=float)

            data_matrix[0, 0] = mem_bw_in

            # Sliding Window

            # Fork
            data_matrix[0, 1] = -1
            data_matrix[1, 1] = 1

            # Convolution 3D
            data_matrix[1, 2] = -1
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
            data_matrix = np.zeros(shape=(6, 7), dtype=float)

            data_matrix[0, 0] = mem_bw_in

            # Sliding Window
            data_matrix[0, 1] = -1
            data_matrix[1, 1] = 1

            # Fork
            data_matrix[1, 2] = -1
            data_matrix[2, 2] = 1

            # Convolution 3D
            data_matrix[2, 3] = -1
            data_matrix[3, 3] = 1

            # Accumulation
            data_matrix[3, 4] = -1
            data_matrix[4, 4] = 1

            # Glue
            data_matrix[4, 5] = -1
            data_matrix[5, 5] = 1

            # Concatenation
            data_matrix[5, 6] = -mem_bw_out

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

        if self.depthwise:
            workload_matrix = np.zeros(shape=(5, 6), dtype=float)

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
            workload_matrix = np.zeros(shape=(5, 6), dtype=float)

            workload_matrix[0, 0] = in_volume * self.channels

            # Sliding Window

            # Fork
            workload_matrix[0, 1] = out_volume * kernel_volume * self.channels
            workload_matrix[1, 1] = out_volume * kernel_volume * self.channels

            # Convolution 3D
            workload_matrix[1, 2] = out_volume * kernel_volume * self.channels
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
            workload_matrix = np.zeros(shape=(6, 7), dtype=float)

            workload_matrix[0, 0] = in_volume * self.channels

            # Sliding Window
            workload_matrix[0, 1] = in_volume_pad * self.channels
            workload_matrix[1, 1] = out_volume * kernel_volume * self.channels

            # Fork
            workload_matrix[1, 2] = out_volume * kernel_volume * self.channels
            workload_matrix[2, 2] = out_volume * kernel_volume * self.channels

            # Convolution 3D
            workload_matrix[2, 3] = out_volume * kernel_volume * self.channels
            workload_matrix[3, 3] = out_volume * self.channels * self.filters

            # Accumulation
            workload_matrix[3, 4] = out_volume * self.channels * self.filters
            workload_matrix[4, 4] = out_volume * self.filters

            # Glue
            workload_matrix[4, 5] = out_volume * self.filters
            workload_matrix[5, 5] = out_volume * self.filters

            # Concatenation
            workload_matrix[5, 6] = out_volume * self.filters

        _logger.debug("WL:\n{}".format(workload_matrix))
        return workload_matrix
