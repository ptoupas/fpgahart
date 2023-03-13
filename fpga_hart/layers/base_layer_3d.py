import bisect
import configparser
import logging
import math
import os

import numpy as np

from fpga_hart import _logger
from fpga_hart.platform.platform import Platform


class BaseLayer3D:
    def __init__(self, max_DSP_util=95.0, max_BRAM_util=95.0, data_format="NHWDC"):
        assert (
            data_format == "NHWDC" or data_format == "NCHWD"
        ), "Wrong data format. Accepted formats are 'NHWDC' or 'NCHWD'"
        # _logger.setLevel(level=logging.DEBUG)

        self.data_format = data_format

        self.platform = Platform()

        self.word_length = self.platform.word_length
        self.clock_freq = self.platform.clock_freq
        self.bram = self.platform.bram
        self.bram_Kbytes = self.platform.bram_Kbytes
        self.dsp = self.platform.dsp
        self.mem_bw = self.platform.mem_bw
        self.fpga_device = self.platform.fpga_device
        self.word_bytes = self.platform.word_bytes
        self.cycles_per_sec = self.platform.cycles_per_sec
        self.mem_bandwidth = self.platform.mem_bandwidth
        self.mem_words_per_cycle = self.platform.mem_words_per_cycle

        self.max_DSP_util = max_DSP_util
        self.max_BRAM_util = max_BRAM_util
        self.BRAM_CONF_WIDTH = {1: 16384, 2: 8192, 4: 4096, 9: 2048, 18: 1024, 36: 512}
        self.BRAM_CONF_DEPTH = {16384: 1, 8192: 2, 4096: 4, 2048: 9, 1024: 18, 512: 36}

    def bram_stream_resource_model(self, depth, width):
        assert width > 0, "width must be greater than zero"
        assert width <= 36, "width must be less than 36"

        # if there is zero depth, return no BRAM usage
        if depth == 0:
            return 0

        # initialize the BRAM configuration
        rest_depth = 0
        brams_total = 0

        # find the closest depth from the BRAM configuration
        if depth in list(self.BRAM_CONF_DEPTH.keys()):
            bram_depth = depth
        elif depth > sorted(list(self.BRAM_CONF_DEPTH.keys()))[-1]:
            bram_depth = sorted(list(self.BRAM_CONF_DEPTH.keys()))[-1]
            rest_depth = depth - bram_depth
        else:
            bram_depth = sorted(list(self.BRAM_CONF_DEPTH.keys()))[
                bisect.bisect_right(sorted(list(self.BRAM_CONF_DEPTH.keys())), depth)
            ]

        # get the depth for the bram
        bram_width = self.BRAM_CONF_DEPTH[bram_depth]
        brams_total += math.ceil(width / bram_width)

        while rest_depth > 0:
            if rest_depth in list(self.BRAM_CONF_DEPTH.keys()):
                bram_depth = rest_depth
                rest_depth = 0
            elif rest_depth > sorted(list(self.BRAM_CONF_DEPTH.keys()))[-1]:
                bram_depth = sorted(list(self.BRAM_CONF_DEPTH.keys()))[-1]
                rest_depth = rest_depth - bram_depth
            else:
                bram_depth = sorted(list(self.BRAM_CONF_DEPTH.keys()))[
                    bisect.bisect_right(
                        sorted(list(self.BRAM_CONF_DEPTH.keys())), rest_depth
                    )
                    - 1
                ]
                rest_depth = rest_depth - bram_depth

            if rest_depth > 0:
                # get the width for the bram
                bram_width = self.BRAM_CONF_DEPTH[bram_depth]
                brams_total += math.ceil(width / bram_width)

        # return the ceiling
        return brams_total

    def bram_memory_resource_model(self, depth, width):
        assert width > 0, "width must be greater than zero"
        assert width <= 36, "width must be less than 36"

        # if there is zero depth, return no BRAM usage
        if depth == 0:
            return 0

        # find the closest width from the BRAM configuration
        if width in list(self.BRAM_CONF_WIDTH.keys()):
            bram_width = width
        else:
            bram_width = sorted(list(self.BRAM_CONF_WIDTH.keys()))[
                bisect.bisect_right(sorted(list(self.BRAM_CONF_WIDTH.keys())), width)
            ]

        # get the depth for the bram
        bram_depth = self.BRAM_CONF_WIDTH[bram_width]

        # return the ceiling
        return math.ceil(depth / bram_depth)

    def dsp_multiplier_resource_model(
        self, multiplicand_width, multiplier_width, dsp_type="DSP48E1"
    ):
        return math.ceil((multiplicand_width + multiplier_width) / 48)

    def get_dp_performance(
        self,
        workload_matrix,
        ii,
        muls,
        adds,
        layer_fifos_arrays,
        depth,
        batch=1,
        kernel_shape=[],
        coarse_in=1,
        coarse_out=1,
        fine=1,
        coarse_inout=1,
        wr_factor=1
    ):
        # mem_kb = (mem * self.word_bytes) / 1e3
        # mem_bram = math.ceil(mem_kb / self.bram_Kbytes) #* coarse_in * coarse_out
        mem_kb_total = 0
        bram_raw = 0

        if "sw_lb_3d" in layer_fifos_arrays.keys():
            filters, channels, kd, kh, kw = kernel_shape
            line_buffer_3d_brams = self.bram_stream_resource_model(
                layer_fifos_arrays["sw_lb_3d"], 16
            )
            line_buffer_2d_brams = self.bram_stream_resource_model(
                layer_fifos_arrays["sw_lb_2d"], 16
            )
            window_buffer_3d_brams = self.bram_stream_resource_model(
                layer_fifos_arrays["sw_wb_3d"], 16
            )

            sw_brams = (
                kh * (kw - 1) * line_buffer_3d_brams
                + (kh - 1) * line_buffer_2d_brams
                + kh * kw * (kd - 1) * window_buffer_3d_brams
            )
            # print("line_buffer_3d_brams:", line_buffer_3d_brams, kh*(kw-1) * line_buffer_3d_brams)
            # print("line_buffer_2d_brams:", layer_fifos_arrays['sw_lb_2d'], line_buffer_2d_brams, (kh-1) * line_buffer_2d_brams)
            # print("window_buffer_3d_brams:", window_buffer_3d_brams, kh*kw*(kd-1) * window_buffer_3d_brams)
            # print("SW:", sw_brams, coarse_in * sw_brams)

            if "acc_fifo" in layer_fifos_arrays.keys():
                fifo_accumulator_brams = self.bram_stream_resource_model(
                    layer_fifos_arrays["acc_fifo"], 30
                )
                array_accumulator_brams = self.bram_memory_resource_model(
                    layer_fifos_arrays["acc_array"], 30
                )
                if layer_fifos_arrays["acc_fifo"] < 100:
                    fifo_accumulator_brams = 0
                if layer_fifos_arrays["acc_array"] < 100:
                    array_accumulator_brams = 0
                # print("ACC:", (fifo_accumulator_brams + array_accumulator_brams), (fifo_accumulator_brams + array_accumulator_brams) * coarse_in * coarse_out)

            weights_depth = int(
                (kd * kh * kw * channels * filters) / (fine * coarse_in * coarse_out)
            )
            weights_bram = self.bram_memory_resource_model(weights_depth, 16)
            if weights_depth < 100:
                weights_bram = 0
            # print(f"WEIGHTS: depth={weights_depth},\tbram={weights_bram},\tcoarse_factors={fine * coarse_in * coarse_out},\ttotal_bram={weights_bram * fine * coarse_in * coarse_out}\t{sw_brams * coarse_in}\t{(fifo_accumulator_brams + array_accumulator_brams) * coarse_in * coarse_out}")

            bram_raw += (
                sw_brams * coarse_in
                + (fifo_accumulator_brams + array_accumulator_brams)
                * coarse_in
                * coarse_out
                + weights_bram * fine * coarse_in * coarse_out
            )
        if "pool_sw_lb_3d" in layer_fifos_arrays.keys():
            filters, channels, kd, kh, kw = kernel_shape
            line_buffer_3d_brams = self.bram_stream_resource_model(
                layer_fifos_arrays["pool_sw_lb_3d"], 16
            )
            line_buffer_2d_brams = self.bram_stream_resource_model(
                layer_fifos_arrays["pool_sw_lb_2d"], 16
            )
            window_buffer_3d_brams = self.bram_stream_resource_model(
                layer_fifos_arrays["pool_sw_wb_3d"], 16
            )

            sw_brams = (
                kh * (kw - 1) * line_buffer_3d_brams
                + (kh - 1) * line_buffer_2d_brams
                + kh * kw * (kd - 1) * window_buffer_3d_brams
            )
            bram_raw += sw_brams * coarse_inout
        if "elemwise_bc" in layer_fifos_arrays.keys():
            array_elemwise_brams = self.bram_memory_resource_model(
                layer_fifos_arrays["elemwise_bc"], 30
            )
            if layer_fifos_arrays["elemwise_bc"] < 100:
                array_elemwise_brams = 0
            bram_raw += array_elemwise_brams * coarse_inout

        if "fc_array" in layer_fifos_arrays.keys():
            array_fc_brams = self.bram_memory_resource_model(
                layer_fifos_arrays["fc_array"], 30
            )
            if layer_fifos_arrays["fc_array"] < 100:
                array_fc_brams = 0
            bram_raw += array_fc_brams * coarse_out # coarse_in

        if "gap_array" in layer_fifos_arrays.keys():
            array_gap_brams = self.bram_memory_resource_model(
                layer_fifos_arrays["gap_array"], 30
            )
            if layer_fifos_arrays["gap_array"] < 100:
                array_gap_brams = 0
            bram_raw += array_gap_brams * coarse_inout

        bram_util = (bram_raw / self.bram) * 100
        dsps_util = (muls / self.dsp) * 100
        dsp_raw = muls

        # latency_cycles = (np.max(np.abs(ii))) * batch + depth
        latency_cycles = ((np.max(np.abs(ii))) * batch + depth) * wr_factor + (wr_factor - 1) * np.prod(np.array(kernel_shape))
        latency_sec = latency_cycles / self.cycles_per_sec

        thr_in = (batch * workload_matrix[0, 0]) / latency_sec  # Input words per second
        thr_out = (
            batch * workload_matrix[-1, -1]
        ) / latency_sec  # Output words per second

        return (
            latency_sec,
            latency_cycles,
            thr_in,
            thr_out,
            dsps_util,
            dsp_raw,
            bram_util,
            bram_raw,
            mem_kb_total,
        )

    def balance_matrix(self, matrix):
        rate_ratio = [
            abs(matrix[i, i] / matrix[i - 1, i]) for i in range(1, matrix.shape[1] - 1)
        ]

        for i in range(1, matrix.shape[0] - 1):
            layer = matrix.shape[0] - i
            if abs(matrix[layer - 1, layer]) > matrix[layer - 1, layer - 1]:
                # propogate forward
                for j in range(layer, matrix.shape[0]):
                    if abs(matrix[j - 1, j]) <= matrix[j - 1, j - 1]:
                        break
                    matrix[j - 1, j] = -matrix[j - 1, j - 1]
                    if j < matrix.shape[0]:
                        matrix[j, j] = matrix[j - 1, j - 1] * rate_ratio[j - 1]
            elif abs(matrix[layer - 1, layer]) < matrix[layer - 1, layer - 1]:
                # propogate backward
                for j in range(0, layer):
                    if (
                        abs(matrix[layer - j - 1, layer - j])
                        >= matrix[layer - j - 1, layer - j - 1]
                    ):
                        break
                    matrix[layer - j - 1, layer - j - 1] = abs(
                        matrix[layer - j - 1, layer - j]
                    )
                    if layer - j - 1 > 0:
                        matrix[layer - j - 2, layer - j - 1] = (
                            -matrix[layer - j - 1, layer - j - 1]
                            / rate_ratio[layer - 1 - j - 1]
                        )

        rate_ratio_new = [
            abs(matrix[i, i] / matrix[i - 1, i]) for i in range(1, matrix.shape[1] - 1)
        ]
        assert np.allclose(rate_ratio, rate_ratio_new), "{} - {}".format(
            rate_ratio, rate_ratio_new
        )

        mem_bounded_in = False
        mem_bounded_out = False

        if matrix[0, 0] < abs(matrix[0, 1]):
            mem_bounded_in = True
            for i in range(0, matrix.shape[0]):
                layer = matrix.shape[0] - i
                if abs(matrix[layer - 1, layer]) > matrix[layer - 1, layer - 1]:
                    # propogate forward
                    for j in range(layer, matrix.shape[0] + 1):
                        if abs(matrix[j - 1, j]) <= matrix[j - 1, j - 1]:
                            break
                        matrix[j - 1, j] = -matrix[j - 1, j - 1]
                        if j < matrix.shape[0]:
                            matrix[j, j] = matrix[j - 1, j - 1] * rate_ratio[j - 1]
                elif abs(matrix[layer - 1, layer]) < matrix[layer - 1, layer - 1]:
                    # propogate backward
                    for j in range(0, layer):
                        if (
                            abs(matrix[layer - j - 1, layer - j])
                            >= matrix[layer - j - 1, layer - j - 1]
                        ):
                            break
                        matrix[layer - j - 1, layer - j - 1] = abs(
                            matrix[layer - j - 1, layer - j]
                        )
                        if layer - j - 1 > 0:
                            matrix[layer - j - 2, layer - j - 1] = (
                                -matrix[layer - j - 1, layer - j - 1]
                                / rate_ratio[layer - 1 - j - 1]
                            )

            rate_ratio_new_2 = [
                abs(matrix[i, i] / matrix[i - 1, i])
                for i in range(1, matrix.shape[1] - 1)
            ]
            assert np.allclose(rate_ratio_new, rate_ratio_new_2), "{} - {}".format(
                rate_ratio_new, rate_ratio_new_2
            )

        if abs(matrix[-1, -1]) < matrix[-1, -2]:
            mem_bounded_out = True

            for i in range(0, matrix.shape[0]):
                layer = matrix.shape[0] - i
                if abs(matrix[layer - 1, layer]) > matrix[layer - 1, layer - 1]:
                    # propogate forward
                    for j in range(layer, matrix.shape[0] + 1):
                        if abs(matrix[j - 1, j]) <= matrix[j - 1, j - 1]:
                            break
                        matrix[j - 1, j] = -matrix[j - 1, j - 1]
                        if j < matrix.shape[0]:
                            matrix[j, j] = matrix[j - 1, j - 1] * rate_ratio[j - 1]
                elif abs(matrix[layer - 1, layer]) < matrix[layer - 1, layer - 1]:
                    # propogate backward
                    for j in range(0, layer):
                        if (
                            abs(matrix[layer - j - 1, layer - j])
                            >= matrix[layer - j - 1, layer - j - 1]
                        ):
                            break
                        matrix[layer - j - 1, layer - j - 1] = abs(
                            matrix[layer - j - 1, layer - j]
                        )
                        if layer - j - 1 > 0:
                            matrix[layer - j - 2, layer - j - 1] = (
                                -matrix[layer - j - 1, layer - j - 1]
                                / rate_ratio[layer - 1 - j - 1]
                            )

            rate_ratio_new_2 = [
                abs(matrix[i, i] / matrix[i - 1, i])
                for i in range(1, matrix.shape[1] - 1)
            ]
            assert np.allclose(rate_ratio_new, rate_ratio_new_2), "{} - {}".format(
                rate_ratio_new, rate_ratio_new_2
            )

        if not mem_bounded_in and not mem_bounded_out:
            matrix[0, 0] = abs(matrix[0, 1])
            matrix[-1, -1] = -matrix[-1, -1 - 1]

        return matrix, mem_bounded_in, mem_bounded_out

    def balance_matrix_elemwise(self, matrix, branch_node):
        mem_bounded_in_1 = False
        mem_bounded_in_2 = False
        mem_bounded_out = False

        branch_ratio = abs(matrix[0, branch_node]) / abs(
            matrix[branch_node - 1, branch_node]
        )

        if (
            abs(matrix[branch_node - 1, branch_node])
            > matrix[branch_node - 1, branch_node - 1]
        ):
            mem_bounded_in_2 = True
            matrix[branch_node - 1, branch_node] = -matrix[
                branch_node - 1, branch_node - 1
            ]
        else:
            matrix[branch_node - 1, branch_node - 1] = abs(
                matrix[branch_node - 1, branch_node]
            )

        if abs(matrix[0, branch_node]) > matrix[0, 0]:
            mem_bounded_in_1 = True
            matrix[0, branch_node] = -matrix[0, 0]
        else:
            matrix[0, 0] = abs(matrix[0, branch_node])

        matrix[branch_node, branch_node] = min(
            abs(matrix[0, branch_node]),
            abs(matrix[branch_node - 1, branch_node]),
            matrix[branch_node, branch_node],
        )

        if abs(matrix[-1, -1]) < matrix[-1, -2]:
            mem_bounded_out = True
            matrix[branch_node, branch_node] = abs(matrix[-1, -1])
        else:
            matrix[-1, -1] = -matrix[branch_node, branch_node]

        if matrix[branch_node, branch_node] <= abs(
            matrix[branch_node - 1, branch_node]
        ):
            matrix[branch_node - 1, branch_node] = -matrix[branch_node, branch_node]
            matrix[branch_node - 1, branch_node - 1] = matrix[branch_node, branch_node]
        else:
            assert False, "Failed to move backwards on Γ matrix for input 2"

        if matrix[branch_node, branch_node] <= abs(matrix[0, branch_node]):
            matrix[0, branch_node] = -matrix[branch_node, branch_node]
            matrix[0, 0] = matrix[branch_node, branch_node]
        else:
            assert False, "Failed to move backwards on Γ matrix for input 1"

        assert abs(matrix[0, branch_node]) == abs(
            matrix[branch_node - 1, branch_node]
        ), "Problem with the graph balancing"
        assert abs(matrix[0, branch_node]) == abs(
            matrix[branch_node, branch_node]
        ), "Problem with the graph balancing"

        return matrix, mem_bounded_in_1, mem_bounded_in_2, mem_bounded_out

    def balance_matrix_elemwise_broadcasting(self, matrix, branch_node, branch_ratio):
        mem_bounded_in_1 = False
        mem_bounded_in_2 = False
        mem_bounded_out = False

        curr_branch_ratio = 1 / (
            abs(matrix[0, branch_node]) / abs(matrix[branch_node - 1, branch_node])
        )
        if curr_branch_ratio < branch_ratio:
            matrix[0, branch_node] = (
                -abs(matrix[branch_node - 1, branch_node]) / branch_ratio
            )
        elif curr_branch_ratio > branch_ratio:
            matrix[branch_node - 1, branch_node] = (
                -abs(matrix[0, branch_node]) * branch_ratio
            )

        if (
            abs(matrix[branch_node - 1, branch_node])
            > matrix[branch_node - 1, branch_node - 1]
        ):
            mem_bounded_in_2 = True
            matrix[branch_node - 1, branch_node] = -matrix[
                branch_node - 1, branch_node - 1
            ]
            matrix[0, branch_node] = (
                -abs(matrix[branch_node - 1, branch_node]) / branch_ratio
            )
        else:
            matrix[branch_node - 1, branch_node - 1] = abs(
                matrix[branch_node - 1, branch_node]
            )

        if abs(matrix[0, branch_node]) > matrix[0, 0]:
            mem_bounded_in_1 = True
            matrix[0, branch_node] = -matrix[0, 0]
            matrix[branch_node - 1, branch_node] = (
                -abs(matrix[0, branch_node]) * branch_ratio
            )
        else:
            matrix[0, 0] = abs(matrix[0, branch_node])

        matrix[branch_node, branch_node] = abs(matrix[0, branch_node])

        if abs(matrix[-1, -1]) < matrix[-1, -2]:
            mem_bounded_out = True
            matrix[branch_node, branch_node] = abs(matrix[-1, -1])
        else:
            matrix[-1, -1] = -matrix[branch_node, branch_node]

        curr_branch_ratio = 1 / (
            abs(matrix[branch_node, branch_node])
            / abs(matrix[branch_node - 1, branch_node])
        )
        if curr_branch_ratio > branch_ratio:
            matrix[branch_node - 1, branch_node] = (
                -abs(matrix[branch_node, branch_node]) * branch_ratio
            )
            matrix[branch_node - 1, branch_node - 1] = (
                abs(matrix[branch_node, branch_node]) * branch_ratio
            )

            matrix[0, branch_node] = -matrix[branch_node, branch_node]
            matrix[0, 0] = matrix[branch_node, branch_node]

        curr_branch_ratio = 1 / (
            abs(matrix[0, branch_node]) / abs(matrix[branch_node - 1, branch_node])
        )
        assert branch_ratio == curr_branch_ratio, "Problem with the graph balancing"
        curr_branch_ratio = 1 / (
            abs(matrix[branch_node, branch_node])
            / abs(matrix[branch_node - 1, branch_node])
        )
        assert branch_ratio == curr_branch_ratio, "Problem with the graph balancing"

        return matrix, mem_bounded_in_1, mem_bounded_in_2, mem_bounded_out
