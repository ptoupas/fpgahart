
import copy
import math
import random
import time
from copy import deepcopy

import numpy as np

from fpga_hart import _logger
from fpga_hart.layers.activation_3d import Activation3DLayer
from fpga_hart.layers.batchnorm_3d import BatchNorm3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise_3d import ElementWise3DLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap_3d import GAP3DLayer
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.layers.squeeze_excitation import SqueezeExcitationLayer
from fpga_hart.utils import utils

def initialize_optimizer_layer(self, layer, wr_factor: int = 1):
    config, mem_bw = self.generate_random_config_layer(layer)
    cost, dp_info = self.get_cost_layer(config, mem_bw, layer, wr_factor=wr_factor)

    if cost is None:
        start_time = time.time()
        while time.time() - start_time < 90.0:
            x = float(time.time() - start_time)
            perc = 1/(1+math.exp(-0.1*(x-45)))
            config, mem_bw = self.generate_random_config_layer(layer,
                keep_percentage=perc)
            cost, dp_info = self.get_cost_layer(config, mem_bw, layer, wr_factor=wr_factor)

            if cost is not None:
                break
        if cost is None:
            print("No configuration found after 90 seconds. Aborting...")
            return None, None, None

    return config, cost, dp_info, mem_bw

def run_optimizer_layer(self, layer):
    """Run the optimizer for a single layer."""

    hw = self.graph.nodes[layer]["hw"]
    wr_factor = 1
    if isinstance(hw, Convolutional3DLayer):
        initial_filters = deepcopy(hw.filters)
        coarsein_min = 1 / np.int32(hw.channels)
        coarseout_min = 1 / np.int32(hw.filters)
        fine_min = 1 / np.prod(np.array(hw.kernel_shape))
        _, bram_util = hw.get_resource_util(f_fine = fine_min,
                                        f_coarseIn = coarsein_min,
                                        f_coarseOut= coarseout_min)
        print("Initial BRAM utilization: ", bram_util)
        if bram_util > self.config.max_bram_util:
            _logger.warning(f"Layer's ({layer}) minimum BRAM utilization is above the device's maximum on chip memory resources.\nSplit the layer execution into multiple instances (weights reloading).")
            for f in utils.get_factors(initial_filters)[1:]:
                new_out_shape = deepcopy(hw.output_shape)
                new_out_shape[1] = int(initial_filters/f)
                hw.update_shapes(hw.input_shape, new_out_shape)
                coarsein_min = 1 / np.int32(hw.channels)
                coarseout_min = 1 / np.int32(hw.filters)
                fine_min = 1 / np.prod(np.array(hw.kernel_shape))
                _, bram_util = hw.get_resource_util(f_fine = fine_min,
                                                f_coarseIn = coarsein_min,
                                                f_coarseOut= coarseout_min)
                if bram_util < self.config.max_bram_util:
                    wr_factor = f
                    break
            if wr_factor == 1:
                return None

    config, cost, dp_info, mem_bw = self.initialize_optimizer_layer(layer, wr_factor=wr_factor)
    if config == None:
        return None

    prev_state = config
    solution_dp = dp_info
    solution_mem = mem_bw
    prev_cost = cost

    current_temp = self.t_max

    print(f"Temperature  |  Latency")
    while current_temp > self.t_min:

        for i in range(self.iterationPerTemp):
            new_state, new_mem_bw = self.generate_random_config_layer(layer)
            new_cost, new_dp_info = self.get_cost_layer(
                new_state, new_mem_bw, layer, wr_factor=wr_factor
            )

            if new_cost is None:
                continue

            cost_diff = prev_cost - new_cost
            if cost_diff > 0:
                prev_state = copy.deepcopy(new_state)
                prev_cost = copy.deepcopy(new_cost)
                solution_mem, solution_dp = (
                    copy.deepcopy(new_mem_bw),
                    copy.deepcopy(new_dp_info),
                )
            else:
                if random.uniform(0, 1) < math.exp(
                    (cost_diff / (self.k * current_temp))
                ):
                    prev_state = copy.deepcopy(new_state)
                    prev_cost = copy.deepcopy(new_cost)
                    solution_mem, solution_dp = (
                        copy.deepcopy(new_mem_bw),
                        copy.deepcopy(new_dp_info),
                    )

        current_temp *= self.cooling_rate
        keep_percentage = 1/(1+math.exp(-2*(current_temp-0.7))) * 100
        print(f"{current_temp:.5e}\t{prev_cost:.5e}", end="\r")

    print(
        f"\n\nLatency: {prev_cost}.\nFinal Memory IN {list(np.array(solution_mem[0]) * self.platform.mem_words_per_cycle)}, Memory OUT {list(np.array(solution_mem[1]) * self.platform.mem_words_per_cycle)}."
    )
    print(
        "Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={}({:.2f}), BRAM(%)={}({:.2f}), rateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}\nPartition Configuration: {}".format(
            solution_dp["latency(C)"],
            solution_dp["latency(S)"],
            solution_dp["GOP/s"],
            solution_dp["vols/s"],
            int((solution_dp["DSP"] / 100) * self.graph.nodes[layer]["hw"].dsp),
            solution_dp["DSP"],
            int((solution_dp["BRAM"] / 100) * self.graph.nodes[layer]["hw"].bram),
            solution_dp["BRAM"],
            solution_dp["rateIn"],
            solution_dp["rateOut"],
            solution_dp["depth"],
            solution_dp["muls"],
            solution_dp["adds"],
            solution_dp["memWords"],
            solution_dp["memKBs"],
            solution_dp["memBoundedIn"],
            solution_dp["memBoundedOut"],
            solution_dp["config"],
        )
    )
    print("*" * 60)
    return solution_dp

def get_cost_layer(self, config, mem_bw, layer, wr_factor: int=1):
    hw = self.graph.nodes[layer]["hw"]
    if isinstance(hw, GAP3DLayer):
        dp_info = hw.get_design_point(
            config[0],
            hw.mem_words_per_cycle * mem_bw[0][0],
            hw.mem_words_per_cycle * mem_bw[1][0],
        )
    elif isinstance(hw, Convolutional3DLayer):
        dp_info = hw.get_design_point(
            config[0],
            config[1],
            config[2],
            hw.mem_words_per_cycle * mem_bw[0][0],
            hw.mem_words_per_cycle * mem_bw[1][0],
            wr_factor=wr_factor
        )
    elif isinstance(hw, Pooling3DLayer):
        dp_info = hw.get_design_point(
            config[0],
            config[1],
            hw.mem_words_per_cycle * mem_bw[0][0],
            hw.mem_words_per_cycle * mem_bw[1][0],
        )
    elif isinstance(hw, Activation3DLayer):
        dp_info = hw.get_design_point(
            config[0],
            hw.mem_words_per_cycle * mem_bw[0][0],
            hw.mem_words_per_cycle * mem_bw[1][0],
        )
    elif isinstance(hw, ElementWise3DLayer):
        dp_info = hw.get_design_point(
            config[0],
            hw.mem_words_per_cycle * mem_bw[0][0],
            hw.mem_words_per_cycle * mem_bw[0][1],
            hw.mem_words_per_cycle * mem_bw[1][0],
        )
    elif isinstance(hw, BatchNorm3DLayer):
        dp_info = hw.get_design_point(
            config[0],
            hw.mem_words_per_cycle * mem_bw[0][0],
            hw.mem_words_per_cycle * mem_bw[1][0],
        )
    elif isinstance(hw, SqueezeExcitationLayer):
        assert False, "Not supported layer (SqueezeExcitationLayer)"
    elif isinstance(hw, FCLayer):
        dp_info = hw.get_design_point(
            config[0],
            config[1],
            hw.mem_words_per_cycle * mem_bw[0][0],
            hw.mem_words_per_cycle * mem_bw[1][0],
        )
    else:
        assert False, "Not supported layer"

    if dp_info["config"]:
        return dp_info["latency(S)"], dp_info
        # return -dp_info['GOP/s'], dp_info
    return None, None

def generate_random_config_layer(self, l: str, keep_percentage: float = -1):
    config = []
    hw = self.graph.nodes[l]["hw"]
    if isinstance(hw, GAP3DLayer):
        channels = hw.channels
        filters = hw.filters
        coarse_inout_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
        coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
        config = [coarse_inout_factor]
    elif isinstance(hw, Convolutional3DLayer):
        channels = hw.channels
        filters = hw.filters
        kernel_size = hw.kernel_shape
        coarse_in_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
        coarse_out_feasible = utils.get_factors(filters, keep_percentage=keep_percentage)
        fine_feasible = utils.get_fine_feasible(kernel_size, keep_percentage=keep_percentage)
        coarse_in_factor = random.choice(coarse_in_feasible) / channels
        coarse_out_factor = random.choice(coarse_out_feasible) / filters
        fine_factor = random.choice(fine_feasible) / np.prod(np.array(kernel_size))
        config = [fine_factor, coarse_in_factor, coarse_out_factor]
    elif isinstance(hw, Pooling3DLayer):
        channels = hw.channels
        kernel_size = hw.kernel_shape
        coarse_inout_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
        fine_feasible = utils.get_fine_feasible(kernel_size, keep_percentage=keep_percentage)
        coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
        fine_factor = random.choice(fine_feasible) / np.prod(np.array(kernel_size))
        config = [coarse_inout_factor, fine_factor]
    elif isinstance(hw, Activation3DLayer):
        channels = hw.channels
        coarse_inout_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
        coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
        config = [coarse_inout_factor]
    elif isinstance(hw, ElementWise3DLayer):
        channels = hw.channels_1
        coarse_inout_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
        coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
        config = [coarse_inout_factor]
    elif isinstance(hw, BatchNorm3DLayer):
        channels = hw.channels
        coarse_inout_feasible = utils.get_factors(channels, keep_percentage=keep_percentage)
        coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
        config = [coarse_inout_factor]
    elif isinstance(hw, SqueezeExcitationLayer):
        assert False, "Not supported layer (SqueezeExcitationLayer)"
    elif isinstance(hw, FCLayer):
        dim_in = hw.dim_in
        dim_out = hw.dim_out
        coarse_in_feasible = utils.get_factors(dim_in, keep_percentage=keep_percentage)
        coarse_out_feasible = utils.get_factors(dim_out, keep_percentage=keep_percentage)
        coarse_in_factor = random.choice(coarse_in_feasible) / dim_in
        coarse_out_factor = random.choice(coarse_out_feasible) / dim_out
        config = [coarse_in_factor, coarse_out_factor]
    else:
        assert False, "Not supported layer"

    if isinstance(hw, ElementWise3DLayer):
        mem_config_in, mem_config_out = self.get_mem_bw_feasible(n_in=2, n_out=1)
    else:
        mem_config_in, mem_config_out = self.get_mem_bw_feasible(n_in=1, n_out=1)

    return config, [mem_config_in, mem_config_out]