import math
import random
from copy import deepcopy

import numpy as np
import scipy.constants as sc

import wandb
from fpga_hart import _logger
from fpga_hart.layers.activation_3d import Activation3DLayer
from fpga_hart.layers.batchnorm_3d import BatchNorm3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise_3d import ElementWise3DLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap_3d import GAP3DLayer
from fpga_hart.layers.memory_interface import MemoryNode
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.layers.squeeze_excitation import SqueezeExcitationLayer
from fpga_hart.partitions.partition_compose import PartitionComposer
from fpga_hart.utils import utils


class SimulatedAnnealing():

    def __init__(
        self,
        graph,
        config,
        platform,
        branch_mem=0,
        partition_name="",
        gap_approx=False,
        cnn_model_name="",
        enable_wandb=False,
    ):
        # _logger.setLevel(level=logging.DEBUG)
        self.cnn_model_name = cnn_model_name
        self.config = config
        self.platform = platform
        self.enable_wandb = enable_wandb

        self.gap_approx = gap_approx
        self.part_name = partition_name

        self.graph = graph

        mem_kb = (branch_mem * self.platform.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.platform.bram_Kbytes)
        self.branch_bram_util = (mem_bram / self.platform.bram) * 100
        self.branch_mem = branch_mem

        # Simulate Annealing Variables
        self.k = sc.Boltzmann
        self.t_min = self.config.simulatedAnnealing["t_min"]
        self.t_max = self.config.simulatedAnnealing["t_max"]
        self.cooling_rate = self.config.simulatedAnnealing["cooling_rate"]
        self.iterationPerTemp = self.config.simulatedAnnealing["iterationPerTemp"]
        self.best_of_iter = self.config.simulatedAnnealing["best_of_iter"]
        self.block_gen = self.config.bblock_generation
        self.bblock_keep_percentage = self.config.bblock_keep_percentage
        self.use_arbitrary_shape = self.config.use_arbitrary_shape
        self.use_previous_config = self.config.use_previous_config
        self.chan_dist_thresh = self.config.chan_dist_thresh
        self.depth_dist_thresh = self.config.depth_dist_thresh
        self.height_dist_thresh = self.config.height_dist_thresh

        self.param_changes = 0
        self.freeze_param = False

        self.partition_composer = PartitionComposer(
            max_DSP_util=self.config.max_dsp_util, max_BRAM_util=self.config.max_bram_util
        )

    from fpga_hart.optimizer.simulated_annealing.sa_latency import (
        generate_building_blocks, generate_building_blocks_config,
        get_cost_latency, run_optimizer_latency,
        validate_building_blocks_setup)
    from fpga_hart.optimizer.simulated_annealing.sa_layer import (
        generate_random_config_layer, get_cost_layer,
        initialize_optimizer_layer, run_optimizer_layer)
    from fpga_hart.optimizer.simulated_annealing.sa_partition import (
        generate_random_config_partition, get_cost_partition,
        initialize_optimizer_partition, run_optimizer_partition,
        run_optimizer_partition_double_graph)

    def run_solver(self, mode, layer=None, alignedfactors=None):
        if mode == "partition":
            # if has_gap(self.graph) and self.branch_bram_util > self.config.max_bram_util:
            #     return self.run_optimizer_partition_double_graph()
            # else:
            return self.run_optimizer_partition()
        elif mode == "layer":
            return self.run_optimizer_layer(layer=layer)
        elif mode == "latency":
            return self.run_optimizer_latency(alignedfactors=alignedfactors)
        else:
            raise ValueError(f"Mode {mode} is not supported")

    def validate_configs(self, graph_1_dp, graph_2_dp):
        g_1_dsp_util = graph_1_dp["DSP"]
        g_2_dsp_util = graph_2_dp["DSP"]

        g_1_bram_util = graph_1_dp["BRAM"]
        g_2_bram_util = graph_2_dp["BRAM"]

        if g_1_dsp_util + g_2_dsp_util >= self.config.max_dsp_util:
            return False
        if g_1_bram_util + g_2_bram_util >= self.config.max_bram_util:
            return False

        return True

    @staticmethod
    def get_mem_bw_feasible(n_in=0, n_out=0, gap_approx=False):
        # if gap_approx:
        #     n_in += 1
        #     n_out += 1

        rand_vals = []
        for i in range(n_in + n_out):
            rand_vals.append(random.randint(1, 100))

        total_sum = np.sum(np.array(rand_vals))

        perc = []
        for i in range(n_in + n_out):
            perc.append(rand_vals[i] / total_sum)

        assert math.isclose(
            np.sum(np.array(perc)), 1.0
        ), "Sum of mem_in_1_perc, mem_in_2_perc and mem_out_perc should be 1"

        perc_in = perc[:n_in]
        perc_out = perc[n_in:]

        return perc_in, perc_out

    @staticmethod
    def fix_inconsistent_config(config, graph):
        new_config = config.copy()
        for node in config.keys():
            if not graph.has_node(node):
                del new_config[node]
        for node in graph.nodes():
            if not node in config.keys():
                op_type = graph.nodes[node]["type"]
                hw = graph.nodes[node]["hw"]
                if isinstance(hw, GAP3DLayer):
                    channels = hw.channels
                    coarse_inout_feasible = utils.get_factors(channels)
                    coarse_inout_factor = (
                        random.choice(coarse_inout_feasible) / channels
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
                elif isinstance(hw, Convolutional3DLayer):
                    channels = hw.channels
                    filters = hw.filters
                    kernel_size = hw.kernel_shape
                    coarse_in_feasible = utils.get_factors(channels)
                    coarse_out_feasible = utils.get_factors(filters)
                    fine_feasible = utils.get_fine_feasible(kernel_size)
                    coarse_in_factor = random.choice(coarse_in_feasible) / channels
                    coarse_out_factor = random.choice(coarse_out_feasible) / filters
                    fine_factor = random.choice(fine_feasible) / np.prod(
                        np.array(kernel_size)
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "fine": fine_factor,
                        "coarse_in": coarse_in_factor,
                        "coarse_out": coarse_out_factor,
                    }
                elif isinstance(hw, Pooling3DLayer):
                    channels = hw.channels
                    kernel_size = hw.kernel_shape
                    coarse_inout_feasible = utils.get_factors(channels)
                    fine_feasible = utils.get_fine_feasible(kernel_size)
                    coarse_inout_factor = random.choice(coarse_inout_feasible) / channels
                    fine_factor = random.choice(fine_feasible) / np.prod(
                        np.array(kernel_size)
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "fine": fine_factor,
                        "coarse_inout": coarse_inout_factor,
                    }
                elif isinstance(hw, Activation3DLayer):
                    channels = hw.channels
                    coarse_inout_feasible = utils.get_factors(channels)
                    coarse_inout_factor = (
                        random.choice(coarse_inout_feasible) / channels
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
                elif isinstance(hw, ElementWise3DLayer):
                    channels = hw.channels_1
                    coarse_inout_feasible = utils.get_factors(channels)
                    coarse_inout_factor = (
                        random.choice(coarse_inout_feasible) / channels
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
                elif isinstance(hw, BatchNorm3DLayer):
                    channels = hw.channels
                    coarse_inout_feasible = utils.get_factors(channels)
                    coarse_inout_factor = (
                        random.choice(coarse_inout_feasible) / channels
                    )
                    new_config[node] = {
                        "op_type": op_type,
                        "coarse_inout": coarse_inout_factor,
                    }
                elif isinstance(hw, SqueezeExcitationLayer):
                    assert False, "Not supported layer (SqueezeExcitationLayer)"
                elif isinstance(hw, FCLayer):
                    dim_in = hw.dim_in
                    dim_out = hw.dim_out
                    coarse_in_feasible = utils.get_factors(dim_in)
                    coarse_out_feasible = utils.get_factors(dim_out)
                    coarse_in_factor = random.choice(coarse_in_feasible) / dim_in
                    coarse_out_factor = random.choice(coarse_out_feasible) / dim_out
                    new_config[node] = {
                        "op_type": op_type,
                        "coarse_in": coarse_in_factor,
                        "coarse_out": coarse_out_factor,
                    }
                elif isinstance(hw, MemoryNode):
                    continue
                else:
                    assert False, "Not supported layer"
        return new_config
