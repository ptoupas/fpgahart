import copy
import json
import math
import os
import random
import time
from copy import deepcopy

import numpy as np

import wandb
from fpga_hart import _logger
from fpga_hart.layers.activation_3d import Activation3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise_3d import ElementWise3DLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap_3d import GAP3DLayer
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.utils import utils
from fpga_hart.utils.shapes import get_random_arbitrary_shape, get_random_shape
from fpga_hart.utils.graph_manipulation import get_nodes_sorted

def run_optimizer_latency(self, alignedfactors: bool) -> None:

    bblocks, lookuptable = self.generate_building_blocks()
    bblocks_config = self.generate_building_blocks_config(
        bblocks, alignedfactors, lookuptable, initialization=True
    )

    cost, scheduling, dsp_util, bram_util, bw_util = self.get_cost_e2e(
        bblocks_config, lookuptable
    )

    if cost is None:
        for _ in range(100):
            bblocks, lookuptable = self.generate_building_blocks()
            bblocks_config = self.generate_building_blocks_config(
                bblocks, alignedfactors, lookuptable, initialization=True
            )
            cost, scheduling, dsp_util, bram_util, bw_util = self.get_cost_e2e(
                bblocks_config, lookuptable
            )
            if cost is not None:
                break
        if cost is None:
            _logger.error("No configuration found in 100 iterations. Exiting...")
            return None

    prev_state = bblocks_config
    prev_scheduling = scheduling
    prev_cost = cost
    prev_dsp = dsp_util
    prev_bram = bram_util
    prev_bw = bw_util

    current_temp = self.t_max
    print(f"Temperature  |  Latency    ")
    while current_temp > self.t_min:
        if self.block_gen == 'pre_while':
            bblocks, lookuptable = self.generate_building_blocks()

        if self.enable_wandb:
            log_dict = {}
            log_dict["temperature"] = current_temp
            log_dict["latency"] = prev_cost
            log_dict["dsp_util"] = prev_dsp
            log_dict["bram_util"] = prev_bram
            log_dict["mem_bw_util"] = prev_bw
            log_dict["num_blocks"] = len(prev_state)
            wandb.log(log_dict)

        num_iterations = 0
        timeout_tmr_start = time.time()
        while num_iterations < self.iterationPerTemp and time.time() - timeout_tmr_start < 10.0:
        # for _ in range(self.iterationPerTemp):
            if self.block_gen == 'post_while':
                bblocks, lookuptable = self.generate_building_blocks()

            if self.use_previous_config:
                new_state = self.generate_building_blocks_config(
                    bblocks, alignedfactors, lookuptable, previous_config=prev_state
                )
            else:
                new_state = self.generate_building_blocks_config(
                    bblocks, alignedfactors, lookuptable, previous_config=None
                )

            if new_state is None:
                continue
            num_iterations += 1

            (
                new_cost,
                new_scheduling,
                dsp_util,
                bram_util,
                bw_util,
            ) = self.get_cost_e2e(new_state, lookuptable)


            cost_diff = prev_cost - new_cost
            if cost_diff >= 0:
                prev_state = copy.deepcopy(new_state)
                prev_cost = copy.deepcopy(new_cost)
                prev_dsp = copy.deepcopy(dsp_util)
                prev_bram = copy.deepcopy(bram_util)
                prev_bw = copy.deepcopy(bw_util)
                prev_scheduling = copy.deepcopy(new_scheduling)
            else:
                if random.uniform(0, 1) < math.exp(cost_diff / current_temp):
                    prev_state = copy.deepcopy(new_state)
                    prev_cost = copy.deepcopy(new_cost)
                    prev_dsp = copy.deepcopy(dsp_util)
                    prev_bram = copy.deepcopy(bram_util)
                    prev_bw = copy.deepcopy(bw_util)
                    prev_scheduling = copy.deepcopy(new_scheduling)

        current_temp *= self.cooling_rate
        print(
            f"{current_temp:.5e}\t{prev_cost:.5e}",
            end="\r",
        )

    print(f"{current_temp:.5e}\t{prev_cost:.5e}\n")
    final_config = deepcopy(prev_state)
    final_DSP_util = 0
    final_BRAM_util = 0
    final_avg_MemBw_util = 0
    for key in final_config:
        final_config[key].pop("hw")
        final_DSP_util += final_config[key]["DSP_util"]
        final_BRAM_util += final_config[key]["BRAM_util"]
        final_avg_MemBw_util += final_config[key]["MemBw_util"]

    final_avg_MemBw_util = final_avg_MemBw_util / len(final_config)
    print(
        f"DSP Utilization: {final_DSP_util:.3f} - BRAM Utilization: {final_BRAM_util:.3f} - MemBw Utilization: {final_avg_MemBw_util:.3f}"
    )
    if self.enable_wandb:
        artifact = wandb.Artifact("config", type="json")
        with artifact.new_file("config.json") as f:
            json.dump(final_config, f, indent=2)
        wandb.log_artifact(artifact)
        artifact = wandb.Artifact("scheduling", type="json")
        with artifact.new_file("scheduling.json") as f:
            json.dump(prev_scheduling, f, indent=2)
        wandb.log_artifact(artifact)
    else:
        if not os.path.exists(
            "fpga_modeling_reports/" + self.cnn_model_name + "/latency_driven"
        ):
            os.makedirs(
                "fpga_modeling_reports/" + self.cnn_model_name + "/latency_driven"
            )
        with open(
            "fpga_modeling_reports/" + self.cnn_model_name + "/latency_driven/config.json",
            "w",
        ) as f:
            json.dump(final_config, f, indent=2)
        with open(
            "fpga_modeling_reports/" + self.cnn_model_name + "/latency_driven/scheduling.json",
            "w",
        ) as f:
            json.dump(prev_scheduling, f, indent=2)

def validate_building_blocks_setup(self, bblocks: list) -> bool:
    """
    Validate the building blocks setup by producing a valid scedulilng of the building blocks that
    can execute the complete network graph.
    """

    nodes = get_nodes_sorted(self.graph)
    _logger.info(msg=f"Validating building blocks setup... {bblocks}")
    for n in nodes:
        bb = self.graph.nodes[n]["hw_type"]
        if any(bb in block for block in bblocks):
            continue
        else:
            _logger.critical(
                msg=f"Hardware building block {bb} not found in existing building blocks list"
            )
            return False

    return True

def generate_building_blocks(self) -> list:
    """
    Generate a set of hardware building blocks that can be used to execute the graph's workload.

    Returns:
        dict: bb_setup
    """
    types = {}
    for n in self.graph.nodes:
        if self.graph.nodes[n]["hw_type"] not in types:
            if "Conv" in self.graph.nodes[n]["hw_type"] or "Pooling" in self.graph.nodes[n]["hw_type"]:
                types[self.graph.nodes[n]["hw_type"]] = {
                    "Padding": [self.graph.nodes[n]["hw"].padding],
                    "Stride": [self.graph.nodes[n]["hw"].stride],
                }
            else:
                types[self.graph.nodes[n]["hw_type"]] = {}
        else:
            if "Conv" in self.graph.nodes[n]["hw_type"] or "Pooling" in self.graph.nodes[n]["hw_type"]:
                types[self.graph.nodes[n]["hw_type"]]["Padding"].append(
                    self.graph.nodes[n]["hw"].padding
                )
                types[self.graph.nodes[n]["hw_type"]]["Stride"].append(
                    self.graph.nodes[n]["hw"].stride
                )

    bblocks = []
    for t in types:
        if "Conv" in t or "Pooling" in t:
            final_padding = np.max(np.array(types[t]["Padding"]), axis=0).tolist()
            final_stride = np.min(np.array(types[t]["Stride"]), axis=0).tolist()
            bblocks.append(
                f"{t}p{''.join([str(elem) for elem in final_padding])}s{''.join([str(elem) for elem in final_stride])}"
            )
        else:
            bblocks.append(t)

    bblocks, lookuptable = utils.combine_building_blocks(bblocks)

    # assert self.validate_building_blocks_setup(
    #     bblocks
    # ), "Invalid building blocks setup. Cannot find a valid scheduling."

    return bblocks, lookuptable

def generate_building_blocks_config(
    self,
    bblocks: list,
    alignedfactors: bool,
    lookuptable: dict,
    previous_config: dict = None,
    initialization: bool = False,
) -> dict:
    """
    Generate a configuration for each building block based on the min and max channels and filters values
    from all its instances across the graph of the network.

    Returns:
        dict: bb_setup
    """
    if "Activation" in bblocks:
        activations_list = []
        for n in self.graph.nodes:
            if self.graph.nodes[n]["hw_type"] == "Activation":
                if self.graph.nodes[n]['hw'].op_type not in activations_list:
                    activations_list.append(self.graph.nodes[n]['hw'].op_type)
    if "ElementWise" in bblocks:
        elementwise_list = []
        for n in self.graph.nodes:
            if self.graph.nodes[n]["hw_type"] == "ElementWise":
                if self.graph.nodes[n]['hw'].op_type not in elementwise_list:
                    elementwise_list.append(self.graph.nodes[n]['hw'].op_type)

    total_dsp = 0
    total_bram = 0
    if (not initialization) and (not previous_config == None) and (bblocks == list(previous_config.keys())):
        bb_setup = deepcopy(previous_config)
        bb_choice = [bb for bb in bblocks]
        bblocks = random.choices(bb_choice, k=int(len(bb_choice)*self.bblock_keep_percentage))
        for b in bb_setup:
            total_dsp += bb_setup[b]["DSP_util"]
            total_bram += bb_setup[b]["BRAM_util"]
    else:
        bb_setup = dict()

    for bb in random.shuffle(bblocks):
        if not bb in bb_setup.keys():
            bb_setup[bb] = dict()

        dsp_util, bram_util = 100, 100
        stop_counter = 0
        while dsp_util > (self.config.max_dsp_util - total_dsp) or bram_util > (
            self.config.max_bram_util - total_bram
        ):
            if self.use_arbitrary_shape:
                shape_in, shape_out = get_random_arbitrary_shape(
                    self.graph, bb, lookuptable, previous_config=previous_config, chan_dist_thresh=self.chan_dist_thresh, depth_dist_thresh=self.depth_dist_thresh, height_dist_thresh=self.height_dist_thresh
                )
            else:
                shape_in, shape_out = get_random_shape(
                    self.graph, bb, lookuptable, previous_config=previous_config, chan_dist_thresh=self.chan_dist_thresh, depth_dist_thresh=self.depth_dist_thresh,height_dist_thresh=self.height_dist_thresh
                )
            if bb != "Gemm":
                _, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim = shape_in
                (
                    _,
                    channels_out_dim,
                    depth_out_dim,
                    height_out_dim,
                    width_out_dim,
                ) = shape_out
            else:
                channels_in_dim, channels_out_dim = shape_in[1], shape_out[1]
                depth_in_dim, height_in_dim, width_in_dim = 1, 1, 1
                depth_out_dim, height_out_dim, width_out_dim = 1, 1, 1

            bb_descriptor = utils.generate_description_from_type(
                bb,
                channels_in_dim,
                depth_in_dim,
                height_in_dim,
                width_in_dim,
                channels_out_dim,
                depth_out_dim,
                height_out_dim,
                width_out_dim,
            )

            if "Conv" in bb:
                bb_setup[bb]["hw"] = Convolutional3DLayer(
                    self.config.max_dsp_util, self.config.max_bram_util, bb_descriptor
                )
            if "Pooling" in bb:
                bb_setup[bb]["hw"] = Pooling3DLayer(
                    self.config.max_dsp_util, self.config.max_bram_util, bb_descriptor
                )
            elif bb == "Activation":
                bb_setup[bb]["hw"] = Activation3DLayer(
                    self.config.max_dsp_util, self.config.max_bram_util, bb_descriptor
                )
            elif bb == "GlobalAveragePool":
                bb_setup[bb]["hw"] = GAP3DLayer(
                    self.config.max_dsp_util, self.config.max_bram_util, bb_descriptor
                )
            elif bb == "ElementWise":
                bb_setup[bb]["hw"] = ElementWise3DLayer(
                    self.config.max_dsp_util, self.config.max_bram_util, bb_descriptor
                )
            elif bb == "Gemm":
                bb_setup[bb]["hw"] = FCLayer(
                    self.config.max_dsp_util, self.config.max_bram_util, bb_descriptor
                )

            if "Conv" in bb:
                if alignedfactors:
                    coarse_in = (
                        random.choice(utils.get_factors(channels_in_dim))
                        / channels_in_dim
                    )
                    coarse_out = (
                        random.choice(utils.get_factors(channels_out_dim))
                        / channels_out_dim
                    )
                else:
                    coarse_in = (
                        np.random.choice(
                            np.arange(channels_in_dim) + 1, replace=False
                        )
                        / channels_in_dim
                    )
                    coarse_out = (
                        np.random.choice(
                            np.arange(channels_out_dim) + 1, replace=False
                        )
                        / channels_out_dim
                    )
                if "Conv3DDw" in bb:
                    coarse_out = coarse_in
                assert coarse_in > 0 and coarse_in <= 1, "Invalid coarse in."
                assert coarse_out > 0 and coarse_out <= 1, "Invalid coarse out."
                # TODO: Add fine factor random generation for Conv3D ops
                dsp_util, bram_util, _ = bb_setup[bb]["hw"].get_resource_util(
                    f_fine=1, f_coarseIn=coarse_in, f_coarseOut=coarse_out
                )
            elif "Pooling" in bb:
                if alignedfactors:
                    coarse_inout = (
                        random.choice(utils.get_factors(channels_in_dim))
                        / channels_in_dim
                    )
                else:
                    coarse_inout = (
                        np.random.choice(
                            np.arange(channels_in_dim) + 1, replace=False
                        )
                        / channels_in_dim
                    )
                assert coarse_inout > 0 and coarse_inout <= 1, "Invalid coarse factor."
                # TODO: Add fine factor random generation for Pooling3D ops
                dsp_util, bram_util, _ = bb_setup[bb]["hw"].get_resource_util(
                    f_fine=1, f_coarse_inout=coarse_inout
                )
            elif bb in ["Activation", "GlobalAveragePool", "ElementWise"]:
                if alignedfactors:
                    coarse_inout = (
                        random.choice(utils.get_factors(channels_in_dim))
                        / channels_in_dim
                    )
                else:
                    coarse_inout = (
                        np.random.choice(
                            np.arange(channels_in_dim) + 1, replace=False
                        )
                        / channels_in_dim
                    )
                assert coarse_inout > 0 and coarse_inout <= 1, "Invalid coarse factor."

                if bb == "GlobalAveragePool":
                    dsp_util, bram_util, _ = bb_setup[bb]["hw"].get_resource_util(
                        f_coarse_inout=coarse_inout, supported_ops=[], gap_approx=self.gap_approx
                    )
                else:
                    if bb == "Activation":
                        supported_ops = deepcopy(activations_list)
                    elif bb == "ElementWise":
                        supported_ops = deepcopy(elementwise_list)
                    dsp_util, bram_util, _ = bb_setup[bb]["hw"].get_resource_util(
                        f_coarse_inout=coarse_inout, supported_ops=supported_ops
                    )
            elif bb == "Gemm":
                if alignedfactors:
                    coarse_in = (
                        random.choice(utils.get_factors(channels_in_dim))
                        / channels_in_dim
                    )
                    coarse_out = (
                        random.choice(utils.get_factors(channels_out_dim))
                        / channels_out_dim
                    )
                else:
                    coarse_in = (
                        np.random.choice(
                            np.arange(channels_in_dim) + 1, replace=False
                        )
                        / channels_in_dim
                    )
                    coarse_out = (
                        np.random.choice(
                            np.arange(channels_out_dim) + 1, replace=False
                        )
                        / channels_out_dim
                    )
                assert coarse_in > 0 and coarse_in <= 1, "Invalid coarse in."
                assert coarse_out > 0 and coarse_out <= 1, "Invalid coarse out."
                dsp_util, bram_util, _ = bb_setup[bb]["hw"].get_resource_util(
                    f_coarseIn=coarse_in, f_coarseOut=coarse_out
                )
            stop_counter += 1
            if stop_counter > 100:
                _logger.debug(
                    "Could not find a valid configuration for the current building block setup. Returning without result."
                )
                return None

        mem_bw = bb_setup[bb]["hw"].mem_words_per_cycle
        if "ElementWise" in bb:
            bw_in, bw_out = self.get_mem_bw_feasible(n_in=2, n_out=1)
            bw_in_1 = bw_in[0] * mem_bw
            bw_in_2 = bw_in[1] * mem_bw
            bw_out = bw_out[0] * mem_bw
            bb_setup[bb]["bw_in"] = [bw_in_1, bw_in_2]
            bb_setup[bb]["bw_out"] = [bw_out]
        else:
            bw_in, bw_out = self.get_mem_bw_feasible(n_in=1, n_out=1)
            bw_in = bw_in[0] * mem_bw
            bw_out = bw_out[0] * mem_bw
            bb_setup[bb]["bw_in"] = [bw_in]
            bb_setup[bb]["bw_out"] = [bw_out]
        if "Conv" in bb:
            layer_config = utils.generate_layer_config(bb_setup[bb]["hw"], [1, coarse_in, coarse_out])
            bb_setup[bb]["config"] = layer_config
            bb_setup[bb]["f_coarseIn"] = coarse_in
            bb_setup[bb]["interleaving_in"] = math.ceil(1 / coarse_in)
            bb_setup[bb]["f_coarseOut"] = coarse_out
            bb_setup[bb]["interleaving_out"] = math.ceil(1 / coarse_out)
        elif "Pooling" in bb:
            layer_config = utils.generate_layer_config(bb_setup[bb]["hw"], [1, coarse_inout])
            bb_setup[bb]["config"] = layer_config
            bb_setup[bb]["coarse_inout"] = coarse_inout
            bb_setup[bb]["coarse_factor"] = math.ceil(coarse_inout * channels_in_dim)
            bb_setup[bb]["interleaving_inout"] = math.ceil(1 / coarse_inout)
        elif bb in ["Activation", "GlobalAveragePool", "ElementWise"]:
            layer_config = utils.generate_layer_config(bb_setup[bb]["hw"], [coarse_inout])
            if bb == "Activation":
                layer_config['supported_ops'] = deepcopy(activations_list)
            elif bb == "ElementWise":
                layer_config['supported_ops'] = deepcopy(elementwise_list)
            bb_setup[bb]["config"] = layer_config
            bb_setup[bb]["coarse_inout"] = coarse_inout
            bb_setup[bb]["coarse_factor"] = math.ceil(coarse_inout * channels_in_dim)
            bb_setup[bb]["interleaving_inout"] = math.ceil(1 / coarse_inout)
        elif bb == "Gemm":
            layer_config = utils.generate_layer_config(bb_setup[bb]["hw"], [coarse_in, coarse_out])
            bb_setup[bb]["config"] = layer_config
            bb_setup[bb]["coarse_in"] = coarse_in
            bb_setup[bb]["interleaving_in"] = math.ceil(1 / coarse_in)
            bb_setup[bb]["coarse_out"] = coarse_out
            bb_setup[bb]["interleaving_out"] = math.ceil(1 / coarse_out)

        bb_setup[bb]["DSP_util"] = dsp_util
        bb_setup[bb]["BRAM_util"] = bram_util
        total_dsp += dsp_util
        total_bram += bram_util

    return bb_setup

def get_cost_e2e(self, bblocks_config: dict, lookuptable: dict) -> float:
    if bblocks_config is None:
        return None, None, None, None, None
    cost = 0.0
    avg_BW = 0.0
    scheduling = {}
    for node in self.graph.nodes:
        bb_type = lookuptable[self.graph.nodes[node]["hw_type"]]
        hw = self.graph.nodes[node]["hw"]

        if len(hw.input_shape) < 5:
            depth_calls = 1
            height_calls = 1
            width_calls = 1
        else:
            # TODO: Search how to deal with cases where the output dimensions are also altered? Like in convolutional layers with stride > 1.
            depth_calls = math.ceil(
                hw.input_shape[2] / bblocks_config[bb_type]["config"]["depth_in"]
            )
            height_calls = math.ceil(
                hw.input_shape[3] / bblocks_config[bb_type]["config"]["height_in"]
            )
            width_calls = math.ceil(
                hw.input_shape[4] / bblocks_config[bb_type]["config"]["width_in"]
            )

        if "Conv" in bb_type or "Gemm" in bb_type:
            in_calls = math.ceil(
                hw.input_shape[1]
                / (
                    bblocks_config[bb_type]["config"]["coarse_in_factor"]
                    * bblocks_config[bb_type]["interleaving_in"]
                )
            )
            out_calls = math.ceil(
                hw.output_shape[1]
                / (
                    bblocks_config[bb_type]["config"]["coarse_out_factor"]
                    * bblocks_config[bb_type]["interleaving_out"]
                )
            )
            if bb_type == "Gemm":
                total_block_calls = (
                    in_calls * out_calls * depth_calls * height_calls * width_calls
                )
            else:
                total_block_calls = (
                    in_calls * out_calls * depth_calls * height_calls * width_calls
                )
        else:
            inout_calls = math.ceil(
                hw.input_shape[1]
                / (
                    bblocks_config[bb_type]["coarse_factor"]
                    * bblocks_config[bb_type]["interleaving_inout"]
                )
            )
            total_block_calls = (
                inout_calls * depth_calls * height_calls * width_calls
            )

        # TODO: Update the shapes of the bblocks_config[bb_type]["hw"] to account for the overlapping regions because of feature map tilling
        if "Conv" in bb_type:
            bblock_padding = deepcopy(bblocks_config[bb_type]["hw"].padding)
            bblock_stride = deepcopy(bblocks_config[bb_type]["hw"].stride)

            bblocks_config[bb_type]["hw"].padding = hw.padding
            bblocks_config[bb_type]["hw"].stride = hw.stride
            performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                f_fine=1,
                f_coarseIn=bblocks_config[bb_type]["f_coarseIn"],
                f_coarseOut=bblocks_config[bb_type]["f_coarseOut"],
                mem_bw_in=bblocks_config[bb_type]["bw_in"][0],
                mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
            )
            bblocks_config[bb_type]["hw"].padding = bblock_padding
            bblocks_config[bb_type]["hw"].stride = bblock_stride
        elif "Pooling" in bb_type:
            bblock_padding = deepcopy(bblocks_config[bb_type]["hw"].padding)
            bblock_stride = deepcopy(bblocks_config[bb_type]["hw"].stride)

            bblocks_config[bb_type]["hw"].padding = hw.padding
            bblocks_config[bb_type]["hw"].stride = hw.stride
            performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                f_fine=1,
                f_coarse_inout=bblocks_config[bb_type]["coarse_inout"],
                mem_bw_in=bblocks_config[bb_type]["bw_in"][0],
                mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
            )
            bblocks_config[bb_type]["hw"].padding = bblock_padding
            bblocks_config[bb_type]["hw"].stride = bblock_stride
        elif bb_type == "Activation":
            bblocks_config[bb_type]["hw"].op_type = hw.op_type
            performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                coarse_inout=bblocks_config[bb_type]["coarse_inout"],
                mem_bw_in=bblocks_config[bb_type]["bw_in"][0],
                mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
            )
            bblocks_config[bb_type]["hw"].op_type = "Activation"
        elif bb_type == "GlobalAveragePool":
            performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                coarse_inout=bblocks_config[bb_type]["coarse_inout"],
                mem_bw_in=bblocks_config[bb_type]["bw_in"][0],
                mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
            )
        elif bb_type == "ElementWise":
            bblocks_config[bb_type]["hw"].op_type = hw.op_type
            bblock_input_shape_2 = deepcopy(
                bblocks_config[bb_type]["hw"].input_shape_2
            )

            layer_input_shape_2 = hw.input_shape_2
            if np.prod(layer_input_shape_2[2:]) == 1:
                bblocks_config[bb_type]["hw"].input_shape_2[
                    2:
                ] = layer_input_shape_2[2:]

            performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                coarse_inout=bblocks_config[bb_type]["coarse_inout"],
                mem_bw_in_1=bblocks_config[bb_type]["bw_in"][0],
                mem_bw_in_2=bblocks_config[bb_type]["bw_in"][1],
                mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
            )
            bblocks_config[bb_type]["hw"].op_type = "ElementWise"
            bblocks_config[bb_type]["hw"].input_shape_2 = bblock_input_shape_2
        elif bb_type == "Gemm":
            performance_modeling = bblocks_config[bb_type]["hw"].get_design_point(
                coarse_in=bblocks_config[bb_type]["coarse_in"],
                coarse_out=bblocks_config[bb_type]["coarse_out"],
                mem_bw_in=bblocks_config[bb_type]["bw_in"][0],
                mem_bw_out=bblocks_config[bb_type]["bw_out"][0],
            )

        # TODO: Revert back the shapes of the bblocks_config[bb_type]["hw"]

        bblocks_config[bb_type]["MemBw_util"] = float(performance_modeling["memBwUtil"])

        latency = performance_modeling["latency(S)"] * math.ceil(total_block_calls)
        avg_BW += performance_modeling["memBwUtil"]
        assert math.ceil(total_block_calls) > 0, "Zero calls aborting..."
        scheduling[node] = {
            "Block Type": bb_type,
            "Shape In": hw.input_shape,
            "Shape In 2": hw.input_shape_red if bb_type == "ElementWise" else [],
            "Shape Out": hw.output_shape,
            "Tiling Channels": in_calls
            if "Conv" in bb_type or "Gemm" in bb_type
            else inout_calls,
            "Tiling Filters": out_calls
            if "Conv" in bb_type or "Gemm" in bb_type
            else 1,
            "Tiling Depth": depth_calls,
            "Tiling Height": height_calls,
            "Tiling Width": width_calls,
            "Latency": latency,
            "Base Latency": performance_modeling["latency(S)"],
            "Read From": list(self.graph.predecessors(node)),
            "Write To": list(self.graph.successors(node)),
            "Store To Mem": (True if self.graph.out_degree(node) > 1 else False),
            "Load From Mem": (True if self.graph.in_degree(node) > 1 else False),
            "Kernel Shape": hw.kernel_shape if "Conv" in bb_type else [],
            "Stride": hw.stride if "Conv" in bb_type else [],
            "Padding": hw.padding if "Conv" in bb_type else [],
            "Broadcast": hw.broadcasting if bb_type == "ElementWise" else False,
            "Op Type": hw.op_type
            if bb_type in ["Activation", "ElementWise"]
            else "None",
        }
        cost += latency

    avg_BW = avg_BW / len(self.graph.nodes)
    final_DSP = 0
    final_BRAM = 0
    for bb in bblocks_config:
        final_DSP += bblocks_config[bb]["DSP_util"]
        final_BRAM += bblocks_config[bb]["BRAM_util"]
    return cost, scheduling, final_DSP, final_BRAM, avg_BW