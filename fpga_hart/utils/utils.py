import csv
import itertools
import json
import math
import os
import random
from copy import deepcopy
from ctypes import c_int
from functools import reduce
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from fpga_hart import _logger
from fpga_hart.layers.activation import ActivationLayer
from fpga_hart.layers.batchnorm_3d import BatchNorm3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise import ElementWiseLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap import GAPLayer
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.layers.squeeze_excitation import SqueezeExcitationLayer

sns.set(rc={"figure.figsize": (15, 8)})
sns.set_style("whitegrid")


def get_factors(n, max_parallel=None, keep_percentage=None) -> list:
    """
    Parameters
    ----------
    n: int

    Returns
    -------
    list
        list of integers that are factors of `n`
    """
    if max_parallel is None:
        result = list(
            set(
                reduce(
                    list.__add__,
                    ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
                )
            )
        )
        if not keep_percentage == None:
            keep_perc = 1 - keep_percentage
            threshold = max(math.ceil(max(result) * keep_perc), min(result))
            return sorted([x for x in result if x <= threshold])
        else:
            return sorted(result)
    else:
        factors = np.arange(n) + 1
        return (factors[factors <= max_parallel]).tolist()


def combine_building_blocks(building_blocks: list) -> Tuple[list, dict]:
    conv_blocks_lookup_choices = []
    conv_blocks_dw_lookup_choices = []
    pool_blocks_lookup_choices = []
    conv_blocks_choices = []
    conv_dw_blocks_choices = []
    pool_blocks_choices = []

    conv_blocks = []
    conv_dw_blocks = []
    pool_blocks = []
    rest_operations = []
    for bb in building_blocks:
        if "Conv" in bb:
            if "Dw" in bb:
                conv_dw_blocks.append(bb)
            else:
                conv_blocks.append(bb)
        elif "Pooling" in bb:
            pool_blocks.append(bb)
        else:
            rest_operations.append(bb)

    # Combine convolutional blocks
    kernel_list = []
    padding_list = []
    stride_list = []
    lookuptable = {}
    for c in conv_blocks:
        lookuptable[c.split("p")[0]] = c
        kernel_shape = [int(x) for x in c.split("k")[-1][:3]]
        kernel_list.append(kernel_shape)
        padding = [int(x) for x in c.split("p")[-1][:3]]
        padding_list.append(padding)
        stride = [int(x) for x in c.split("s")[-1][:3]]
        stride_list.append(stride)

    if kernel_list:
        conv_blocks_choices.append(conv_blocks)
        conv_blocks_lookup_choices.append(deepcopy(lookuptable))

        combined_kernel = np.max(np.array(kernel_list), axis=0).tolist()
        combined_padding = np.max(np.array(padding_list), axis=0).tolist()
        combined_stride = np.min(np.array(stride_list), axis=0).tolist()

        try:
            index = kernel_list.index(combined_kernel)
        except ValueError:
            index = -1

        if index != -1:
            block = f"{conv_blocks[index].split('p')[0]}p{''.join([str(elem) for elem in combined_padding])}s{''.join([str(elem) for elem in combined_stride])}"
        else:
            block = f"Conv3Dk{''.join([str(elem) for elem in combined_kernel])}p{''.join([str(elem) for elem in combined_padding])}s{''.join([str(elem) for elem in combined_stride])}"
        lookuptable.clear()
        for c in conv_blocks:
            lookuptable[c.split("p")[0]] = block
        conv_blocks_choices.append([block])
        conv_blocks_lookup_choices.append(deepcopy(lookuptable))

    # Combine pooling blocks
    kernel_list = []
    padding_list = []
    stride_list = []
    lookuptable = {}
    for c in pool_blocks:
        lookuptable[c.split("p")[0]] = c
        kernel_shape = [int(x) for x in c.split("k")[-1][:3]]
        kernel_list.append(kernel_shape)
        padding = [int(x) for x in c.split("p")[-1][:3]]
        padding_list.append(padding)
        stride = [int(x) for x in c.split("s")[-1][:3]]
        stride_list.append(stride)

    if kernel_list:
        pool_blocks_choices.append(pool_blocks)
        pool_blocks_lookup_choices.append(deepcopy(lookuptable))

        combined_kernel = np.max(np.array(kernel_list), axis=0).tolist()
        combined_padding = np.max(np.array(padding_list), axis=0).tolist()
        combined_stride = np.min(np.array(stride_list), axis=0).tolist()

        try:
            index = kernel_list.index(combined_kernel)
        except ValueError:
            index = -1

        if index != -1:
            block = f"{pool_blocks[index].split('p')[0]}p{''.join([str(elem) for elem in combined_padding])}s{''.join([str(elem) for elem in combined_stride])}"
        else:
            block = f"Poolingk{''.join([str(elem) for elem in combined_kernel])}p{''.join([str(elem) for elem in combined_padding])}s{''.join([str(elem) for elem in combined_stride])}"
        lookuptable.clear()
        for c in pool_blocks:
            lookuptable[c.split("p")[0]] = block
        pool_blocks_choices.append([block])
        pool_blocks_lookup_choices.append(deepcopy(lookuptable))

    # Combine convolutional depthwise blocks
    kernel_list = []
    padding_list = []
    stride_list = []
    lookuptable = {}
    for c in conv_dw_blocks:
        lookuptable[c.split("p")[0]] = c
        kernel_shape = [int(x) for x in c.split("k")[-1][:3]]
        kernel_list.append(kernel_shape)
        padding = [int(x) for x in c.split("p")[-1][:3]]
        padding_list.append(padding)
        stride = [int(x) for x in c.split("s")[-1][:3]]
        stride_list.append(stride)

    if kernel_list:
        conv_dw_blocks_choices.append(conv_dw_blocks)
        conv_blocks_dw_lookup_choices.append(deepcopy(lookuptable))
        combined_kernel = np.max(np.array(kernel_list), axis=0).tolist()
        combined_padding = np.max(np.array(padding_list), axis=0).tolist()
        combined_stride = np.min(np.array(stride_list), axis=0).tolist()

        try:
            index = kernel_list.index(combined_kernel)
        except ValueError:
            index = -1

        if index != -1:
            block = f"{conv_dw_blocks[index].split('p')[0]}p{''.join([str(elem) for elem in combined_padding])}s{''.join([str(elem) for elem in combined_stride])}"
        else:
            block = f"Conv3DDwk{''.join([str(elem) for elem in combined_kernel])}p{''.join([str(elem) for elem in combined_padding])}s{''.join([str(elem) for elem in combined_stride])}"
        lookuptable.clear()
        for c in conv_dw_blocks:
            lookuptable[c.split("p")[0]] = block
        conv_blocks_dw_lookup_choices.append(deepcopy(lookuptable))
        conv_dw_blocks_choices.append([block])

    assert len(conv_blocks_choices) == len(conv_blocks_lookup_choices)
    assert len(pool_blocks_choices) == len(pool_blocks_lookup_choices)
    assert len(conv_dw_blocks_choices) == len(conv_blocks_dw_lookup_choices)

    conv_blocks_idx = (
        random.randint(0, len(conv_blocks_choices) - 1) if conv_blocks_choices else -1
    )
    pool_blocks_idx = (
        random.randint(0, len(pool_blocks_choices) - 1) if pool_blocks_choices else -1
    )
    conv_dw_blocks_idx = (
        random.randint(0, len(conv_dw_blocks_choices) - 1)
        if conv_dw_blocks_choices
        else -1
    )
    final_building_blocks = deepcopy(rest_operations)
    if conv_blocks_idx != -1:
        final_building_blocks += conv_blocks_choices[conv_blocks_idx]
    if pool_blocks_idx != -1:
        final_building_blocks += pool_blocks_choices[pool_blocks_idx]
    if conv_dw_blocks_idx != -1:
        final_building_blocks += conv_dw_blocks_choices[conv_dw_blocks_idx]

    final_lookup_table = {}
    for op in rest_operations:
        final_lookup_table[op] = op

    if conv_blocks_idx != -1:
        final_lookup_table |= conv_blocks_lookup_choices[conv_blocks_idx]
    if pool_blocks_idx != -1:
        final_lookup_table |= pool_blocks_lookup_choices[pool_blocks_idx]
    if conv_dw_blocks_idx != -1:
        final_lookup_table |= conv_blocks_dw_lookup_choices[conv_dw_blocks_idx]

    return final_building_blocks, final_lookup_table


def generate_description_from_type(
    bb: str,
    channels_in_dim: int,
    depth_in_dim: int,
    height_in_dim: int,
    width_in_dim: int,
    channels_out_dim: int,
    depth_out_dim: int,
    height_out_dim: int,
    width_out_dim: int,
):
    if "Conv" in bb:
        dw = True if "Dw" in bb else False
        pw = True if "Pw" in bb else False
        kernel_shape = [int(x) for x in bb.split("k")[-1][:3]]
        padding = [int(x) for x in bb.split("p")[-1][:3]]
        stride = [int(x) for x in bb.split("s")[-1][:3]]

        bb_descriptor = {
            "operation": "Conv",
            "shape_in": [
                [1, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim]
            ],
            "shape_out": [
                1,
                channels_out_dim,
                depth_out_dim,
                height_out_dim,
                width_out_dim,
            ],
            "kernel": [channels_out_dim, 1] + kernel_shape
            if dw
            else [channels_out_dim, channels_in_dim] + kernel_shape,
            "bias": [channels_out_dim],
            "padding": padding,
            "stride": stride,
            "groups": channels_in_dim if dw else 1,
            "dilation": [1, 1, 1],
            "branching": False,
        }
    elif "Pooling" in bb:
        kernel_shape = [int(x) for x in bb.split("k")[-1][:3]]
        padding = [int(x) for x in bb.split("p")[-1][:3]]
        stride = [int(x) for x in bb.split("s")[-1][:3]]

        bb_descriptor = {
            "operation": "Pooling",
            "shape_in": [
                [1, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim]
            ],
            "shape_out": [
                1,
                channels_out_dim,
                depth_out_dim,
                height_out_dim,
                width_out_dim,
            ],
            "kernel": kernel_shape,
            "padding": padding,
            "stride": stride,
            "branching": False,
        }
    elif bb == "Activation":
        bb_descriptor = {
            "operation": "Activation",
            "shape_in": [
                [1, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim]
            ],
            "shape_out": [
                1,
                channels_in_dim,
                depth_in_dim,
                height_in_dim,
                width_in_dim,
            ],
        }
    elif bb == "GlobalAveragePool":
        bb_descriptor = {
            "operation": "GlobalAveragePool",
            "shape_in": [
                [1, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim]
            ],
            "shape_out": [
                1,
                channels_in_dim,
                1,
                1,
                1,
            ],
        }
    elif bb == "ElementWise":
        # TODO: We assume here that we always have two inputs of the same shape. (i.e. no broadcasting)
        bb_descriptor = {
            "operation": "ElementWise",
            "shape_in": [
                [1, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim],
                [1, channels_in_dim, depth_in_dim, height_in_dim, width_in_dim],
            ],
            "shape_out": [
                1,
                channels_in_dim,
                depth_in_dim,
                height_in_dim,
                width_in_dim,
            ],
        }
    elif bb == "Gemm":
        bb_descriptor = {
            "operation": "Gemm",
            "shape_in": [[1, channels_in_dim]],
            "shape_out": [1, channels_out_dim],
            "kernel": [channels_in_dim, channels_out_dim],
            "bias": [channels_out_dim],
        }
    else:
        raise ValueError("Unknown block type: {}".format(bb))
    return bb_descriptor


def get_fine_feasible(kernel_size: list, keep_percentage: float = None):
    fine_feasible = []
    if kernel_size[0] != kernel_size[1] and kernel_size[1] == kernel_size[2]:
        if kernel_size[0] == 1:
            fine_feasible = [1, kernel_size[1], kernel_size[1] * kernel_size[2]]
        elif kernel_size[1] == 1:
            fine_feasible = [1, kernel_size[0]]
        else:
            fine_feasible = [
                1,
                kernel_size[0],
                kernel_size[1],
                kernel_size[0] * kernel_size[1],
                kernel_size[1] * kernel_size[2],
                kernel_size[0] * kernel_size[1] * kernel_size[2],
            ]
    elif kernel_size[0] == kernel_size[1] and kernel_size[1] == kernel_size[2]:
        if kernel_size[0] == 1:
            fine_feasible = [1]
        else:
            fine_feasible = [
                1,
                kernel_size[0],
                kernel_size[0] * kernel_size[1],
                kernel_size[0] * kernel_size[1] * kernel_size[2],
            ]
    else:
        fine_feasible = [
            1,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            kernel_size[0] * kernel_size[1],
            kernel_size[0] * kernel_size[2],
            kernel_size[1] * kernel_size[2],
            kernel_size[0] * kernel_size[1] * kernel_size[2],
        ]
    if not keep_percentage == None:
        keep_perc = 1 - keep_percentage
        threshold = max(math.ceil(max(fine_feasible) * keep_perc), min(fine_feasible))
        return sorted([x for x in fine_feasible if x <= threshold])
    return sorted(fine_feasible)

def find_pareto(scores, domination_type="MaxMin"):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            # print("[{}][0] = {}. [{}][0] = {}. [{}][1] = {}. [{}][1] = {}.".format(j, scores[j][0], i, scores[i][0], j, scores[j][1], i, scores[i][1]))
            if domination_type == "MaxMin":
                if (scores[j][0] >= scores[i][0] and scores[j][1] <= scores[i][1]) and (
                    scores[j][0] > scores[i][0] or scores[j][1] < scores[i][1]
                ):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break
            elif domination_type == "MinMin":
                if (scores[j][0] <= scores[i][0] and scores[j][1] <= scores[i][1]) and (
                    scores[j][0] < scores[i][0] or scores[j][1] < scores[i][1]
                ):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def plot_graph(
    throughput_ops,
    throughput_vols,
    latency,
    dsp_util,
    bram_util,
    layer_name,
    model_name,
    calculate_pareto,
    pareto_type="MaxMin",
):
    throughput = throughput_vols
    dsps_dir = os.path.join(
        os.getcwd(), "fpga_modeling_reports", "graphs", model_name, "throughput_dsps"
    )
    if not os.path.exists(dsps_dir):
        os.makedirs(dsps_dir)

    if calculate_pareto:
        scores = np.zeros((len(throughput), 2))
        scores[:, 0] = throughput
        scores[:, 1] = dsp_util

        pareto = find_pareto(scores, pareto_type)
        pareto_front = scores[pareto]

        pareto_front_df = pd.DataFrame(pareto_front)
        pareto_front_df.sort_values(0, inplace=True)
        pareto_front = pareto_front_df.values

        sns.lineplot(x=pareto_front[:, 0], y=pareto_front[:, 1], color="red")

    sns.scatterplot(x=np.array(throughput), y=np.array(dsp_util), size=bram_util)

    plt.title(layer_name)
    plt.xlabel("Throughtput(outputs/sec)")
    plt.ylabel("DSPS %")
    if max(dsp_util) > 100:
        plt.yscale("log")
    else:
        plt.ylim([-5, max(100, max(dsp_util) + 0.1 * max(dsp_util))])
    if max(throughput) > 100:
        plt.xscale("log")

    plt.legend(
        frameon=False,
        prop={"size": 8},
        loc="upper right",
        bbox_to_anchor=(1.11, 1.12),
        borderaxespad=0.0,
    )

    file_name = layer_name + ".jpg"
    plt.savefig(os.path.join(dsps_dir, file_name))
    plt.clf()


def drop_duplicates_csv(file_name):
    layers_df = pd.read_csv(file_name)

    original_size = len(layers_df.index)
    columns = layers_df.columns.tolist()
    del columns[-1]
    del columns[columns.index("Adds")]

    layers_df = layers_df.drop_duplicates(subset=columns, ignore_index=True)
    final_size = len(layers_df.index)
    print("Dropped {} rows due to duplicate".format(original_size - final_size))

    os.remove(file_name)
    layers_df.to_csv(file_name, index=False)


def plot_layers_csv(
    file_name,
    model_name,
    calculate_pareto=True,
    pareto_type="MaxMin",
    xaxis="volumes/s",
    yaxis="DSP(%)",
):
    plot_dir = os.path.join(
        os.getcwd(), "fpga_modeling_reports", "graphs", model_name, "throughput_dsps"
    )
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    layers_df = pd.read_csv(file_name)

    layers = layers_df["Layer"].unique()
    for l in layers:
        curr_df = layers_df.loc[layers_df["Layer"] == l]

        curr_df.plot.scatter(x=xaxis, y=yaxis)

        x_axis = curr_df[xaxis].to_numpy()
        y_axis = curr_df[yaxis].to_numpy()
        if calculate_pareto:
            scores = np.zeros((x_axis.shape[0], 2))
            scores[:, 0] = x_axis
            scores[:, 1] = y_axis

            pareto = find_pareto(scores, pareto_type)
            pareto_front = scores[pareto]

            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df.sort_values(0, inplace=True)
            pareto_front = pareto_front_df.values

            plt.plot(pareto_front[:, 0], pareto_front[:, 1], color="red")

        plt.ylim([-5, max(100, max(y_axis) + 0.1 * max(y_axis))])
        plt.xscale("log")
        plt.title(l)
        file_name = l + ".jpg"
        plt.savefig(os.path.join(plot_dir, file_name))
        plt.clf()


def get_nodes_sorted(graph):
    g_sorted = nx.topological_sort(graph)
    return list(g_sorted)


def generate_layer_config(layer, config, wr_factor=1):

    layer_config = {}
    if isinstance(layer, GAPLayer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        coarse_factor = math.ceil(config[0] * layer.channels)
        assert input_shape[0] == output_shape[0], "Input and output batch dimension must match"
        layer_config["batch_size"] = input_shape[0]
        layer_config["channels_in"] = input_shape[1]
        layer_config["depth_in"] = input_shape[2]
        layer_config["height_in"] = input_shape[3]
        layer_config["width_in"] = input_shape[4]
        layer_config["channels_out"] = output_shape[1]
        layer_config["depth_out"] = output_shape[2]
        layer_config["height_out"] = output_shape[3]
        layer_config["width_out"] = output_shape[4]
        layer_config["coarse_factor"] = coarse_factor
        layer_config["wr_factor"] = wr_factor
    elif isinstance(layer, Convolutional3DLayer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        kerner_shape = layer.kernel_shape
        bias_shape = layer.bias_shape
        padding = layer.padding
        stride = layer.stride
        groups = layer.groups
        depthwise = 1 if layer.depthwise else 0
        pointwise = 1 if layer.pointwise else 0
        fine_factor = math.ceil(config[0] * layer.kd * layer.kh * layer.kw)
        coarse_in_factor = math.ceil(config[1] * layer.channels)
        coarse_out_factor = math.ceil(config[2] * layer.filters)
        # wr_factor = config[3] if len(config) >= 4 else 1
        assert input_shape[0] == output_shape[0], "Input and output batch dimension must match"
        layer_config["batch_size"] = input_shape[0]
        layer_config["channels_in"] = input_shape[1]
        layer_config["depth_in"] = input_shape[2]
        layer_config["height_in"] = input_shape[3]
        layer_config["width_in"] = input_shape[4]
        layer_config["channels_out"] = output_shape[1]
        layer_config["depth_out"] = output_shape[2]
        layer_config["height_out"] = output_shape[3]
        layer_config["width_out"] = output_shape[4]
        layer_config["kernel_depth"] = kerner_shape[0]
        layer_config["kernel_height"] = kerner_shape[1]
        layer_config["kernel_width"] = kerner_shape[2]
        layer_config["shape_bias"] = bias_shape[0] if bias_shape else 0
        layer_config["pad_depth"] = padding[0]
        layer_config["pad_height"] = padding[1]
        layer_config["pad_width"] = padding[2]
        layer_config["stride_depth"] = stride[0]
        layer_config["stride_height"] = stride[1]
        layer_config["stride_width"] = stride[2]
        layer_config["groups"] = groups
        layer_config["depthwise"] = depthwise
        layer_config["pointwise"] = pointwise
        layer_config["fine_factor"] = fine_factor
        layer_config["coarse_in_factor"] = coarse_in_factor
        layer_config["coarse_out_factor"] = (
            coarse_out_factor if not depthwise else coarse_in_factor
        )
        layer_config["wr_factor"] = wr_factor
    elif isinstance(layer, Pooling3DLayer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        kerner_shape = layer.kernel_shape
        padding = layer.padding
        stride = layer.stride
        fine_factor = math.ceil(config[0] * layer.kd * layer.kh * layer.kw)
        coarse_factor = math.ceil(config[1] * layer.channels)
        assert input_shape[0] == output_shape[0], "Input and output batch dimension must match"
        layer_config["batch_size"] = input_shape[0]
        layer_config["channels_in"] = input_shape[1]
        layer_config["depth_in"] = input_shape[2]
        layer_config["height_in"] = input_shape[3]
        layer_config["width_in"] = input_shape[4]
        layer_config["channels_out"] = output_shape[1]
        layer_config["depth_out"] = output_shape[2]
        layer_config["height_out"] = output_shape[3]
        layer_config["width_out"] = output_shape[4]
        layer_config["kernel_depth"] = kerner_shape[0]
        layer_config["kernel_height"] = kerner_shape[1]
        layer_config["kernel_width"] = kerner_shape[2]
        layer_config["pad_depth"] = padding[0]
        layer_config["pad_height"] = padding[1]
        layer_config["pad_width"] = padding[2]
        layer_config["stride_depth"] = stride[0]
        layer_config["stride_height"] = stride[1]
        layer_config["stride_width"] = stride[2]
        layer_config["op_type"] = layer.op_type
        layer_config["fine_factor"] = fine_factor
        layer_config["coarse_factor"] = coarse_factor
        layer_config["wr_factor"] = wr_factor
    elif isinstance(layer, ActivationLayer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        coarse_factor = math.ceil(config[0] * layer.channels)
        assert input_shape[0] == output_shape[0], "Input and output batch dimension must match"
        layer_config["batch_size"] = input_shape[0]
        layer_config["channels_in"] = input_shape[1]
        layer_config["depth_in"] = input_shape[2]
        layer_config["height_in"] = input_shape[3]
        layer_config["width_in"] = input_shape[4]
        layer_config["channels_out"] = output_shape[1]
        layer_config["depth_out"] = output_shape[2]
        layer_config["height_out"] = output_shape[3]
        layer_config["width_out"] = output_shape[4]
        layer_config["coarse_factor"] = coarse_factor
        layer_config["op_type"] = layer.op_type
        layer_config["wr_factor"] = wr_factor
    elif isinstance(layer, ElementWiseLayer):
        input_shape = layer.input_shape
        output_shape = layer.input_shape
        broadcasting = 1 if layer.broadcasting else 0
        op_type = layer.op_type
        coarse_factor = math.ceil(config[0] * layer.input_shape[1])
        assert input_shape[0] == output_shape[0], "Input and output batch dimension must match"
        layer_config["batch_size"] = input_shape[0]
        layer_config["channels_in"] = input_shape[1]
        layer_config["depth_in"] = input_shape[2]
        layer_config["height_in"] = input_shape[3]
        layer_config["width_in"] = input_shape[4]
        layer_config["channels_out"] = output_shape[1]
        layer_config["depth_out"] = output_shape[2]
        layer_config["height_out"] = output_shape[3]
        layer_config["width_out"] = output_shape[4]
        layer_config["broadcasting"] = broadcasting
        layer_config["op_type"] = op_type
        layer_config["coarse_factor"] = coarse_factor
        layer_config["wr_factor"] = wr_factor
    elif isinstance(layer, BatchNorm3DLayer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        coarse_factor = math.ceil(config[0] * layer.channels)
        assert input_shape[0] == output_shape[0], "Input and output batch dimension must match"
        layer_config["batch_size"] = input_shape[0]
        layer_config["channels_in"] = input_shape[1]
        layer_config["depth_in"] = input_shape[2]
        layer_config["height_in"] = input_shape[3]
        layer_config["width_in"] = input_shape[4]
        layer_config["channels_out"] = output_shape[1]
        layer_config["depth_out"] = output_shape[2]
        layer_config["height_out"] = output_shape[3]
        layer_config["width_out"] = output_shape[4]
        layer_config["coarse_factor"] = coarse_factor
        layer_config["wr_factor"] = wr_factor
    elif isinstance(layer, SqueezeExcitationLayer):
        pass
    elif isinstance(layer, FCLayer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        weights_shape = layer.weights_shape
        bias_shape = layer.bias_shape
        coarse_in_factor = math.ceil(config[0] * layer.dim_in)
        coarse_out_factor = math.ceil(config[1] * layer.dim_out)
        assert input_shape[0] == output_shape[0], "Input and output batch dimension must match"
        layer_config["batch_size"] = input_shape[0]
        layer_config["features_in"] = input_shape[1]
        layer_config["features_out"] = output_shape[1]
        layer_config["weights_rows"] = weights_shape[0]
        layer_config["weights_cols"] = weights_shape[1]
        layer_config["shape_bias"] = bias_shape[0] if bias_shape else 0
        layer_config["coarse_in_factor"] = coarse_in_factor
        layer_config["coarse_out_factor"] = coarse_out_factor
        layer_config["wr_factor"] = wr_factor
    else:
        assert False, "Not supported layer"

    return layer_config


def get_config_points(name, file_name, is_partitioning=False):
    layers_df = pd.read_csv(file_name)

    curr_layer_df = layers_df.loc[layers_df["Layer"] == name].reset_index()

    if not is_partitioning:
        return curr_layer_df["config"].apply(lambda x: json.loads(x)).to_list()
    else:
        if "BatchNormalization" in name.split("_") or "Swish" in name.split("_"):
            first_point = math.floor(len(curr_layer_df["config"]) * 0.25)
            second_point = math.floor(len(curr_layer_df["config"]) * 0.75)
            return [
                json.loads(curr_layer_df["config"][first_point]),
                json.loads(curr_layer_df["config"][second_point]),
            ]
        else:
            return curr_layer_df["config"].apply(lambda x: json.loads(x)).to_list()


def get_paretto_csv(
    file_name_par, file_name, pareto_type="MinMin", xaxis="Latency(C)", yaxis="DSP(%)"
):

    with open(file_name_par, mode="a") as pareto_results:
        csv_writer_par = csv.writer(
            pareto_results, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        layers_df = pd.read_csv(file_name)

        layers = layers_df["Layer"].unique()
        for l in layers:
            curr_df = layers_df.loc[layers_df["Layer"] == l].reset_index()
            if not ("Conv" in l.split("_") or "Se" in l.split("_")):
                for ind in curr_df.index:
                    csv_writer_par.writerow(curr_df.iloc[ind].to_list()[1:])
            else:
                print("Calculating pareto front for layer {}".format(l))

                x_axis = curr_df[xaxis].to_numpy()
                y_axis = curr_df[yaxis].to_numpy()

                scores = np.zeros((x_axis.shape[0], 2))
                scores[:, 0] = x_axis
                scores[:, 1] = y_axis

                pareto = find_pareto(scores, pareto_type)
                pareto_front = scores[pareto]

                pareto_front_df = pd.DataFrame(pareto_front)
                pareto_front_df.sort_values(0, inplace=True)
                pareto_front = pareto_front_df.values

                for p in pareto:
                    csv_writer_par.writerow(curr_df.iloc[p].to_list()[1:])

def update_report_config(template_dict: dict, result_dict: dict, name: str, layer_type: str, layer_hw) -> dict:
    template_dict.pop("Layer", None)
    template_dict["Type"] = layer_type
    template_dict["Latency(C)-No-Depth"] = result_dict["latency(C)"] - result_dict["depth"],
    template_dict["Latency(C)"] = result_dict["latency(C)"],
    template_dict["Latency(S)"] = result_dict["latency(S)"],
    template_dict["GOP/s"] = result_dict["GOP/s"],
    template_dict["GOPs"] = result_dict["GOPs"],
    template_dict["volumes/s"] = result_dict["vols/s"],
    template_dict["DSPs"] = result_dict["DSP_RAW"],
    template_dict["DSP(%)"] = result_dict["DSP"],
    template_dict["BRAMs"] = result_dict["BRAM_RAW"],
    template_dict["BRAM(%)"] = result_dict["BRAM"],
    template_dict["RateIn"] = result_dict["rateIn"],
    template_dict["RateOut"] = result_dict["rateOut"],
    template_dict["Depth"] = result_dict["depth"],
    template_dict["Branch Depth"] = result_dict["branch_depth"],
    template_dict["Muls"] = result_dict["muls"],
    template_dict["Adds"] = result_dict["adds"],
    template_dict["Mem(W)"] = result_dict["memWords"],
    template_dict["Mem(KB)"] = result_dict["memKBs"],
    template_dict["DataSizeIn(MB)"] = result_dict["dataSizeIn"],
    template_dict["DataSizeOut(MB)"] = result_dict["dataSizeOut"],
    template_dict["MemBoundIn"] = result_dict["memBoundedIn"],
    template_dict["MemBoundOut"] = result_dict["memBoundedOut"],

    if isinstance(layer_hw, Convolutional3DLayer):
        layer_config = generate_layer_config(layer_hw, result_dict["config"], wr_factor=result_dict["wr_factor"])
    elif isinstance(layer_hw, Pooling3DLayer):
        layer_config = generate_layer_config(layer_hw, result_dict["config"])
    elif isinstance(layer_hw, GAPLayer):
        layer_config = generate_layer_config(layer_hw, result_dict["config"])
    elif isinstance(layer_hw, FCLayer):
        layer_config = generate_layer_config(layer_hw, result_dict["config"])
    elif isinstance(layer_hw, ElementWiseLayer):
        layer_config = generate_layer_config(layer_hw, result_dict["config"])
    elif isinstance(layer_hw, ActivationLayer):
        layer_config = generate_layer_config(layer_hw, result_dict["config"])
    else:
        raise ValueError("Layer type {} not supported".format(layer_type))
    template_dict["config"] = {}
    for key in layer_config:
        template_dict["config"][key] = layer_config[key]

    for key in template_dict:
        if isinstance(template_dict[key], tuple):
            template_dict[key] = template_dict[key][0]

    return {name :template_dict}

def update_report_file(filename: str, final_dict: dict) -> None:
    if os.path.isfile(filename) is False:
        _logger.info("File {} does not exists. Creating a new one.".format(filename))
        open(filename, 'a').close()

    if os.path.getsize(filename) == 0:
        dictObj = {}
    else:
        with open(filename, 'r') as fp:
            dictObj = json.load(fp)

    dictObj |= final_dict

    with open(filename, 'w') as json_file:
        json.dump(dictObj, json_file,
                            indent=2)

def check_configuration_validation(config, layers):
    valid = True

    layer_graph = {}
    for i, layer in enumerate(layers.values()):
        input_streams = []
        if isinstance(layer["layer"], Convolutional3DLayer):
            streams_in, streams_out = layer["layer"].get_num_streams()
            streams_in = math.ceil(streams_in * config[i][1])
            streams_out = math.ceil(streams_out * config[i][2])
            input_streams.append(streams_in)
        if isinstance(layer["layer"], Pooling3DLayer):
            streams_in, streams_out = layer["layer"].get_num_streams()
            streams_in = math.ceil(streams_in * config[i][1])
            streams_out = math.ceil(streams_out * config[i][1])
            input_streams.append(streams_in)
        elif isinstance(layer["layer"], BatchNorm3DLayer):
            streams_in, streams_out = layer["layer"].get_num_streams()
            streams_in = math.ceil(streams_in * config[i][0])
            streams_out = math.ceil(streams_out * config[i][0])
            input_streams.append(streams_in)
        elif isinstance(layer["layer"], GAPLayer):
            streams_in, streams_out = layer["layer"].get_num_streams()
            streams_in = math.ceil(streams_in * config[i][0])
            streams_out = math.ceil(streams_out * config[i][1])
            input_streams.append(streams_in)
        elif isinstance(layer["layer"], ActivationLayer):
            streams_in, streams_out = layer["layer"].get_num_streams()
            streams_in = math.ceil(streams_in * config[i][0])
            streams_out = math.ceil(streams_out * config[i][0])
            input_streams.append(streams_in)
        elif isinstance(layer["layer"], SqueezeExcitationLayer):
            print("config for layer (not supported) {} -> {}".format(layer, config))
        elif isinstance(layer["layer"], ElementWiseLayer):
            streams_in1, streams_in2, streams_out = layer["layer"].get_num_streams()
            streams_in1 = math.ceil(streams_in1 * config[i][0])
            streams_in2 = math.ceil(streams_in2 * config[i][1])
            streams_out = math.ceil(streams_out * config[i][2])
            input_streams.append(streams_in1)
            input_streams.append(streams_in2)
        else:
            assert False, "Layer {} is not yet supported".format(layer)

        if i not in layer_graph.keys():
            layer_graph[i] = {}
        layer_graph[i]["node_in"] = layer["node_in"]
        layer_graph[i]["node_out"] = layer["node_out"]
        layer_graph[i]["streams_in"] = input_streams
        layer_graph[i]["streams_out"] = streams_out

    input_node = layer_graph[list(layer_graph.keys())[0]]["node_in"]
    output_node = layer_graph[list(layer_graph.keys())[-1]]["node_out"]
    for n, v in layer_graph.items():
        for nd_in, strm_in in zip(v["node_in"], v["streams_in"]):
            if nd_in in input_node:
                continue
            if strm_in > get_out_streams(layer_graph, nd_in):
                valid = False
                break
        if not valid:
            break

    return valid


def get_out_streams(layer_graph, node_out):
    for v in layer_graph.values():
        if v["node_out"] == node_out:
            return v["streams_out"]
    assert False, "Cannot find node {} in the layer graph.".format(node_out)


def get_split_points(graph):
    split_points = []
    for node in graph.nodes():
        if graph.out_degree[node] > 1:
            split_points.append(node)
    return split_points


def get_merge_points(graph):
    merge_points = []
    for node in graph.nodes():
        if graph.in_degree[node] > 1:
            merge_points.append(node)
    return merge_points


def get_worst_case_buffering(graph):
    branch_edges = get_branch_start_end_points(graph)

    branch_buffer = 0
    for (splt, mrg) in branch_edges:
        all_paths = [p for p in nx.all_simple_paths(graph, splt, mrg)]
        num_sub_branches = len(all_paths) - 2

        shortest_path = nx.shortest_path(graph, splt, mrg)
        merge_node = shortest_path[-1]
        pre_merge_node = shortest_path[-2]
        assert (graph.nodes[pre_merge_node]["hw"].output_shape
                    == graph.nodes[merge_node]["hw"].input_shape
                ), "Layers input and output shapes does not match"
        #TODO: This is the work case scenario for buffering the whole feature map. A more accurate design would be to calculate the depths for each layer in each branch and accumulate the depths to get the total buffer depth which will be the minimum between the work case scenario and the actual buffer depth.
        branch_buffer += np.prod(np.array(graph.nodes[pre_merge_node]["hw"].output_shape))
    return branch_buffer

def get_branch_edges(graph):
    merge_points = get_merge_points(graph)

    branch_edges = []
    for mrg in merge_points:
        branch_edges.append(list(graph.in_edges(mrg)))

    return branch_edges


def get_conbinations(list1, list2):
    combs = [x * y for x, y in itertools.product(list1, list2)]
    return list(np.unique(np.array(combs)))


def get_input_node(graph):
    for node in graph.nodes():
        if graph.in_degree[node] == 0:
            return node


def get_output_node(graph):
    for node in graph.nodes():
        if graph.out_degree[node] == 0:
            return node


def get_branch_start_end_points(graph):
    result = []
    split_points = get_split_points(graph)
    for sp in split_points:
        merge_point = None
        next_node = sp
        extra_split_points = 0
        while True:
            if graph.out_degree[next_node] == 1:
                next_node = list(graph.successors(next_node))[0]
            elif graph.out_degree[next_node] > 1:
                extra_split_points += 1
                next_node = list(graph.successors(next_node))[0]
            else:
                break

            if graph.in_degree[next_node] > 1:
                extra_split_points -= 1
                if extra_split_points == 0:
                    merge_point = next_node
                    break
        result.append((sp, merge_point))
    return result


def update_graph(graph, split_points=None, squeeze_layers=None):
    if split_points is None and squeeze_layers is None:
        return graph

    if split_points is not None:
        new_nodes = []
        new_edges = []
        for sp in split_points:
            new_node_name = "Split_" + sp
            new_nodes.append(new_node_name)

            next_nodes = list(graph.successors(sp))

            edges_out = list(graph.out_edges(sp))

            assert (
                len(next_nodes) > 1 and len(edges_out) > 1
            ), "Split point {} cannot have only one successor".format(sp)

            graph.remove_edges_from(edges_out)

            edge = (sp, new_node_name)
            new_edges.append(edge)
            for nd in next_nodes:
                edge = (new_node_name, nd)
                new_edges.append(edge)

        if new_nodes or new_edges:
            graph.update(edges=new_edges, nodes=new_nodes)

    if squeeze_layers is not None:
        new_nodes = []
        new_edges = []
        for sl in squeeze_layers:
            new_node_name = "Squeeze_" + sl[0] + "_" + sl[1]
            new_nodes.append(new_node_name)

            prev_nodes = list(graph.predecessors(sl[1]))

            edges_in = list(graph.in_edges(sl[1]))

            for ei in edges_in:
                if sl[0] in ei[0] and sl[1] in ei[1]:
                    graph.remove_edge(ei[0], ei[1])

            edge = (new_node_name, sl[1])
            new_edges.append(edge)
            for pn in prev_nodes:
                if sl[0] in pn:
                    edge = (pn, new_node_name)
                    new_edges.append(edge)

        if new_nodes or new_edges:
            graph.update(edges=new_edges, nodes=new_nodes)

    return graph


def add_supportive_nodes_config(graph, config):
    for node in graph.nodes():
        if "Split_" in node and not node in config.keys():
            parent_node = node.split("Split_")[1]
            config[node] = {
                "shape_in": config[parent_node]["shape_out"],
                "shape_out": config[parent_node]["shape_out"],
                "coarse_factor": config[parent_node]["coarse_factor"]
                if "coarse_factor" in config[parent_node].keys()
                else config[parent_node]["coarse_out_factor"],
            }
        if "Squeeze_" in node and not node in config.keys():
            node_decomp = node.split("_")
            parent_node_1 = f"{node_decomp[1]}_{node_decomp[2]}"
            # parent_node_2 = f"{node_decomp[3]}_{node_decomp[4]}"
            config[node] = {
                "shape_in": config[parent_node_1]["shape_out"],
                "shape_out": config[parent_node_1]["shape_out"],
                "coarse_factor": config[parent_node_1]["coarse_factor"]
                if "coarse_factor" in config[parent_node_1].keys()
                else config[parent_node_1]["coarse_out_factor"],
            }
    return config


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_channels_bins(channels, plot_lbow=False, plot_hist=False):
    X = np.array(channels).reshape(-1, 1)

    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 10)

    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)

        distortions.append(
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = (
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )
        mapping2[k] = kmeanModel.inertia_

    distortions_res = normalizeData(np.array(sorted(mapping1.values(), reverse=True)))
    k_dist = 0
    for i in range(len(distortions_res) - 1):
        dist = np.linalg.norm(distortions_res[i] - distortions_res[i + 1])
        if dist < 0.05:
            k_dist = i + 1
            _logger.info(f"Optimal number of clusters calculated: {i + 1}")
            break

    Kmean = KMeans(n_clusters=k_dist)
    Kmean.fit(X)
    kmeans_bins = np.sort(Kmean.cluster_centers_[:, 0])

    if plot_lbow:
        plt.plot(K, distortions, "bx-")
        plt.xlabel("Values of K")
        plt.ylabel("Distortion")
        plt.title("The Elbow Method using Distortion")
        plt.tight_layout()
        plt.show()

    # inertias_res = normalizeData(
    #     np.array(sorted(mapping2.values(), reverse=True)))
    # k_inertia = 0
    # for i in range(len(inertias_res)-1):
    #     dist = np.linalg.norm(inertias_res[i]-inertias_res[i+1])
    #     if dist < 0.1:
    #         k_inertia = i + 1
    #         print(f"{i + 1} -> {dist}")
    #         break

    df = pd.DataFrame({"channels": channels})
    df["channels_bin"] = pd.qcut(df["channels"], q=k_dist)
    bin_edges = df["channels_bin"].unique()

    # bin_edges = np.histogram_bin_edges(X, bins=k_dist+1)
    if plot_hist:
        sns.histplot(X, bins=X.shape[0])
        plt.tight_layout()
        plt.show()

    return bin_edges


def get_pool_type(
    layer: dict(),
    discriminate_kernel_size: bool = False,
    discriminate_stide: bool = False,
    discriminate_padding: bool = False,
) -> str:
    pool_type = "Pooling"
    kernel_shape = layer["kernel"]
    padding = layer["padding"]
    stride = layer["stride"]

    if discriminate_kernel_size:
        pool_type += "k{}".format("".join(map(str, kernel_shape)))
    if discriminate_stide:
        pool_type += "s{}".format("".join(map(str, stride)))
    if discriminate_padding:
        pool_type += "p{}".format("".join(map(str, padding)))

    return pool_type

def get_conv_type(
    layer: dict(),
    discriminate_kernel_size: bool = False,
    discriminate_stide: bool = False,
    discriminate_padding: bool = False,
    discriminate_channels_filters: bool = False,
) -> str:
    conv_type = "Conv3D"
    cin = layer["shape_in"][0][1]
    cout = layer["shape_out"][1]
    kernel_shape = layer["kernel"][2:]
    padding = layer["padding"]
    stride = layer["stride"]
    groups = layer["groups"]
    if cin == cout and groups == cout:
        conv_type += "Dw"
    if kernel_shape.count(1) == len(kernel_shape):
        conv_type += "Pw"
    if discriminate_kernel_size:
        conv_type += "k{}".format("".join(map(str, kernel_shape)))
    if discriminate_stide:
        conv_type += "s{}".format("".join(map(str, stride)))
    if discriminate_padding:
        conv_type += "p{}".format("".join(map(str, padding)))
    if discriminate_channels_filters:
        conv_type += "c{}".format(cin)
        conv_type += "f{}".format(cout)
    return conv_type

def calc_mape(value_a: int, value_b: int) -> float:
    return (abs(value_a - value_b) / abs(value_a)) * 100

def calc_conv_out_shape(cin: int, din: int, hin: int, kernel_shape: list, pad: list, stride: list, dw: bool, chan_dist_thresh: int = 10, is_pool: bool = False) -> list:
    kd, kh, _ = kernel_shape
    pad_d, pad_h, _ = pad
    if kd == 1:
        pad_d = 0
    if kh == 1:
        pad_h = 0
    stride_d, stride_h, _ = stride
    dout = max(1, math.floor((din + 2*pad_d -kd)/stride_d + 1))
    hout = max(1, math.floor((hin + 2*pad_h -kh)/stride_h + 1))
    wout = hout

    if dw or is_pool:
        cout = cin
    else:
        c_in_range = math.ceil(cin*chan_dist_thresh/100)
        cout = np.random.randint(cin, cin+c_in_range)

    return [1, int(cout), int(dout), int(hout), int(wout)]

def get_random_arbitrary_shape(
    graph: nx.DiGraph, bb_type: str, lookuptable: dict, previous_config: dict = None, chan_dist_thresh: int = 10, depth_dist_thresh: int = 10, height_dist_thresh: int = 10
) -> np.array:
    in_shapes = []
    out_shapes = []
    kernel_shape = [int(x) for x in bb_type.split("k")[-1][:3]] if "Conv" in bb_type or "Pooling" in bb_type else []
    padding = [int(x) for x in bb_type.split("p")[-1][:3]] if "Conv" in bb_type or "Pooling" in bb_type else []
    stride = [int(x) for x in bb_type.split("s")[-1][:3]] if "Conv" in bb_type or "Pooling" in bb_type else []
    dw = True if "Dw" in bb_type else False
    for node in graph.nodes:
        bb = lookuptable[graph.nodes[node]["hw_type"]]
        if bb == "Activation" and len(graph.nodes[node]["hw"].input_shape) < 5:
            continue
        if bb in bb_type:
            in_shapes.append(graph.nodes[node]["hw"].input_shape)
            out_shapes.append(graph.nodes[node]["hw"].output_shape)

    final_shape_in = []
    final_shape_out = []
    if len(in_shapes[0]) == 5:
        if previous_config is not None and bb_type in previous_config:
            prev_c_in = previous_config[bb_type]["config"]["channels_in"]
            prev_d_in = previous_config[bb_type]["config"]["depth_in"]
            prev_h_in = previous_config[bb_type]["config"]["height_in"]

            c_in_range = math.ceil(prev_c_in*chan_dist_thresh/100)
            d_in_range = math.ceil(prev_d_in*depth_dist_thresh/100)
            h_in_range = math.ceil(prev_h_in*height_dist_thresh/100)

            c_in = np.random.randint(max(1, prev_c_in-c_in_range), prev_c_in+c_in_range)
            d_in = np.random.randint(max(1, prev_d_in-d_in_range), prev_d_in+d_in_range)
            h_in = np.random.randint(max(1, prev_h_in-h_in_range), prev_h_in+h_in_range)

            if "Conv" in bb_type:
                _, c_out, d_out, h_out, _ = calc_conv_out_shape(c_in, d_in, h_in, kernel_shape, padding, stride, dw, chan_dist_thresh)
            elif "Pooling" in bb_type:
                _, c_out, d_out, h_out, _ = calc_conv_out_shape(c_in, d_in, h_in, kernel_shape, padding, stride, dw, chan_dist_thresh, is_pool=True)
            elif "GlobalAveragePool" in bb_type:
                c_out = c_in
                d_out = 1
                h_out = 1
            else:
                c_out = c_in
                d_out = d_in
                h_out = h_in

            w_in = h_in
            w_out = h_out

            assert c_in<=c_out and d_in >= d_out and h_in >= h_out and w_in >= w_out, "Invalid output shape: {} -> {}".format([1, c_in, d_in, h_in, w_in], [1, c_out, d_out, h_out, w_out])
            final_shape_in, final_shape_out = [1, int(c_in), int(d_in), int(h_in), int(w_in)], [1, int(c_out), int(d_out), int(h_out), int(w_out)]

        _, c_min, d_min, h_min, _ = np.min(np.array(in_shapes), axis=0)
        _, c_max, d_max, h_max, _ = np.max(np.array(in_shapes), axis=0)
        c_in = np.random.randint(c_min, c_max) if c_min != c_max else c_min
        d_in = np.random.randint(d_min, d_max) if d_min != d_max else d_min
        h_in = np.random.randint(h_min, h_max) if h_min != h_max else h_min
        w_in = h_in

        if "Conv" in bb_type:
            _, c_out, d_out, h_out, _ = calc_conv_out_shape(c_in, d_in, h_in, kernel_shape, padding, stride, dw, chan_dist_thresh)
        elif "Pooling" in bb_type:
                _, c_out, d_out, h_out, _ = calc_conv_out_shape(c_in, d_in, h_in, kernel_shape, padding, stride, dw, chan_dist_thresh, is_pool=True)
        elif "GlobalAveragePool" in bb_type:
            c_out = c_in
            d_out = 1
            h_out = 1
        else:
            c_out = c_in
            d_out = d_in
            h_out = h_in

        w_in = h_in
        w_out = h_out

        assert c_in<=c_out and d_in >= d_out and h_in >= h_out and w_in >= w_out, "Invalid output shape: {} -> {}".format([1, c_in, d_in, h_in, w_in], [1, c_out, d_out, h_out, w_out])
        final_shape_in, final_shape_out = [1, int(c_in), int(d_in), int(h_in), int(w_in)], [1, int(c_out), int(d_out), int(h_out), int(w_out)]
    elif len(in_shapes[0]) == 2:
        _, features_min = np.min(np.array(in_shapes), axis=0)
        _, features_max = np.max(np.array(in_shapes), axis=0)
        features_in = np.random.randint(features_min, features_max) if features_min != features_max else features_min

        _, features_min_out = np.min(np.array(out_shapes), axis=0)
        _, features_max_out = np.max(np.array(out_shapes), axis=0)
        features_out = np.random.randint(features_min_out, features_max_out) if features_min_out != features_max_out else features_min_out
        final_shape_in, final_shape_out = [1, int(features_in)], [1, int(features_out)]

    return final_shape_in, final_shape_out

def get_random_shape(
    graph: nx.DiGraph, bb_type: str, lookuptable: dict, previous_config: dict = None, chan_dist_thresh: int = 10, depth_dist_thresh: int = 10, height_dist_thresh: int = 10
) -> np.array:
    shapes_list = []
    for n in graph.nodes():
        bb = lookuptable[graph.nodes[n]["hw_type"]]
        if bb in bb_type and bb == "Gemm":
            shapes_list.append(
                [graph.nodes[n]["hw"].input_shape, graph.nodes[n]["hw"].output_shape]
            )
        elif bb in bb_type and np.prod(graph.nodes[n]["hw"].input_shape[2:]) > 1:
            shapes_list.append(
                [graph.nodes[n]["hw"].input_shape, graph.nodes[n]["hw"].output_shape]
            )

    while True:
        final_shapes = random.choice(shapes_list)
        shape_in = final_shapes[0]
        shape_out = final_shapes[1]
        if previous_config is not None and bb_type in previous_config:
            if len(shape_in) >= 5:
                prev_shape_in = [previous_config[bb_type]["config"]["batch_size"],
                                previous_config[bb_type]["config"]["channels_in"],
                                previous_config[bb_type]["config"]["depth_in"],
                                previous_config[bb_type]["config"]["height_in"],
                                previous_config[bb_type]["config"]["width_in"]]
            else:
                prev_shape_in = [previous_config[bb_type]["config"]["batch_size"],
                                previous_config[bb_type]["config"]["features_in"]]

            mape_channels = calc_mape(prev_shape_in[1], shape_in[1])
            if len(prev_shape_in) >= 5:
                mape_height = calc_mape(prev_shape_in[3], shape_in[3])
            else:
                mape_height = 0
            if mape_channels > chan_dist_thresh or mape_height > height_dist_thresh:
                continue
            break
        else:
            break
    return shape_in, shape_out


def get_minmax_input_channels(graph: nx.DiGraph, bb_type: str) -> int:
    max_input_channels = -1
    min_input_channels = 10000
    for n in graph.nodes():
        bb = graph.nodes[n]["hw_type"]
        if bb == bb_type:
            max_input_channels = max(
                max_input_channels, graph.nodes[n]["hw"].input_shape[1]
            )
            min_input_channels = min(
                min_input_channels, graph.nodes[n]["hw"].input_shape[1]
            )
    return min_input_channels, max_input_channels


def get_minmax_output_channels(graph: nx.DiGraph, bb_type: str) -> int:
    max_output_channels = -1
    min_output_channels = 10000
    for n in graph.nodes():
        bb = graph.nodes[n]["hw_type"]
        if bb == bb_type:
            max_output_channels = max(
                max_output_channels, graph.nodes[n]["hw"].output_shape[1]
            )
            min_output_channels = min(
                min_output_channels, graph.nodes[n]["hw"].output_shape[1]
            )
    return min_output_channels, max_output_channels


def get_minmax_depth(graph: nx.DiGraph, bb_type: str) -> int:
    max_depth = -1
    min_depth = 10000
    for n in graph.nodes():
        bb = graph.nodes[n]["hw_type"]
        if bb == bb_type:
            max_depth = max(max_depth, graph.nodes[n]["hw"].input_shape[2])
            min_depth = min(min_depth, graph.nodes[n]["hw"].input_shape[2])
    return min_depth, max_depth


def get_minmax_height(graph: nx.DiGraph, bb_type: str) -> int:
    max_height = -1
    min_height = 10000
    for n in graph.nodes():
        bb = graph.nodes[n]["hw_type"]
        if bb == bb_type:
            max_height = max(max_height, graph.nodes[n]["hw"].input_shape[3])
            min_height = min(min_height, graph.nodes[n]["hw"].input_shape[3])
    return min_height, max_height


def get_minmax_width(graph: nx.DiGraph, bb_type: str) -> int:
    max_width = -1
    min_width = 10000
    for n in graph.nodes():
        bb = graph.nodes[n]["hw_type"]
        if bb == bb_type:
            max_width = max(max_width, graph.nodes[n]["hw"].input_shape[4])
            min_width = min(min_width, graph.nodes[n]["hw"].input_shape[4])
    return min_width, max_width
