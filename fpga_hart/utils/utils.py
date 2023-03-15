import csv
import itertools
import json
import math
import os
import random
import re
from copy import deepcopy
from functools import reduce
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from fpga_hart import _logger
from fpga_hart.layers.activation_3d import Activation3DLayer
from fpga_hart.layers.batchnorm_3d import BatchNorm3DLayer
from fpga_hart.layers.convolutional_3d import Convolutional3DLayer
from fpga_hart.layers.elemwise_3d import ElementWise3DLayer
from fpga_hart.layers.fully_connected import FCLayer
from fpga_hart.layers.gap_3d import GAP3DLayer
from fpga_hart.layers.pooling_3d import Pooling3DLayer
from fpga_hart.layers.squeeze_excitation import SqueezeExcitationLayer
from fpga_hart.utils.graph_manipulation import get_out_streams

import scienceplots
plt.style.use(["science", "ieee", "grid"])


# helper function to perform sort
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

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

def generate_supportive_layer_config(layers, layers_config):
    for layer, config in layers.items():
        if config['type'] == 'Split':
            assert config["streams_in"] == config["streams_out"], "Split layer should have same number of streams in and out"
            reference_layer = config['ref_layer']
            layers_config[layer] = {
                "batch_size": layers_config[reference_layer]['batch_size'],
                "channels_in": layers_config[reference_layer]['channels_in'] if 'channels_in' in layers_config[reference_layer] else layers_config[reference_layer]['features_in'],
                "depth_in": layers_config[reference_layer]['depth_in'] if 'depth_in' in layers_config[reference_layer] else 1,
                "height_in": layers_config[reference_layer]['height_in'] if 'height_in' in layers_config[reference_layer] else 1,
                "width_in": layers_config[reference_layer]['width_in'] if 'width_in' in layers_config[reference_layer] else 1,
                "channels_out": layers_config[reference_layer]['channels_out'] if 'channels_out' in layers_config[reference_layer] else layers_config[reference_layer]['features_out'],
                "depth_out": layers_config[reference_layer]['depth_out'] if 'depth_out' in layers_config[reference_layer] else 1,
                "height_out": layers_config[reference_layer]['height_out'] if 'height_out' in layers_config[reference_layer] else 1,
                "width_out": layers_config[reference_layer]['width_out'] if 'width_out' in layers_config[reference_layer] else 1,
                "coarse_factor": config["streams_out"]
            }
        elif config['type'] == 'Squeeze':
            reference_layer = config['ref_layer_in']
            layers_config[layer] = {
                "batch_size": layers_config[reference_layer]['batch_size'],
                "channels_in": layers_config[reference_layer]['channels_in'] if 'channels_in' in layers_config[reference_layer] else layers_config[reference_layer]['features_in'],
                "depth_in": layers_config[reference_layer]['depth_in'] if 'depth_in' in layers_config[reference_layer] else 1,
                "height_in": layers_config[reference_layer]['height_in'] if 'height_in' in layers_config[reference_layer] else 1,
                "width_in": layers_config[reference_layer]['width_in'] if 'width_in' in layers_config[reference_layer] else 1,
                "channels_out": layers_config[reference_layer]['channels_out'] if 'channels_out' in layers_config[reference_layer] else layers_config[reference_layer]['features_out'],
                "depth_out": layers_config[reference_layer]['depth_out'] if 'depth_out' in layers_config[reference_layer] else 1,
                "height_out": layers_config[reference_layer]['height_out'] if 'height_out' in layers_config[reference_layer] else 1,
                "width_out": layers_config[reference_layer]['width_out'] if 'width_out' in layers_config[reference_layer] else 1,
                "coarse_in_factor": config["streams_in"],
                "coarse_out_factor": config["streams_out"]
            }
    return layers_config

def generate_layer_config(layer, config, wr_factor=1):

    layer_config = {}
    if isinstance(layer, GAP3DLayer):
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
    elif isinstance(layer, Activation3DLayer):
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
    elif isinstance(layer, ElementWise3DLayer):
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
    elif isinstance(layer_hw, GAP3DLayer):
        layer_config = generate_layer_config(layer_hw, result_dict["config"])
    elif isinstance(layer_hw, FCLayer):
        layer_config = generate_layer_config(layer_hw, result_dict["config"])
    elif isinstance(layer_hw, ElementWise3DLayer):
        layer_config = generate_layer_config(layer_hw, result_dict["config"])
    elif isinstance(layer_hw, Activation3DLayer):
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
        elif isinstance(layer["layer"], GAP3DLayer):
            streams_in, streams_out = layer["layer"].get_num_streams()
            streams_in = math.ceil(streams_in * config[i][0])
            streams_out = math.ceil(streams_out * config[i][1])
            input_streams.append(streams_in)
        elif isinstance(layer["layer"], Activation3DLayer):
            streams_in, streams_out = layer["layer"].get_num_streams()
            streams_in = math.ceil(streams_in * config[i][0])
            streams_out = math.ceil(streams_out * config[i][0])
            input_streams.append(streams_in)
        elif isinstance(layer["layer"], SqueezeExcitationLayer):
            print("config for layer (not supported) {} -> {}".format(layer, config))
        elif isinstance(layer["layer"], ElementWise3DLayer):
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

def get_conbinations(list1, list2):
    combs = [x * y for x, y in itertools.product(list1, list2)]
    return list(np.unique(np.array(combs)))

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

    df = pd.DataFrame({"channels": channels})
    df["channels_bin"] = pd.qcut(df["channels"], q=k_dist)
    bin_edges = df["channels_bin"].unique()

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
