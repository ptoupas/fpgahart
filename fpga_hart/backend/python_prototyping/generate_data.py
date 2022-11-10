import argparse
import json
import math
import os
import random
import sys
from collections import deque
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
import torch
from fpbinary import FpBinary
from torch import nn

from fpga_hart.network_representation.onnx_parser import OnnxModelParser
from fpga_hart.utils.utils import get_branch_start_end_points

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

np.set_printoptions(suppress=True, precision=5, threshold=sys.maxsize)


def get_x3d_m_layer_type(layer_order):

    LAYER_TYPE_1 = [
        "Relu",
        "Conv",
        "Relu",
        "Conv",
        "GlobalAveragePool",
        "Conv",
        "Relu",
        "Conv",
        "Sigmoid",
        "Mul",
        "Swish",
        "Conv",
        "Conv",
        "Add",
    ]
    LAYER_TYPE_2 = [
        "Relu",
        "Conv",
        "Relu",
        "Conv",
        "GlobalAveragePool",
        "Conv",
        "Relu",
        "Conv",
        "Sigmoid",
        "Mul",
        "Swish",
        "Conv",
        "Add",
    ]
    LAYER_TYPE_3 = ["Relu", "Conv", "Relu", "Conv", "Swish", "Conv", "Add"]
    LAYER_TYPE_4 = [
        "Conv",
        "Conv",
        "Relu",
        "Conv",
        "Relu",
        "Conv",
        "GlobalAveragePool",
        "Conv",
        "Relu",
        "Conv",
        "Sigmoid",
        "Mul",
        "Swish",
        "Conv",
        "Conv",
        "Add",
    ]
    LAYER_TYPE_5 = ["Relu", "Conv", "Relu", "GlobalAveragePool", "Gemm", "Relu", "Gemm"]

    if layer_order == LAYER_TYPE_1:
        return "type_1"
    elif layer_order == LAYER_TYPE_2:
        return "type_2"
    elif layer_order == LAYER_TYPE_3:
        return "type_3"
    elif layer_order == LAYER_TYPE_4:
        return "type_first"
    elif layer_order == LAYER_TYPE_5:
        return "type_last"
    elif layer_order == [
        "Relu",
        "Conv",
        "Relu",
        "Conv",
        "GlobalAveragePool",
        "Conv",
        "Relu",
        "Conv",
        "Sigmoid",
        "Mul",
        "Swish",
        "Conv",
    ]:
        return "type_seq"
    else:
        raise Exception("Unknown layer type")


def create_queue(data_in):
    batch, height, width, depth, channel = data_in.shape
    data_stream = deque([])

    for b in range(batch):
        for h in range(height):
            for w in range(width):
                for d in range(depth):
                    for c in range(channel):
                        # print(f"{b}, {h}, {w}, {d}, {c} = {data_in[b, h, w, d, c]}")
                        data_stream.appendleft(data_in[b, h, w, d, c])

    return data_stream


def show_wb(window_buffer):
    print("\nWindow Buffer")
    print(
        f"[[[{list(window_buffer[0][0][0])}, {list(window_buffer[0][1][0])}, {list(window_buffer[0][2][0])}]"
    )
    print(
        f"  [{list(window_buffer[1][0][0])}, {list(window_buffer[1][1][0])}, {list(window_buffer[1][2][0])}]"
    )
    print(
        f"  [{list(window_buffer[2][0][0])}, {list(window_buffer[2][1][0])}, {list(window_buffer[2][2][0])}]]"
    )
    print()
    print(
        f"[[[{list(window_buffer[0][0][1])}, {list(window_buffer[0][1][1])}, {list(window_buffer[0][2][1])}]"
    )
    print(
        f"  [{list(window_buffer[1][0][1])}, {list(window_buffer[1][1][1])}, {list(window_buffer[1][2][1])}]"
    )
    print(
        f"  [{list(window_buffer[2][0][1])}, {list(window_buffer[2][1][1])}, {list(window_buffer[2][2][1])}]]"
    )
    print()
    print(f" [[{0}, {0}, {0}]")
    print(f"  [{0}, {0}, {0}]")
    print(f"  [{0}, {0}, {0}]]]")


def show_lb(line_buffer, extra_buffer):
    print("\nLine Buffer 0")
    print(list(line_buffer[0][0]), len(line_buffer[0][0]))
    print(list(line_buffer[0][1]), len(line_buffer[0][1]))

    print("\nLine Buffer 1")
    print(list(line_buffer[1][0]), len(line_buffer[1][0]))
    print(list(line_buffer[1][1]), len(line_buffer[1][1]))

    print("\nLine Buffer 2")
    print(list(line_buffer[2][0]), len(line_buffer[2][0]))
    print(list(line_buffer[2][1]), len(line_buffer[2][1]))

    print("\nExtra Buffer 0")
    print(list(extra_buffer[0]), len(extra_buffer[0]))

    print("\nExtra Buffer 1")
    print(list(extra_buffer[1]), len(extra_buffer[1]))


def showres(
    out_buffer,
    shape,
    file_format,
    kh=3,
    kw=3,
    kd=3,
    pad_h=0,
    pad_w=0,
    pad_d=0,
    stride_h=1,
    stride_w=1,
    stride_d=1,
):
    print("\nInterprenting SW Output results\n")
    results = []
    batch, channel, depth, height, width = shape

    for b in range(0, batch):
        for h in range(0, height + 2 * pad_h):
            for w in range(0, width + 2 * pad_w):
                for d in range(0, depth + 2 * pad_d):
                    for c in range(0, channel):
                        print(f"b={b}  |  row={h}  |  col={w}  |  d={d}  |  c={c}")

                        if not (
                            (h < kh - 1)
                            or (h == kh - 1 and w < kw - 1)
                            or (h == kh - 1 and w == kw - 1 and d < kd - 1)
                        ):
                            for i in range(kh):
                                for j in range(kw):
                                    for k in range(kd):
                                        # for c in range(0, channel): # for 1 to 1 comparisson with HLS
                                        tmp = out_buffer[i][j][k].pop()
                                        if (
                                            d >= kd - 1
                                            and (d - kd + 1) % stride_d == 0
                                            and w >= kw - 1
                                            and (w - kw + 1) % stride_w == 0
                                            and h >= kh - 1
                                            and (h - kh + 1) % stride_h == 0
                                        ):
                                            results.append(tmp)
                                            if k == kd - 1:
                                                print(f"{tmp:.4f}")
                                            else:
                                                print(f"{tmp:.4f}", end=" ")
                                if d >= kd - 1 and w >= kw - 1 and h >= kh - 1:
                                    print()
                            if d >= kd - 1 and w >= kw - 1 and h >= kh - 1:
                                print("*" * 30)

    for k in range(kd):
        for i in range(kh):
            for j in range(kw):
                assert (
                    len(out_buffer[i][j][k]) == 0
                ), "There are still values in buffers"

    write_output_binary = np.array(results, dtype=np.float32)
    file_format = "txt"
    if file_format == "bin":
        write_output_binary.tofile("sw_output.bin")
    elif file_format == "txt":
        np.savetxt("sw_output.txt", write_output_binary, fmt="%.8f")
    else:
        raise Exception("Format not supported")


def array_init(arr):
    prev = arr
    for i in range(len(arr.shape) - 1):
        curr = np.zeros((prev.shape[0:-1]), dtype="object")
        for index, _ in np.ndenumerate(curr):
            if i == 0:
                curr[index] = (
                    "\t" * len(curr.shape)
                    + "{"
                    + ",".join([str(val) for val in prev[index]])
                    + "}"
                )
            else:
                curr[index] = (
                    "\t" * len(curr.shape)
                    + "{\n"
                    + ",\n".join([str(val) for val in prev[index]])
                    + "\n"
                    + "\t" * len(curr.shape)
                    + "}"
                )
        prev = curr
    return ",\n".join([i for i in prev])


def transform_weights(
    weights_raw, coarse_in, coarse_out, wr_factor=1, coarse_group=1, groups=1
):
    # parameters
    num_filters = int(weights_raw.shape[0] / (groups * coarse_out * wr_factor))
    num_channels = int(weights_raw.shape[1] / coarse_in)
    kd_size = weights_raw.shape[2]
    kh_size = weights_raw.shape[3]
    kw_size = weights_raw.shape[4]
    print(
        f"num_filters={num_filters}  |  num_channels={num_channels}  |  kd_size={kd_size}  |  kh_size={kh_size}  |  kw_size={kw_size}"
    )
    # correct output shape for weights
    weights = np.ndarray(
        shape=(
            wr_factor,
            coarse_group,
            coarse_in,
            coarse_out,
            int(groups / coarse_group),
            num_channels,
            num_filters,
            kh_size,
            kw_size,
            kd_size,
        ),
        dtype=np.float32,
        order="C",
    )

    # transform weights raw shape
    for index, _ in np.ndenumerate(weights):
        # print(f"index={index}, weights_raw[{index[4]*coarse_group*num_filters*wr_factor*coarse_out+index[1]*num_filters*wr_factor*coarse_out+index[6]*wr_factor*coarse_out+index[0]*coarse_out+index[3]}][{index[5]*coarse_in+index[2]}][{index[9]}][{index[7]}][{index[8]}]")
        weights[index] = weights_raw[
            index[4] * coarse_group * num_filters * wr_factor * coarse_out
            + index[1] * num_filters * wr_factor * coarse_out
            + index[6] * wr_factor * coarse_out
            + index[0] * coarse_out
            + index[3],
            index[5] * coarse_in + index[2],
            index[9],
            index[7],
            index[8],
        ]
    # merge channel and filter dimensions
    print("*" * 30)
    print(weights.shape)
    print(
        [
            wr_factor,
            coarse_in * coarse_group,
            coarse_out,
            int(groups / coarse_group) * num_channels,
            num_filters,
            kh_size,
            kw_size,
            kd_size,
        ]
    )
    weights = np.reshape(
        weights,
        [
            wr_factor,
            coarse_in * coarse_group,
            coarse_out,
            int(groups / coarse_group) * num_channels,
            num_filters,
            kh_size,
            kw_size,
            kd_size,
        ],
    )
    # print(weights)
    print(weights.shape)
    print("*" * 30)
    # return transformed weights
    return weights


def transform_weights_fc(
    weights_raw, coarse_in, coarse_out, wr_factor=1,):
    # parameters
    num_filters = int(weights_raw.shape[0] / (coarse_out * wr_factor))
    num_channels = int(weights_raw.shape[1] / coarse_in)
    print(
        f"num_filters={num_filters}  |  num_channels={num_channels}"
    )
    # correct output shape for weights
    weights = np.ndarray(
        shape=(
            wr_factor,
            coarse_in,
            coarse_out,
            num_channels,
            num_filters,
        ),
        dtype=np.float32,
        order="C",
    )

    # transform weights raw shape
    for index, _ in np.ndenumerate(weights):
        weights[index] = weights_raw[
            + index[4] * wr_factor * coarse_out
            + index[0] * coarse_out
            + index[2],
            index[3] * coarse_in + index[1],
        ]

    # merge channel and filter dimensions
    print("*" * 30)
    print(weights.shape)

    weights = np.reshape(
        weights,
        [
            wr_factor,
            coarse_in,
            coarse_out,
            num_channels,
            num_filters,
        ],
    )
    print(weights.shape)
    print("*" * 30)
    # return transformed weights
    return weights

def create_host_binary(data, fp_int_part, fp_frac_part, coarse_factor, dma_width) -> list:
    # create host binary
    word_bytes = (fp_int_part + fp_frac_part) // 8
    word_mask = ((2**(fp_int_part + fp_frac_part)) - 1)
    fp_data = np.array(data.flatten()).reshape(-1, coarse_factor)

    padding_offset = int(dma_width/(fp_int_part + fp_frac_part)) - coarse_factor

    fp_data = np.pad(fp_data, ((0,0), (0, padding_offset)), 'constant', constant_values=0)

    fp_data = [FpBinary(int_bits=fp_int_part, frac_bits=fp_frac_part, signed=True, value=v).bits_to_signed() for v in fp_data.flatten()]

    fp_data = b"".join([(f & word_mask).to_bytes(word_bytes, byteorder='little') for f in fp_data])

    return fp_data

def gap_3d(input_shape, coarse_in, file_format, store_path="generated_data/gap_3d"):
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    x = torch.randn(input_shape)
    print(x.numpy().shape)
    write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_input_binary.tofile(store_path + "/input.bin")
    elif file_format == "txt":
        np.savetxt(
            store_path + "/input.txt", write_input_binary.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")

    gap = torch.nn.AdaptiveAvgPool3d(1)
    out = gap(x)

    print(out.detach().numpy().shape)

    write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_out_binary.tofile(store_path + "/output.bin")
    elif file_format == "txt":
        np.savetxt(
            store_path + "/output.txt", write_out_binary.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")


def gemm(
    input_shape: int,
    output_shape: int,
    coarse_in: int,
    coarse_out: int,
    bias: bool = False,
    file_format: str = "bin",
    store_path: str = "generated_data/gemm",
    layer_name: str = "gemm",
) -> None:
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    x = torch.randn(input_shape)
    write_input_binary = x.numpy()
    if file_format == "bin":
        write_input_binary.tofile(store_path + "/input.bin")
    elif file_format == "txt":
        np.savetxt(store_path + "/input.txt", write_input_binary.flatten(), fmt="%.8f")
    else:
        raise Exception("Format not supported")

    gemm = torch.nn.Linear(input_shape[1], output_shape[1], bias=bias)
    weights = gemm.weight.detach().numpy()
    print(
        f"weights shape: {weights.shape}. Size in KB: {(weights.size * 2) / 1024:.4f}"
    )
    weights = transform_weights_fc(weights, coarse_in, coarse_out)

    if file_format == "bin":
        weights.tofile(store_path + "/weights.bin")
    elif file_format == "txt":
        np.savetxt(
            store_path + "/weights.txt", weights.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")
    # with open(
    #     f"{store_path}/weights_{layer_name}_cin{coarse_in}_cout{coarse_out}.csv",
    #     "w",
    # ) as f:
    #     f.write(array_init(weights[0]))

    if bias:
        bias = gemm.bias.detach().numpy()
    out = gemm(x)

    write_out_binary = out.detach().numpy()
    if file_format == "bin":
        write_out_binary.tofile(store_path + "/output.bin")
    elif file_format == "txt":
        np.savetxt(store_path + "/output.txt", write_out_binary.flatten(), fmt="%.8f")
    else:
        raise Exception("Format not supported")


def elemwise_3d(
    input_shape,
    input_shape_2,
    coarse_in,
    elemwise_op_type,
    file_format,
    store_path="generated_data/elemwise_3d",
):
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    x = torch.randn(input_shape)
    # print(x.numpy())
    print(x.numpy().shape)
    y = torch.randn(input_shape_2)
    # print(y.numpy())
    print(y.numpy().shape)

    write_input_binary_1 = x.numpy().transpose(0, 3, 4, 2, 1)
    write_input_binary_2 = y.numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_input_binary_1.tofile(store_path + "/input1.bin")
        host_data = create_host_binary(write_input_binary_1, 8, 8, coarse_in, 16*coarse_in)
        with open(store_path + "/ei1.bin", "wb") as f:
            f.write(host_data)
        write_input_binary_2.tofile(store_path + "/input2.bin")
        host_data = create_host_binary(write_input_binary_2, 8, 8, coarse_in, 16*coarse_in)
        with open(store_path + "/ei2.bin", "wb") as f:
            f.write(host_data)
    elif file_format == "txt":
        np.savetxt(
            store_path + "/input1.txt",
            write_input_binary_1.flatten(),
            fmt="%.8f",
        )
        np.savetxt(
            store_path + "/input2.txt",
            write_input_binary_2.flatten(),
            fmt="%.8f",
        )
    else:
        raise Exception("Format not supported")

    if elemwise_op_type == "Add":
        out = torch.add(x, y)
    elif elemwise_op_type == "Mul":
        out = torch.mul(x, y)
    # print(out.detach().numpy())
    print(out.detach().numpy().shape)

    write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_out_binary.tofile(store_path + "/output.bin")
        host_data = create_host_binary(write_out_binary, 8, 8, coarse_in, 16*coarse_in)
        with open(store_path + "/eo.bin", "wb") as f:
            f.write(host_data)
    elif file_format == "txt":
        np.savetxt(
            store_path + "/output.txt", write_out_binary.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")


def shish_3d(input_shape, coarse_in, file_format, store_path="generated_data/shish_3d"):
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    x = torch.randn(input_shape)
    # print(x.numpy())
    print(x.numpy().shape)
    write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_input_binary.tofile(store_path + "/input.bin")
    elif file_format == "txt":
        np.savetxt(
            store_path + "/input.txt", write_input_binary.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")

    sigmoid = torch.nn.Sigmoid()
    out = sigmoid(x)
    out = out * x

    # print(out.detach().numpy())
    print(out.detach().numpy().shape)

    write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_out_binary.tofile(store_path + "/output.bin")
    elif file_format == "txt":
        np.savetxt(
            store_path + "/output.txt", write_out_binary.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")


def sigmoid_3d(input_shape, coarse_in, file_format, store_path="generated_data/sigmoid_3d"):
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    x = torch.randn(input_shape)
    # print(x.numpy())
    print(x.numpy().shape)
    write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_input_binary.tofile(store_path + "/input.bin")
        host_data = create_host_binary(write_input_binary, 8, 8, coarse_in, 16*coarse_in)
        with open(store_path + "/si.bin", "wb") as f:
            f.write(host_data)
    elif file_format == "txt":
        np.savetxt(
            store_path + "/input.txt", write_input_binary.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")

    sigmoid = torch.nn.Sigmoid()
    out = sigmoid(torch.permute(x, (0, 3, 4, 2, 1)))

    # print(out.detach().numpy())
    print(out.detach().numpy().shape)

    write_out_binary = out.detach().numpy()  # .transpose(1, 0, 2, 3, 4)
    if file_format == "bin":
        write_out_binary.tofile(store_path + "/output.bin")
        host_data = create_host_binary(write_out_binary, 8, 8, coarse_in, 16*coarse_in)
        with open(store_path + "/so.bin", "wb") as f:
            f.write(host_data)
    elif file_format == "txt":
        np.savetxt(
            store_path + "/output.txt", write_out_binary.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")


def relu_3d(input_shape, coarse_in, file_format, store_path="generated_data/relu_3d"):
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    x = torch.randn(input_shape)
    # print(x.numpy())
    print(x.numpy().shape)
    write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_input_binary.tofile(store_path + "/input.bin")
    elif file_format == "txt":
        np.savetxt(
            store_path + "/input.txt", write_input_binary.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")

    relu = torch.nn.ReLU()
    out = relu(torch.permute(x, (0, 3, 4, 2, 1)))

    # print(out.detach().numpy())
    print(out.detach().numpy().shape)

    write_out_binary = out.detach().numpy()  # .transpose(1, 0, 2, 3, 4)
    if file_format == "bin":
        write_out_binary.tofile(store_path + "/output.bin")
    elif file_format == "txt":
        np.savetxt(
            store_path + "/output.txt", write_out_binary.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")

def partition_3d(part, partition_structure, onnx_parser, file_format, store_path="generated_data/partition_3d"):
    class X3d_m_layer(nn.Module):
        def __init__(self, layer_type, conv_config, file_path, part):
            super().__init__()

            self.layer_type = layer_type
            self.conv_config = conv_config
            self.conv_layers = [*conv_config]
            self.file_path = file_path
            self.part = part

            if self.layer_type == "type_1":
                self.relu1 = nn.ReLU()
                self.conv1 = self.create_conv_layer(0)
                self.weights_1 = self.conv1.weight
                self.relu2 = nn.ReLU()
                self.conv2 = self.create_conv_layer(1)
                self.weights_2 = self.conv2.weight
                self.gap1 = nn.AdaptiveAvgPool3d(1)
                self.conv3 = self.create_conv_layer(2)
                self.weights_3 = self.conv3.weight
                self.relu3 = nn.ReLU()
                self.conv4 = self.create_conv_layer(3)
                self.weights_4 = self.conv4.weight
                self.sigmoid1 = nn.Sigmoid()
                self.conv5 = self.create_conv_layer(4)
                self.weights_5 = self.conv5.weight
                self.conv6 = self.create_conv_layer(5)
                self.weights_6 = self.conv6.weight
            elif self.layer_type == "type_2":
                self.relu1 = nn.ReLU()
                self.conv1 = self.create_conv_layer(0)
                self.weights_1 = self.conv1.weight
                self.relu2 = nn.ReLU()
                self.conv2 = self.create_conv_layer(1)
                self.weights_2 = self.conv2.weight
                self.gap1 = nn.AdaptiveAvgPool3d(1)
                self.conv3 = self.create_conv_layer(2)
                self.weights_3 = self.conv3.weight
                self.relu3 = nn.ReLU()
                self.conv4 = self.create_conv_layer(3)
                self.weights_4 = self.conv4.weight
                self.sigmoid1 = nn.Sigmoid()
                self.conv5 = self.create_conv_layer(4)
                self.weights_5 = self.conv5.weight
            elif self.layer_type == "type_3":
                self.relu1 = nn.ReLU()
                self.conv1 = self.create_conv_layer(0)
                self.weights_1 = self.conv1.weight
                self.relu2 = nn.ReLU()
                self.conv2 = self.create_conv_layer(1)
                self.weights_2 = self.conv2.weight
                self.conv3 = self.create_conv_layer(2)
                self.weights_3 = self.conv3.weight
            elif self.layer_type == "type_seq":
                self.relu1 = nn.ReLU()
                self.conv1 = self.create_conv_layer(0)
                self.weights_1 = self.conv1.weight
                self.relu2 = nn.ReLU()
                self.conv2 = self.create_conv_layer(1)
                self.weights_2 = self.conv2.weight
                self.gap1 = nn.AdaptiveAvgPool3d(1)
                self.conv3 = self.create_conv_layer(2)
                self.weights_3 = self.conv3.weight
                self.relu3 = nn.ReLU()
                self.conv4 = self.create_conv_layer(3)
                self.weights_4 = self.conv4.weight
                self.sigmoid1 = nn.Sigmoid()
                self.conv5 = self.create_conv_layer(4)
                self.weights_5 = self.conv5.weight
            elif self.layer_type == "type_first":
                raise Exception(f"Layer type {self.layer_type} is not implemented yet")
            elif self.layer_type == "type_last":
                raise Exception(f"Layer type {self.layer_type} is not implemented yet")
            else:
                raise Exception(f"Layer type {self.layer_type} is not supported")

        def create_conv_layer(self, conv_idx):
            return nn.Conv3d(
                in_channels=self.conv_config[self.conv_layers[conv_idx]]["channels"],
                out_channels=self.conv_config[self.conv_layers[conv_idx]]["filters"],
                kernel_size=self.conv_config[self.conv_layers[conv_idx]][
                    "kernel_shape"
                ],
                stride=self.conv_config[self.conv_layers[conv_idx]]["stride"],
                padding=self.conv_config[self.conv_layers[conv_idx]]["padding"],
                groups=self.conv_config[self.conv_layers[conv_idx]]["groups"],
                bias=False,
            )

        def swish(self, x):
            return x * torch.sigmoid(x)

        def forward(self, x):
            if self.layer_type == "type_1":
                x = self.relu1(x)
                relu1_out = x
                x = self.conv1(x)
                x = self.relu2(x)
                x = self.conv2(x)
                conv2_out = x
                x = self.gap1(x)
                x = self.conv3(x)
                x = self.relu3(x)
                x = self.conv4(x)
                x = self.sigmoid1(x)
                x = x * conv2_out
                x = self.swish(x)
                x = self.conv5(x)
                conv6_out = self.conv6(relu1_out)
                x = x + conv6_out
                return x
            elif self.layer_type == "type_2":
                x = self.relu1(x)
                relu1_out = x
                x = self.conv1(x)
                x = self.relu2(x)
                x = self.conv2(x)
                conv2_out = x
                x = self.gap1(x)
                x = self.conv3(x)
                x = self.relu3(x)
                x = self.conv4(x)
                x = self.sigmoid1(x)
                x = x * conv2_out
                x = self.swish(x)
                x = self.conv5(x)
                x = x + relu1_out
                return x
            elif self.layer_type == "type_3":
                x = self.relu1(x)
                relu1_out = x
                x = self.conv1(x)
                x = self.relu2(x)
                x = self.conv2(x)
                x = self.swish(x)
                x = self.conv3(x)
                x = x + relu1_out
                return x
            elif self.layer_type == "type_seq":
                x = self.relu1(x)
                x = self.conv1(x)
                x = self.relu2(x)
                x = self.conv2(x)
                conv2_out = x
                x = self.gap1(x)
                x = self.conv3(x)
                x = self.relu3(x)
                x = self.conv4(x)
                x = self.sigmoid1(x)
                x = x * conv2_out
                x = self.swish(x)
                x = self.conv5(x)
                return x
            elif self.layer_type == "type_first":
                raise Exception(f"Layer type {self.layer_type} is not implemented yet")
            elif self.layer_type == "type_last":
                raise Exception(f"Layer type {self.layer_type} is not implemented yet")
            else:
                raise Exception(f"Layer type {self.layer_type} is not supported")

        def transform_store_weights(self, weights, idx):
            node = self.conv_layers[idx]
            depthwise = self.conv_config[node]["depthwise"]
            coarse_in = self.conv_config[node]["coarse_in"]
            coarse_out = self.conv_config[node]["coarse_out"]
            groups = self.conv_config[node]["groups"]
            if not depthwise:
                print(
                    "weights_{}_cin{}_cout{}:".format(node, coarse_in, coarse_out),
                    weights.detach().numpy().shape,
                )
                weights_transformed = transform_weights(
                    weights.detach().numpy(), coarse_in, coarse_out, 1, 1, groups=groups
                )
                with open(
                    os.path.join(self.file_path, prefix, self.part)
                    + "/weights_{}_cin{}_cout{}.csv".format(
                        node, coarse_in, coarse_out
                    ),
                    "w",
                ) as f:
                    f.write(array_init(weights_transformed[0]))
            else:
                print(
                    "weights_{}_cin{}_cout{}:".format(node, coarse_in, coarse_out),
                    weights.detach().numpy().shape,
                )
                weights_transformed = transform_weights(
                    weights.detach().numpy(),
                    1,
                    1,
                    1,
                    coarse_group=coarse_in,
                    groups=groups,
                )
                with open(
                    os.path.join(self.file_path, prefix, self.part)
                    + "/weights_{}_cin{}_cout{}.csv".format(
                        node, coarse_in, coarse_out
                    ),
                    "w",
                ) as f:
                    f.write(array_init(weights_transformed[0]))

        def save_weights(self):
            if self.layer_type == "type_1":
                self.transform_store_weights(self.weights_1, 0)
                self.transform_store_weights(self.weights_2, 1)
                self.transform_store_weights(self.weights_3, 2)
                self.transform_store_weights(self.weights_4, 3)
                self.transform_store_weights(self.weights_5, 4)
                self.transform_store_weights(self.weights_6, 5)
            elif self.layer_type == "type_2":
                self.transform_store_weights(self.weights_1, 0)
                self.transform_store_weights(self.weights_2, 1)
                self.transform_store_weights(self.weights_3, 2)
                self.transform_store_weights(self.weights_4, 3)
                self.transform_store_weights(self.weights_5, 4)
            elif self.layer_type == "type_3":
                self.transform_store_weights(self.weights_1, 0)
                self.transform_store_weights(self.weights_2, 1)
                self.transform_store_weights(self.weights_3, 2)
            elif self.layer_type == "type_seq":
                self.transform_store_weights(self.weights_1, 0)
                self.transform_store_weights(self.weights_2, 1)
                self.transform_store_weights(self.weights_3, 2)
                self.transform_store_weights(self.weights_4, 3)
                self.transform_store_weights(self.weights_5, 4)
            elif self.layer_type == "type_first":
                raise Exception(f"Layer type {self.layer_type} is not implemented yet")
            elif self.layer_type == "type_last":
                raise Exception(f"Layer type {self.layer_type} is not implemented yet")
            else:
                raise Exception(f"Layer type {self.layer_type} is not supported")

    mem_input_nodes = partition_structure['input_nodes']
    mem_output_nodes = partition_structure['output_nodes']
    input_nodes = []
    for node in mem_input_nodes:
        assert len(partition_structure['layers'][node]['out_nodes']) == 1, "Memory input node should have only one output node"
        input_nodes.append(partition_structure['layers'][node]['out_nodes'][0])
    output_nodes = []
    for node in mem_output_nodes:
        in_nodes = partition_structure['layers'][node]['in_nodes']
        assert len(in_nodes) == 1, "Memory output node should have only one input node"
        if partition_structure['layers'][in_nodes[0]]['type'] == "Split":
            output_nodes.append(partition_structure['layers'][in_nodes[0]]['ref_layer'])
        elif partition_structure['layers'][in_nodes[0]]['type'] == "Squeeze":
            output_nodes.append(partition_structure['layers'][in_nodes[0]]['ref_layer_out'])
        else:
            output_nodes.append(in_nodes[0])

    inter_input_nodes = []
    for i_n in input_nodes:
        if i_n not in onnx_parser.initial_model_inputs:
            prev_nodes = onnx_parser.get_prev_nodes_from_node(i_n)
            for p_n in prev_nodes:
                if p_n not in onnx_parser.initial_model_inputs and p_n.name not in partition_structure['layers']:
                        inter_input_nodes.append(p_n.name)

    updated_output_nodes = deepcopy(output_nodes)
    for i_n in inter_input_nodes:
        if i_n not in output_nodes:
            updated_output_nodes.append(i_n)

    onnx_parser.add_outputs_to_model(updated_output_nodes)

    in_shapes = onnx_parser.get_model_input_shapes()

    assert len(in_shapes) == len(onnx_parser.initial_model_inputs), "Number of input nodes and input shapes should be same"
    in_data = {}
    for name, shape in zip(onnx_parser.initial_model_inputs, in_shapes):
        in_data[name] = np.random.random_sample(shape).astype(np.float32)
    out, out_names = onnx_parser.onnx_forward(in_data)

    #TODO: Check the case of original model input is not in the intermedidate input nodes
    int_int_data = {}
    for o, o_n in zip(out, out_names):
        if o_n in inter_input_nodes:
            int_int_data[o_n] = o

    out_data = {}
    for o, o_n in zip(out, out_names):
        if o_n in output_nodes:
            out_data[o_n] = o

    #TODO: get the weights from the conv and gemm layers
    return
    inputs = []
    for in_node in input_nodes:
        input_shape = [layers_config[in_node]["batch_size"],
                       layers_config[in_node]["channels_in"],
                       layers_config[in_node]["depth_in"],
                       layers_config[in_node]["height_in"],
                       layers_config[in_node]["width_in"]]
        inputs.append(torch.randn(input_shape))

    in_out_branch_dict = {}
    for (i, o) in get_branch_start_end_points(graph):
        in_out_branch_dict[i] = o

    parse_graph(graph, input_nodes[0], in_out_branch_dict, deque(), input_nodes, output_nodes)

    layers = [*layers_config]
    for layer, config in layers_config.items():
        print(layer, config)
        continue

        layer_order = [l.split("_")[0] for l in layer_nodes]
        layer_type = get_x3d_m_layer_type(layer_order)

        input_shape = layers[layer_nodes[0]]["shape_in"]

        conv_config = {}
        for p in layers:
            if "Conv" in p:
                conv_config[p.lower()] = {
                    "channels": layers[p]["shape_in"][1],
                    "filters": layers[p]["shape_out"][1],
                    "kernel_shape": layers[p]["shape_kernel"],
                    "padding": layers[p]["padding"],
                    "stride": layers[p]["stride"],
                    "groups": layers[p]["groups"],
                    "depthwise": layers[p]["depthwise"],
                    "coarse_in": layers[p]["coarse_in_factor"],
                    "coarse_out": layers[p]["coarse_out_factor"],
                }

        # file_path = "/".join(__file__.split("/")[:-1])
        file_path = os.getcwd()
        if not os.path.exists(os.path.join(file_path, prefix, part)):
            os.makedirs(os.path.join(file_path, prefix, part))

        x = torch.randn(input_shape)

        print("*" * 30)
        print("input:", x.numpy().shape)
        print("*" * 30)
        write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
        if file_format == "bin":
            write_input_binary.tofile(
                os.path.join(file_path, prefix, part) + "/input.bin"
            )
        elif file_format == "txt":
            np.savetxt(
                os.path.join(file_path, prefix, part) + "/input.txt",
                write_input_binary.flatten(),
                fmt="%.8f",
            )
        else:
            raise Exception("Format not supported")

        partition_model = X3d_m_layer(layer_type, conv_config, file_path, part)
        generate_onnx(partition_model, x, f"models/{part}.onnx")
        partition_model.save_weights()

        out = partition_model(x)

        print("*" * 30)
        print("output:", out.detach().numpy().shape)
        print("*" * 30)
        write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
        if file_format == "bin":
            write_out_binary.tofile(
                os.path.join(file_path, prefix, part) + "/output.bin"
            )
        elif file_format == "txt":
            np.savetxt(
                os.path.join(file_path, prefix, part) + "/output.txt",
                write_out_binary.flatten(),
                fmt="%.8f",
            )
        else:
            raise Exception("Format not supported")

def pool_3d(input_shape,
            kernel_shape,
            padding,
            stride,
            coarse_in,
            pool_op_type,
            file_format,
            store_path="generated_data/pool_3d"):
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    x = torch.randn(input_shape)
    print(x.numpy().shape)
    write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_input_binary.tofile(store_path + "/input.bin")
    elif file_format == "txt":
        np.savetxt(
            store_path + "/input.txt", write_input_binary.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")

    if pool_op_type == "max":
        pool = nn.MaxPool3d(kernel_shape, stride=stride, padding=padding)
    elif pool_op_type == "avg":
        pool = nn.AvgPool3d(kernel_shape, stride=stride, padding=padding)
    else:
        raise Exception("Pool op type not supported")
    out = pool(x)

    print(out.detach().numpy().shape)

    write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_out_binary.tofile(store_path + "/output.bin")
    elif file_format == "txt":
        np.savetxt(
            store_path + "/output.txt", write_out_binary.flatten(), fmt="%.8f"
        )
    else:
        raise Exception("Format not supported")

def conv_3d(
    input_shape,
    kernel_shape,
    bias,
    filters,
    padding,
    stride,
    groups,
    depthwise,
    coarse_in,
    coarse_out,
    file_format,
    store_path="generated_data/conv_3d",
    layer_name="conv_3d",
):
    def get_sliding_window_output(data, kernel, padding, stride):
        batch_size, height, width, depth, channels = data.shape
        kernel_depth, kernel_height, kernel_width = kernel
        pad_depth, pad_height, pad_width = padding
        stride_depth, stride_height, stride_width = stride

        data_padded = np.ndarray(
            (
                batch_size,
                height + 2 * pad_height,
                width + 2 * pad_width,
                depth + 2 * pad_depth,
                channels,
            ),
            dtype=np.float32,
        )

        for index, _ in np.ndenumerate(data_padded):
            if index[1] < pad_height:
                data_padded[index] = 0
            elif index[2] < pad_width:
                data_padded[index] = 0
            elif index[3] < pad_depth:
                data_padded[index] = 0
            elif index[1] > height - 1 + pad_height:
                data_padded[index] = 0
            elif index[2] > width - 1 + pad_width:
                data_padded[index] = 0
            elif index[3] > depth - 1 + pad_depth:
                data_padded[index] = 0
            else:
                data_padded[index] = data[
                    index[0],
                    index[1] - pad_height,
                    index[2] - pad_width,
                    index[3] - pad_depth,
                    index[4],
                ]

        height_out = math.floor(
            (height + 2 * pad_height - kernel_height) / stride_height + 1
        )
        width_out = math.floor(
            (width + 2 * pad_width - kernel_width) / stride_width + 1
        )
        depth_out = math.floor(
            (depth + 2 * pad_depth - kernel_depth) / stride_depth + 1
        )
        out = np.ndarray(
            (
                batch_size,
                height_out,
                width_out,
                depth_out,
                channels,
                kernel_height,
                kernel_width,
                kernel_depth,
            ),
            dtype=np.float32
        )

        for index, _ in np.ndenumerate(out):
            out[index] = data_padded[
                index[0],
                index[1] * stride_height + index[5],
                index[2] * stride_width + index[6],
                index[3] * stride_depth + index[7],
                index[4],
            ]

        return out

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    x = torch.randn(input_shape)
    print(f"input shape: {x.numpy().shape}")
    # sliding_window_out = get_sliding_window_output(
    #     x.numpy().transpose(0, 3, 4, 2, 1), kernel_shape, padding, stride
    # )

    write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_input_binary.tofile(store_path + "/input.bin")
    elif file_format == "txt":
        np.savetxt(
            store_path + "/input.txt",
            write_input_binary.flatten(),
            fmt="%.8f",
        )
    else:
        raise Exception("Format not supported")

    conv = torch.nn.Conv3d(
        x.shape[1],
        filters,
        kernel_shape,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
    )
    weights = conv.weight
    print(f"weights shape: {weights.detach().numpy().shape}")
    # print(weights.detach().numpy().transpose(0, 1, 3, 4, 2))
    out = conv(x)

    write_weights_binary = weights.detach().numpy()  # .transpose(1, 0, 2, 3, 4)
    # if file_format == "bin":
    # 	write_weights_binary.tofile(store_path + "/weights.bin")
    # elif file_format == "txt":
    # 	np.savetxt(store_path + "/weights.txt", write_weights_binary.flatten(), fmt='%.8f')
    # else:
    # 	raise Exception("Format not supported")

    if depthwise:
        weights_transformed = transform_weights(
            weights_raw=write_weights_binary,
            coarse_in=1,
            coarse_out=1,
            wr_factor=1,
            coarse_group=coarse_in,
            groups=groups,
        )
    else:
        weights_transformed = transform_weights(
            weights_raw=write_weights_binary,
            coarse_in=coarse_in,
            coarse_out=coarse_out,
            wr_factor=1,
            coarse_group=1,
            groups=groups,
        )
    with open(
        f"{store_path}/weights_{layer_name}_cin{coarse_in}_cout{coarse_out}.csv",
        "w",
    ) as f:
        f.write(array_init(weights_transformed[0]))

    print("=" * 40)
    print("=" * 40)
    # print(out.detach().numpy())
    print(f"output shape: {out.detach().numpy().shape}")

    write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
    if file_format == "bin":
        write_out_binary.tofile(store_path + "/output.bin")
    elif file_format == "txt":
        np.savetxt(
            store_path + "/output.txt",
            write_out_binary.flatten(),
            fmt="%.8f",
        )
    else:
        raise Exception("Format not supported")

    return
    stream = create_queue(x.numpy().transpose(0, 3, 4, 2, 1))

    batch = x.shape[0]
    channel = x.shape[1]
    depth = x.shape[2]
    height = x.shape[3]
    width = x.shape[4]

    kd, kh, kw = kernel_shape
    pad_d, pad_h, pad_w = padding
    stride_d, stride_h, stride_w = stride

    # window_buffer = [kh][kw][kd-1]
    window_buffer = []
    for i in range(kh):
        width_buffers = []
        for j in range(kw):
            depth_buffers = []
            for k in range(kd - 1):
                depth_buffers.append(deque([]))
            width_buffers.append(depth_buffers)
        window_buffer.append(width_buffers)

    # line_buffer = [kh][kw-1]
    line_buffer = []
    for i in range(kh):
        width_buffers = []
        for j in range(kw - 1):
            width_buffers.append(deque([]))
        line_buffer.append(width_buffers)

    # extra_buffer = [kh-1]
    extra_buffer = []
    for i in range(kh - 1):
        extra_buffer.append(deque([]))

    # window_buffer = [kh][kw][kd]
    out_buffer = []
    for i in range(kh):
        width_buffers = []
        for j in range(kw):
            depth_buffers = []
            for k in range(kd):
                depth_buffers.append(deque([]))
            width_buffers.append(depth_buffers)
        out_buffer.append(width_buffers)

    frame_cache = np.zeros(shape=(kh, kw, kd))

    count = 0
    for b in range(0, batch):
        for h in range(0, height + 2 * pad_h):
            for w in range(0, width + 2 * pad_w):
                for d in range(0, depth + 2 * pad_d):
                    for c in range(0, channel):
                        count += 1
                        if h < pad_h:
                            tmp = 0
                        elif h > height + pad_h - 1:
                            tmp = 0
                        elif w < pad_w:
                            tmp = 0
                        elif w > width + pad_w - 1:
                            tmp = 0
                        elif d < pad_d:
                            tmp = 0
                        elif d > depth + pad_d - 1:
                            tmp = 0
                        else:
                            tmp = stream.pop()

                        print(
                            f"b={b}  |  row={h}  |  col={w}  |  d={d}  |  c={c}  |  => tmp={tmp}"
                        )

                        if h < kh - 1:
                            if w < kw - 1:
                                if d < kd - 1:
                                    window_buffer[h][w][d].appendleft(tmp)
                                    frame_cache[h][w][d] = tmp
                                    if w > 0:
                                        line_buffer[h][w - 1].appendleft(tmp)
                                else:
                                    line_buffer[h][w].appendleft(tmp)
                            elif (w == kw - 1 or w == kw) and d < kd - 1:
                                if w == kw - 1:
                                    window_buffer[h][w][d].appendleft(tmp)
                                    line_buffer[h][w - 1].appendleft(tmp)
                                elif w == kw:
                                    extra_buffer[h].appendleft(tmp)
                            elif w == kw - 1 and d >= kd - 1:
                                extra_buffer[h].appendleft(tmp)
                            else:
                                extra_buffer[h].appendleft(tmp)
                        elif h == kh - 1 and w < kw - 1:
                            if d < kd - 1:
                                window_buffer[h][w][d].appendleft(tmp)
                                frame_cache[h][w][d] = tmp
                                if w > 0:
                                    line_buffer[h][w - 1].appendleft(tmp)
                            else:
                                line_buffer[h][w].appendleft(tmp)
                        elif h == kh - 1 and w == kw - 1 and d < kd - 1:
                            window_buffer[h][w][d].appendleft(tmp)
                            line_buffer[h][w - 1].appendleft(tmp)
                        else:

                            for i in range(0, kh):
                                for j in range(0, kw):
                                    for k in range(0, kd - 1):
                                        curr = window_buffer[i][j][k].pop()
                                        frame_cache[i][j][k] = curr

                            for i in range(0, kh):
                                for j in range(0, kw - 1):
                                    curr = line_buffer[i][j].pop()
                                    frame_cache[i][j][-1] = curr

                            for i in range(0, kh - 1):
                                curr = extra_buffer[i].pop()
                                frame_cache[i][-1][-1] = curr

                            frame_cache[-1][-1][-1] = tmp

                            print(
                                f"WINDOW_BUFFER\t(POP):\t\tWindow buffer depth: {len(window_buffer[0][0][0])}"
                            )
                            if not (
                                h == height + 2 * pad_h - 1
                                and w == width + 2 * pad_w - 1
                                and d == depth + 2 * pad_d - 1
                            ):
                                for i in range(0, kh):
                                    for j in range(0, kw):
                                        for k in range(0, kd - 1):
                                            curr = frame_cache[i][j][k + 1]
                                            window_buffer[i][j][k].appendleft(curr)
                                print(
                                    f"WINDOW_BUFFER\t(APPEND):\tWindow buffer depth: {len(window_buffer[0][0][0])}"
                                )
                            else:
                                print("WINDOW_BUFFER:\t\t\tSkip appending...")

                            print(
                                f"LINE_BUFFER\t(POP):\t\tLine buffer depth: {len(line_buffer[0][0])}"
                            )
                            if not (
                                h == height + 2 * pad_h - 1
                                and w == width + 2 * pad_w - 1
                            ):
                                for i in range(0, kh):
                                    for j in range(0, kw - 1):
                                        curr = frame_cache[i][j + 1][kd - 1]
                                        line_buffer[i][j].appendleft(curr)
                                print(
                                    f"LINE_BUFFER\t(APPEND):\tLine buffer depth: {len(line_buffer[0][0])}"
                                )
                            else:
                                print("LINE_BUFFER:\t\t\tSkip appending...")

                            print(
                                f"EXTRA_BUFFER\t(POP):\t\tExtra buffer depth: {len(extra_buffer[0])}"
                            )
                            if not (
                                (
                                    h == height + 2 * pad_h - 1
                                    and w == kw - 1
                                    and d >= kd - 1
                                )
                                or (h == height + 2 * pad_h - 1 and w > kw - 1)
                            ):
                                for i in range(0, kh - 1):
                                    curr = frame_cache[i + 1][0][0]
                                    extra_buffer[i].appendleft(curr)
                                print(
                                    f"EXTRA_BUFFER\t(APPEND):\tExtra buffer depth: {len(extra_buffer[0])}"
                                )
                            else:
                                print("EXTRA_BUFFER:\t\t\tSkip appending...")

                            for i in range(kh):
                                for j in range(kw):
                                    for k in range(kd):
                                        curr = frame_cache[i][j][k]
                                        out_buffer[i][j][k].appendleft(curr)
                            print(count)
                            return
                            # print(frame_cache.transpose((2, 0, 1)))

                        print(
                            f"Window buffer: {len(window_buffer[0][0][0])}, Line buffer: {len(line_buffer[0][0])}, Extra buffer: {len(extra_buffer[0])}"
                        )
                        # show_wb(window_buffer)
                        # show_lb(line_buffer, extra_buffer)
                        print("=" * 40)

    for i in range(kh):
        for j in range(kw):
            for k in range(kd - 1):
                assert (
                    len(window_buffer[i][j][k]) == 0
                ), "There are still %d values in window_buffer [%d][%d][%d]" % (
                    len(window_buffer[i][j][k]),
                    i,
                    j,
                    k,
                )

    for i in range(kh):
        for j in range(kw - 1):
            assert (
                len(line_buffer[i][j]) == 0
            ), "There are still %d values in line_buffer [%d][%d]" % (
                len(line_buffer[i][j]),
                i,
                j,
            )

    for i in range(kh - 1):
        assert (
            len(extra_buffer[i]) == 0
        ), "There are still %d values in line_buffer [%d]" % (len(extra_buffer[i]), i)

    showres(
        out_buffer,
        input_shape,
        file_format,
        kh,
        kw,
        kd,
        pad_h,
        pad_w,
        pad_d,
        stride_h,
        stride_w,
        stride_d,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Data generator for 3D operations")
    parser.add_argument(
        "op_type",
        type=str,
        choices=[
            "3d_conv",
            "3d_pool",
            "3d_relu",
            "3d_swish",
            "3d_sigmoid",
            "3d_elemwise",
            "3d_gap",
            "3d_part",
            "gemm",
        ],
    )
    parser.add_argument(
        "--input_shape", nargs="+", default=[1, 24, 16, 32, 32], type=int
    )
    parser.add_argument(
        "--input_shape_2", nargs="+", default=[1, 24, 16, 32, 32], type=int
    )
    parser.add_argument("--kernel_shape", nargs="+", default=[3, 3, 3], type=int)
    parser.add_argument("--filters", default=1, type=int)
    parser.add_argument("--groups", default=1, type=int)
    parser.add_argument("--padding", nargs="+", default=[1, 1, 1], type=int)
    parser.add_argument("--stride", nargs="+", default=[1, 1, 1], type=int)
    parser.add_argument("--depthwise", default=False, action="store_true")
    parser.add_argument("--bias", default=False, action="store_true")
    parser.add_argument("--in_features", nargs="+", default=[1, 200], type=int)
    parser.add_argument("--out_features", nargs="+", default=[1, 400], type=int)
    parser.add_argument("--coarse_in", default=1, type=int)
    parser.add_argument("--coarse_out", default=1, type=int)
    parser.add_argument(
        "--elemwise_op_type", choices=["add", "mul"], default="add", type=str
    )
    parser.add_argument(
        "--pool_op_type", choices=["max", "avg"], default="max", type=str
    )
    parser.add_argument("--partition_name", default="partition_3d", type=str)
    parser.add_argument("--model_name", default="", type=str)
    parser.add_argument("--format", choices=["txt", "bin"], default="bin", type=str)
    parser.add_argument("--config_file", default="", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    op_type = args.op_type

    if op_type == "3d_conv":
        conv_3d(
            args.input_shape,
            args.kernel_shape,
            args.bias,
            args.filters,
            args.padding,
            args.stride,
            args.groups,
            args.depthwise,
            args.coarse_in,
            args.coarse_out,
            args.format,
        )
    elif op_type == "3d_pool":
        pool_3d(args.input_shape,
                args.kernel_shape,
                args.padding,
                args.stride,
                args.coarse_in,
                args.pool_op_type,
                args.format)
    elif op_type == "3d_relu":
        relu_3d(args.input_shape, args.coarse_in, args.format)
    elif op_type == "3d_swish":
        shish_3d(args.input_shape, args.coarse_in, args.format)
    elif op_type == "3d_sigmoid":
        sigmoid_3d(args.input_shape, args.coarse_in, args.format)
    elif op_type == "3d_elemwise":
        elemwise_3d(
            args.input_shape,
            args.input_shape_2,
            args.coarse_in,
            args.elemwise_op_type,
            args.format,
        )
    elif op_type == "3d_gap":
        gap_3d(args.input_shape, args.coarse_in, args.coarse_out, args.format)
    elif op_type == "3d_part":
        #TODO: generate a dictionary from the args.config_file
        config_dict = {}
        partition_3d(args.partition_name, config_dict, OnnxModelParser(args.model_name), file_format=args.format)
    elif op_type == "gemm":
        gemm(args.in_features, args.out_features, args.coarse_in, args.coarse_out, bias=args.bias, file_format=args.format)
    else:
        print("Invalid op_type: %s" % op_type)
