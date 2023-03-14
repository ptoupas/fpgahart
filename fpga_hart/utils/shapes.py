import math
import networkx as nx
import numpy as np
import random

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
    for n in nx.topological_sort(graph):
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
    for n in nx.topological_sort(graph):
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
    for n in nx.topological_sort(graph):
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
    for n in nx.topological_sort(graph):
        bb = graph.nodes[n]["hw_type"]
        if bb == bb_type:
            max_depth = max(max_depth, graph.nodes[n]["hw"].input_shape[2])
            min_depth = min(min_depth, graph.nodes[n]["hw"].input_shape[2])
    return min_depth, max_depth


def get_minmax_height(graph: nx.DiGraph, bb_type: str) -> int:
    max_height = -1
    min_height = 10000
    for n in nx.topological_sort(graph):
        bb = graph.nodes[n]["hw_type"]
        if bb == bb_type:
            max_height = max(max_height, graph.nodes[n]["hw"].input_shape[3])
            min_height = min(min_height, graph.nodes[n]["hw"].input_shape[3])
    return min_height, max_height


def get_minmax_width(graph: nx.DiGraph, bb_type: str) -> int:
    max_width = -1
    min_width = 10000
    for n in nx.topological_sort(graph):
        bb = graph.nodes[n]["hw_type"]
        if bb == bb_type:
            max_width = max(max_width, graph.nodes[n]["hw"].input_shape[4])
            min_width = min(min_width, graph.nodes[n]["hw"].input_shape[4])
    return min_width, max_width