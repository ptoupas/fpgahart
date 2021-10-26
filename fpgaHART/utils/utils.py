from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.batchnorm_3d import BatchNorm3DLayer
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..layers.gap import GAPLayer
from ..layers.elemwise import ElementWiseLayer
from ..layers.activation import ActivationLayer
import os
import csv
import math
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json

sns.set(rc={'figure.figsize':(15,8)})
sns.set_style("whitegrid")


def find_pareto(scores, domination_type='MaxMin'):
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
            if domination_type == 'MaxMin':
                if (scores[j][0] >= scores[i][0] and scores[j][1] <= scores[i][1]) and (scores[j][0] > scores[i][0] or scores[j][1] < scores[i][1]):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break
            elif domination_type == 'MinMin':
                if (scores[j][0] <= scores[i][0] and scores[j][1] <= scores[i][1]) and (scores[j][0] < scores[i][0] or scores[j][1] < scores[i][1]):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def plot_graph(throughput_ops, throughput_vols, latency, dsp_util, bram_util, layer_name, model_name, calculate_pareto, pareto_type='MaxMin'):
    throughput = throughput_vols
    dsps_dir = os.path.join(os.getcwd(), 'fpga_modeling_reports', 'graphs', model_name, 'throughput_dsps')
    if not os.path.exists(dsps_dir):
        os.makedirs(dsps_dir)

    if calculate_pareto:
        scores = np.zeros((len(throughput), 2))
        scores[:,0] = throughput
        scores[:,1] = dsp_util

        pareto = find_pareto(scores, pareto_type)
        pareto_front = scores[pareto]

        pareto_front_df = pd.DataFrame(pareto_front)
        pareto_front_df.sort_values(0, inplace=True)
        pareto_front = pareto_front_df.values

        sns.lineplot(x=pareto_front[:, 0], y=pareto_front[:, 1], color='red')

    sns.scatterplot(x=np.array(throughput), y=np.array(dsp_util), size=bram_util)

    plt.title(layer_name)
    plt.xlabel('Throughtput(outputs/sec)')
    plt.ylabel('DSPS %')
    if max(dsp_util) > 100:
        plt.yscale("log")
    else:
        plt.ylim([-5, max(100, max(dsp_util) + 0.1*max(dsp_util))])
    if max(throughput) > 100:
        plt.xscale("log")

    plt.legend(frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)

    file_name = layer_name + '.jpg'
    plt.savefig(os.path.join(dsps_dir, file_name))
    plt.clf()

def drop_duplicates_csv(file_name):
    layers_df = pd.read_csv(file_name)

    original_size = len(layers_df.index)
    columns = layers_df.columns.tolist()
    del(columns[-1])

    layers_df = layers_df.drop_duplicates(subset=columns, ignore_index=True)
    final_size = len(layers_df.index)
    print("Dropped {} rows due to duplicate".format(original_size - final_size))

    os.remove(file_name)
    layers_df.to_csv(file_name, index=False)

def plot_layers_csv(file_name, model_name, calculate_pareto=True, pareto_type='MaxMin', xaxis='volumes/s', yaxis='DSP(%)'):
    plot_dir = os.path.join(os.getcwd(), 'fpga_modeling_reports', 'graphs', model_name, 'throughput_dsps')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    layers_df = pd.read_csv(file_name)

    layers = layers_df['Layer'].unique()
    for l in layers:
        curr_df = layers_df.loc[layers_df['Layer'] == l]

        curr_df.plot.scatter(x=xaxis, y=yaxis)

        x_axis = curr_df[xaxis].to_numpy()
        y_axis = curr_df[yaxis].to_numpy()
        if calculate_pareto:
            scores = np.zeros((x_axis.shape[0], 2))
            scores[:,0] = x_axis
            scores[:,1] = y_axis

            pareto = find_pareto(scores, pareto_type)
            pareto_front = scores[pareto]

            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df.sort_values(0, inplace=True)
            pareto_front = pareto_front_df.values

            plt.plot(pareto_front[:, 0], pareto_front[:, 1], color='red')

        plt.ylim([-5, max(100, max(y_axis) + 0.1*max(y_axis))])
        plt.xscale('log')
        plt.title(l)
        file_name = l + '.jpg'
        plt.savefig(os.path.join(plot_dir, file_name))
        plt.clf()

def get_config_points(name, file_name):
    layers_df = pd.read_csv(file_name)

    curr_layer_df = layers_df.loc[layers_df['Layer'] == name].reset_index()

    return curr_layer_df['config'].apply(lambda x: json.loads(x)).to_list()

def get_paretto_csv(file_name_par, file_name, pareto_type='MinMin', xaxis='Latency(C)', yaxis='DSP(%)'):
    
    with open(file_name_par, mode='a') as pareto_results:
        csv_writer_par = csv.writer(pareto_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        layers_df = pd.read_csv(file_name)

        layers = layers_df['Layer'].unique()
        for l in layers:
            curr_df = layers_df.loc[layers_df['Layer'] == l].reset_index()
            if not ('Conv' in l.split('_') or 'Se' in l.split('_')):
                for ind in curr_df.index:
                    csv_writer_par.writerow(curr_df.iloc[ind].to_list()[1:])
            else:
                print('Calculating pareto front for layer {}'.format(l))

                x_axis = curr_df[xaxis].to_numpy()
                y_axis = curr_df[yaxis].to_numpy()

                scores = np.zeros((x_axis.shape[0], 2))
                scores[:,0] = x_axis
                scores[:,1] = y_axis

                pareto = find_pareto(scores, pareto_type)
                pareto_front = scores[pareto]

                pareto_front_df = pd.DataFrame(pareto_front)
                pareto_front_df.sort_values(0, inplace=True)
                pareto_front = pareto_front_df.values
                
                for p in pareto:
                    csv_writer_par.writerow(curr_df.iloc[p].to_list()[1:])

def check_configuration_validation(config, layers):
    valid = True

    layer_graph = {}
    for i, layer in enumerate(layers.values()):
        input_streams = []
        if isinstance(layer['layer'], Convolutional3DLayer):
            streams_in, streams_out = layer['layer'].get_num_streams()
            streams_in = math.ceil(streams_in * config[i][1])
            streams_out = math.ceil(streams_out * config[i][2])
            input_streams.append(streams_in)
        elif isinstance(layer['layer'], BatchNorm3DLayer):
            streams_in, streams_out = layer['layer'].get_num_streams()
            streams_in = math.ceil(streams_in * config[i][0])
            streams_out = math.ceil(streams_out * config[i][0])
            input_streams.append(streams_in)
        elif isinstance(layer['layer'], GAPLayer):
            streams_in, streams_out = layer['layer'].get_num_streams()
            streams_in = math.ceil(streams_in * config[i][0])
            streams_out = math.ceil(streams_out * config[i][1])
            input_streams.append(streams_in)
        elif isinstance(layer['layer'], ActivationLayer):
            streams_in, streams_out = layer['layer'].get_num_streams()
            streams_in = math.ceil(streams_in * config[i][0])
            streams_out = math.ceil(streams_out * config[i][0])
            input_streams.append(streams_in)
        elif isinstance(layer['layer'], SqueezeExcitationLayer):
            print("config for layer {} -> {}".format(name, comb[i]))
        elif isinstance(layer['layer'], ElementWiseLayer):
            streams_in1, streams_in2, streams_out = layer['layer'].get_num_streams()
            streams_in1 = math.ceil(streams_in1 * config[i][0])
            streams_in2 = math.ceil(streams_in2 * config[i][1])
            streams_out = math.ceil(streams_out * config[i][2])
            input_streams.append(streams_in1)
            input_streams.append(streams_in2)
        else:
            assert False, "Layer {} is not yet supported".format(name)
        
        if i not in layer_graph.keys():
            layer_graph[i] = {}
        layer_graph[i]['node_in'] = layer['node_in']
        layer_graph[i]['node_out'] = layer['node_out']
        layer_graph[i]['streams_in'] = input_streams
        layer_graph[i]['streams_out'] = streams_out

    input_node = layer_graph[list(layer_graph.keys())[0]]['node_in']
    output_node = layer_graph[list(layer_graph.keys())[-1]]['node_out']
    for n, v in layer_graph.items():
        for nd_in, strm_in in zip(v['node_in'], v['streams_in']):
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
        if v['node_out'] == node_out:
            return v['streams_out']
    assert False, "Cannot find node {} in the layer graph.".format(node_out)