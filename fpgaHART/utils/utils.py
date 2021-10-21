import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set(rc={'figure.figsize':(15,8)})
sns.set_style("whitegrid")


def find_pareto(scores):
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
            if (scores[j][0] >= scores[i][0] and scores[j][1] <= scores[i][1]) and (scores[j][0] > scores[i][0] or scores[j][1] < scores[i][1]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def plot_graph(throughput_ops, throughput_vols, latency, dsp_util, bram_util, layer_name, model_name, calculate_pareto):
    throughput = throughput_vols
    dsps_dir = os.path.join(os.getcwd(), 'fpga_modeling_reports', 'graphs', model_name, 'throughput_dsps')
    if not os.path.exists(dsps_dir):
        os.makedirs(dsps_dir)

    if calculate_pareto:
        scores = np.zeros((len(throughput), 2))
        scores[:,0] = throughput
        scores[:,1] = dsp_util

        pareto = find_pareto(scores)
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