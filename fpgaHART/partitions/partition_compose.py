from ..onnx_parser.partition_descriptor import PartitionDescriptor
from .layer_compose import layer_compose
from ..utils import utils
import os
import csv

class PartitionComposer():
    def __init__(self, model_name, optimization, singlethreaded, per_layer_plot):
        self.se_full = False
        self.model_name = model_name
        self.optimization = optimization
        self.singlethreaded = singlethreaded
        self.per_layer_plot = per_layer_plot
        self.model_descriptor = PartitionDescriptor(model_name)

        if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports')):
            os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports'))
        self.layer_model_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + '.csv')

        with open(self.layer_model_file, mode='w') as layer_dp:
            csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Layer", "Latency(C)", "Latency(S)", "GOP/s", "volumes/s", "DSP(%)", "BRAM(%)", "RateIn1", "RateIn2", "RateOut", "Depth", "Muls", "Adds", "Mem(W)", "Mem(KB)", "MemBoundIn1", "MemBoundIn2", "MemBoundOut", "config"])

    def model_partition(self, partition):
        #TODO: This will NOT call the model layer but will read instead from the csv file all the valid configurations for each layer and find all the combinations for each partition
        for layer in partition:
            if self.se_full:
                self.model_layer(layer, self.model_descriptor.layers[layer])
            else:
                if self.model_descriptor.layers[layer]['operation'] == 'SqueezeExcitation':
                    for sub_layer, sub_description in self.model_descriptor.layers[layer]['primitive_ops'].items():
                        self.model_layer(layer + '_' + sub_layer, sub_description)
                else:
                    self.model_layer(layer, self.model_descriptor.layers[layer])

    def model_layer(self, layer, layer_description):
        print("Modeling {} layer...".format(layer))
        throughput_gops, throughput_vols, latency, dsp_util, bram_util = layer_compose(layer, layer_description, self.layer_model_file, self.optimization, self.singlethreaded)

        # if self.per_layer_plot and latency:
        #     calculate_pareto = True
        #     utils.plot_graph(throughput_gops, throughput_vols, latency, dsp_util, bram_util, layer, self.model_name, calculate_pareto)

    def parser(self):
        for partition in self.model_descriptor.partitions:
            self.model_partition(partition)

    def model_individual_layers(self):
        for name, descriptor in self.model_descriptor.layers.items():
            if self.se_full:
                self.model_layer(name, descriptor)
            else:
                if descriptor['operation'] == 'SqueezeExcitation':
                    for sub_layer, sub_description in descriptor['primitive_ops'].items():
                        self.model_layer(name + '_' + sub_layer, sub_description)
                else:
                    self.model_layer(name, descriptor)

        utils.drop_duplicates_csv(self.layer_model_file)
        if self.per_layer_plot:
            utils.plot_layers_csv(self.layer_model_file, self.model_name)

