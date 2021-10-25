from ..onnx_parser.partition_descriptor import PartitionDescriptor
from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.batchnorm_3d import BatchNorm3DLayer
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..layers.gap import GAPLayer
from ..layers.elemwise import ElementWiseLayer
from ..layers.activation import ActivationLayer
from .layer_compose import layer_compose
from ..utils import utils
import os
import csv
import itertools

class PartitionComposer():
    def __init__(self, model_name, optimization, singlethreaded, per_layer_plot):
        self.se_full = True
        self.model_name = model_name
        self.optimization = optimization
        self.singlethreaded = singlethreaded
        self.per_layer_plot = per_layer_plot
        self.model_descriptor = PartitionDescriptor(model_name)

        if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports')):
            os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports'))
        if self.se_full:
            self.layer_model_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + '.csv')
            self.layer_model_file_par = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + '_pareto.csv')
        else:
            self.layer_model_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + '_complete.csv')
            self.layer_model_file_par = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + '_complete_pareto.csv')

        with open(self.layer_model_file, mode='w') as layer_dp:
            csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Layer", "Latency(C)", "Latency(S)", "GOP/s", "volumes/s", "DSP(%)", "BRAM(%)", "RateIn1", "RateIn2", "RateOut", "Depth", "Muls", "Adds", "Mem(W)", "Mem(KB)", "MemBoundIn1", "MemBoundIn2", "MemBoundOut", "config"])
        with open(self.layer_model_file_par, mode='w') as layer_dp:
            csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Layer", "Latency(C)", "Latency(S)", "GOP/s", "volumes/s", "DSP(%)", "BRAM(%)", "RateIn1", "RateIn2", "RateOut", "Depth", "Muls", "Adds", "Mem(W)", "Mem(KB)", "MemBoundIn1", "MemBoundIn2", "MemBoundOut", "config"])

    def model_partition(self, partition):
        #TODO: This will NOT call the model layer but will read instead from the csv file all the valid configurations for each layer and find all the combinations for each partition
        for layer in partition:
            pass

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
                        self.model_layer(sub_layer, sub_description)
                else:
                    self.model_layer(name, descriptor)

        utils.drop_duplicates_csv(self.layer_model_file)
        utils.get_paretto_csv(self.layer_model_file_par, self.layer_model_file)
        if self.per_layer_plot:
            utils.plot_layers_csv(self.layer_model_file_par, self.model_name)

