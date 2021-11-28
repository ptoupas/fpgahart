from ..onnx_parser.partition_descriptor import PartitionDescriptor
from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.batchnorm_3d import BatchNorm3DLayer
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..layers.gap import GAPLayer
from ..layers.elemwise import ElementWiseLayer
from ..layers.activation import ActivationLayer
from ..layers.fully_connected import FCLayer
from .layer_compose import layer_compose
from .partition_compose import PartitionComposer
from ..optimizer.simulated_annealing import SimulatedAnnealing
from ..utils import utils
import os
import csv
import itertools
import numpy as np
from multiprocessing import Pool
import networkx as nx


def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results

class PartitionParser():
    def __init__(self, model_name, optimization, singlethreaded, per_layer_plot, detailed):
        self.detailed = detailed
        self.model_name = model_name
        self.optimization = optimization
        self.singlethreaded = singlethreaded
        self.per_layer_plot = per_layer_plot
        self.model_descriptor = PartitionDescriptor(model_name, detailed)
        self.partition_composer = PartitionComposer()

        if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports')):
            os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports'))

        if self.detailed:
            self.layer_model_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + '_detailed.csv')
            self.layer_model_file_par = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + '_detailed_pareto.csv')
        else:
            self.layer_model_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + '.csv')
            self.layer_model_file_par = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + '_pareto.csv')

    def is_partition_input(self, partition, node_ids):
        if len(node_ids) > 1:
            return False
        for layer in partition:
            if node_ids[0] == self.model_descriptor.layers[layer]['node_out']:
                return False
        return True

    def is_partition_output(self, partition, node_id):
        for layer in partition:
            if node_id in self.model_descriptor.layers[layer]['node_in']:
                return False
        return True

    def connected_nodes(self, partition, node_id):
        nodes = []
        for layer in partition:
            if node_id in self.model_descriptor.layers[layer]['node_in']:
                nodes.append(layer)
        return nodes
    
    @staticmethod
    def visualize_graph(graph, path):
        PG = nx.nx_pydot.to_pydot(graph)
        PG.write_png(path + ".png")

    def create_graph(self, partition):
        graph = nx.DiGraph()
        for layer in partition:
            if self.model_descriptor.layers[layer]['operation'] == 'GlobalAveragePool':
                hw_layer = GAPLayer(self.model_descriptor.layers[layer], self.optimization)
                layer_type = self.model_descriptor.layers[layer]['operation']
            elif self.model_descriptor.layers[layer]['operation'] == 'Conv':
                hw_layer = Convolutional3DLayer(self.model_descriptor.layers[layer], self.optimization)
                layer_type = self.model_descriptor.layers[layer]['operation']
            elif self.model_descriptor.layers[layer]['operation'] == 'Relu' or self.model_descriptor.layers[layer]['operation'] == 'Sigmoid' or self.model_descriptor.layers[layer]['operation'] == 'Swish':
                hw_layer = ActivationLayer(self.model_descriptor.layers[layer], self.optimization)
                layer_type = 'Activation'
            elif self.model_descriptor.layers[layer]['operation'] == 'Mul' or self.model_descriptor.layers[layer]['operation'] == 'Add':
                hw_layer = ElementWiseLayer(self.model_descriptor.layers[layer], self.optimization)
                layer_type = 'ElementWise'
            elif self.model_descriptor.layers[layer]['operation'] == 'Gemm' or self.model_descriptor.layers[layer]['operation'] == 'MatMul':
                pass
                # layer_type = self.model_descriptor.layers[layer]['operation']
                # hw_layer = FCLayer(self.model_descriptor.layers[layer], self.optimization)
            elif self.model_descriptor.layers[layer]['operation'] == 'SqueezeExcitation':
                layer_type = self.model_descriptor.layers[layer]['operation']
                hw_layer = SqueezeExcitationLayer(self.model_descriptor.layers[layer], self.optimization)
            elif self.model_descriptor.layers[layer]['operation'] == 'BatchNormalization':
                layer_type = self.model_descriptor.layers[layer]['operation']
                hw_layer = BatchNorm3DLayer(self.model_descriptor.layers[layer], self.optimization)
            else:
                assert False, "{} operation in layer {} is not supported".format(self.model_descriptor.layers[layer]['operation'], layer)
            graph.add_node(layer, type=layer_type, hw=hw_layer)

        edges = []
        for name in graph.nodes():
            inputs = self.model_descriptor.layers[name]['node_in']
            outputs = self.model_descriptor.layers[name]['node_out']

            for conn_node in self.connected_nodes(partition, outputs):
                edges.append((name, conn_node))
        
        for edge in edges:
            graph.add_edge(*edge)
        
        return graph

    def model_partition(self, partition, name):

        graph = self.create_graph(partition)
        branch_edges = utils.get_branch_edges(graph)

        # Worst case scenario
        branch_buffer = 0
        branch_shapes = []
        for edge in branch_edges:
            assert graph.nodes[edge[0]]['hw'].output_shape == graph.nodes[edge[1]]['hw'].input_shape_1, "Layers input and output shapes does not match"
            branch_buffer += np.prod(np.array(graph.nodes[edge[0]]['hw'].output_shape[1:]))
            branch_shapes.append(graph.nodes[edge[0]]['hw'].output_shape)

        self.visualize_graph(graph, os.getcwd() + '/fpga_modeling_reports/partition_graphs/' + name)

        print("Partition: {}: ".format(name))
        optimizer = SimulatedAnnealing(graph, branch_buffer, partition_name=name)
        optimizer.run_optimizer()

        # config_points = {}
        # for layer in partition:
        #     config_points[layer] = utils.get_config_points(layer, self.layer_model_file_par, is_partitioning=True)

        # total = []
        # for c in config_points.values():
        #     total.append(c)

        # combinations = itertools.product(*total)

        # throughput_gops = []
        # throughput_vols = []
        # latency = []
        # dsp_util = []
        # bram_util = []

        # min_latency = 1000000000000
        # best = {}

        # #TODO: Iterate over different mem bandwidth values
        # bw_in_1, bw_in_2, bw_out = 1000000, 1000000, 1000000

        # with open(self.partition_model_file, mode='a') as partition_dp:
        #     csv_writer = csv.writer(partition_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        #     if not self.singlethreaded:
        #         processes_pool = Pool(10)
        #         input_vars = []

        #         for c in combinations:
        #             input_vars.append([sequencial, c, bw_in_1, bw_in_2, bw_out])
        #         results = multithreaded_modeling(self.partition_composer.get_design_point, input_vars, processes_pool)
        #         processes_pool.close()
        #         for r in results:
        #             if r['config']:
        #                 throughput_gops.append(r['GOP/s'])
        #                 throughput_vols.append(r['vols/s'])
        #                 latency.append(r['latency(C)'])
        #                 dsp_util.append(r['DSP'])
        #                 bram_util.append(r['BRAM'])

        #                 csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

        #                 if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
        #                     min_latency = r['latency(C)']
        #                     best = r
        #                 elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
        #                     if r['DSP'] < best['DSP']:
        #                         min_latency = r['latency(C)']
        #                         best = r
        #     else:
        #         for c in combinations:
        #             r = self.partition_composer.get_design_point(sequencial, c, bw_in_1, bw_in_2, bw_out)

        #             if r['config']:
        #                 throughput_gops.append(r['GOP/s'])
        #                 throughput_vols.append(r['vols/s'])
        #                 latency.append(r['latency(C)'])
        #                 dsp_util.append(r['DSP'])
        #                 bram_util.append(r['BRAM'])

        #                 csv_writer.writerow([name, r['latency(C)'], r['latency(S)'], r['GOP/s'], r['vols/s'], r['DSP'], r['BRAM'], r['rateIn1'], -1, r['rateOut'], r['depth'], r['muls'], r['adds'], r['memWords'], r['memKBs'], r['memBoundedIn1'], -1, r['memBoundedOut'], r['config']])

        #                 if r['latency(C)'] < min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
        #                     min_latency = r['latency(C)']
        #                     best = r
        #                 elif r['latency(C)'] == min_latency and (r['DSP'] < 90. and r['BRAM'] < 90.):
        #                     if r['DSP'] < best['DSP']:
        #                         min_latency = r['latency(C)']
        #                         best = r

        #     print("Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn1={:.2f}, RateOut={:.2f}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}".format(best['latency(C)'], best['latency(S)'], best['GOP/s'], best['vols/s'], best['DSP'], best['BRAM'], best['rateIn1'], best['rateOut'], best['depth'], best['muls'], best['adds'], best['memWords'], best['memKBs'], best['memBoundedIn1'], best['memBoundedOut']))
        #     print("*"*40)

    def model_layer(self, layer, layer_description):
        print("Modeling {} layer...".format(layer))
        throughput_gops, throughput_vols, latency, dsp_util, bram_util = layer_compose(layer, layer_description, self.layer_model_file, self.optimization, self.singlethreaded)

    def parse(self):
        self.partition_model_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', self.model_name + '_partitions.csv')

        with open(self.partition_model_file, mode='w') as partition_dp:
            csv_writer = csv.writer(partition_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Layer", "Latency(C)", "Latency(S)", "GOP/s", "volumes/s", "DSP(%)", "BRAM(%)", "RateIn1", "RateIn2", "RateOut", "Depth", "Muls", "Adds", "Mem(W)", "Mem(KB)", "MemBoundIn1", "MemBoundIn2", "MemBoundOut", "config"])

        for i, partition in enumerate(self.model_descriptor.partitions):
            part_name = 'part_{}'.format(i)
            self.model_partition(partition, name=part_name)

        utils.drop_duplicates_csv(self.partition_model_file)
        if self.per_layer_plot:
            utils.plot_layers_csv(self.partition_model_file, self.model_name)

    def model_individual_layers(self):

        with open(self.layer_model_file, mode='w') as layer_dp:
            csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Layer", "Latency(C)", "Latency(S)", "GOP/s", "volumes/s", "DSP(%)", "BRAM(%)", "RateIn1", "RateIn2", "RateOut", "Depth", "Muls", "Adds", "Mem(W)", "Mem(KB)", "MemBoundIn1", "MemBoundIn2", "MemBoundOut", "config"])
        with open(self.layer_model_file_par, mode='w') as layer_dp:
            csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Layer", "Latency(C)", "Latency(S)", "GOP/s", "volumes/s", "DSP(%)", "BRAM(%)", "RateIn1", "RateIn2", "RateOut", "Depth", "Muls", "Adds", "Mem(W)", "Mem(KB)", "MemBoundIn1", "MemBoundIn2", "MemBoundOut", "config"])

        for name, descriptor in self.model_descriptor.layers.items():
            self.model_layer(name, descriptor)

        utils.drop_duplicates_csv(self.layer_model_file)
        utils.get_paretto_csv(self.layer_model_file_par, self.layer_model_file)
        if self.per_layer_plot:
            utils.plot_layers_csv(self.layer_model_file_par, self.model_name)

