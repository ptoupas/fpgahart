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
import time
import itertools
import numpy as np
from multiprocessing import Pool
import networkx as nx


def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results

class PartitionParser():
    def __init__(self, model_name, singlethreaded, per_layer_plot, se_block, gap_approx):
        self.gap_approx = gap_approx
        self.se_block = se_block
        self.model_name = model_name
        self.singlethreaded = singlethreaded
        self.per_layer_plot = per_layer_plot
        self.model_descriptor = PartitionDescriptor(model_name, se_block)
        self.partition_composer = PartitionComposer()

        if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports')):
            os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports'))

        if self.se_block:
            self.layer_model_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + '_se.csv')
            self.layer_model_file_par = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + '_se_pareto.csv')
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
        print("*"*40)
        for layer in partition:
            print("Adding {} layer to graph...".format(layer))
            if self.model_descriptor.layers[layer]['operation'] == 'GlobalAveragePool':
                hw_layer = GAPLayer(self.model_descriptor.layers[layer])
                layer_type = self.model_descriptor.layers[layer]['operation']
            elif self.model_descriptor.layers[layer]['operation'] == 'Conv':
                hw_layer = Convolutional3DLayer(self.model_descriptor.layers[layer])
                layer_type = self.model_descriptor.layers[layer]['operation']
            elif self.model_descriptor.layers[layer]['operation'] == 'Relu' or self.model_descriptor.layers[layer]['operation'] == 'Sigmoid' or self.model_descriptor.layers[layer]['operation'] == 'Swish':
                hw_layer = ActivationLayer(self.model_descriptor.layers[layer])
                layer_type = 'Activation'
            elif self.model_descriptor.layers[layer]['operation'] == 'Mul' or self.model_descriptor.layers[layer]['operation'] == 'Add':
                hw_layer = ElementWiseLayer(self.model_descriptor.layers[layer])
                layer_type = 'ElementWise'
            elif self.model_descriptor.layers[layer]['operation'] == 'Gemm' or self.model_descriptor.layers[layer]['operation'] == 'MatMul':
                layer_type = self.model_descriptor.layers[layer]['operation']
                hw_layer = FCLayer(self.model_descriptor.layers[layer])
            elif self.model_descriptor.layers[layer]['operation'] == 'SqueezeExcitation':
                layer_type = self.model_descriptor.layers[layer]['operation']
                hw_layer = SqueezeExcitationLayer(self.model_descriptor.layers[layer])
            elif self.model_descriptor.layers[layer]['operation'] == 'BatchNormalization':
                layer_type = self.model_descriptor.layers[layer]['operation']
                hw_layer = BatchNorm3DLayer(self.model_descriptor.layers[layer])
            else:
                assert False, "{} operation in layer {} is not supported".format(self.model_descriptor.layers[layer]['operation'], layer)
            graph.add_node(layer, type=layer_type, hw=hw_layer)
        print("*"*40)

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
        for edge in branch_edges:
            max_shape = 0
            for pair in edge:
                if (graph.nodes[pair[0]]['type'] == 'ElementWise' and graph.nodes[pair[0]]['hw'].type == 'Mul') or (graph.nodes[pair[1]]['type'] == 'ElementWise' and graph.nodes[pair[1]]['hw'].type == 'Mul'):
                    continue
                assert graph.nodes[pair[0]]['hw'].output_shape == graph.nodes[pair[1]]['hw'].input_shape_1 or graph.nodes[pair[0]]['hw'].output_shape == graph.nodes[pair[1]]['hw'].input_shape_2, "Layers input and output shapes does not match"
                max_shape = max(max_shape, np.prod(np.array(graph.nodes[pair[0]]['hw'].output_shape[1:])))
            branch_buffer += max_shape

        self.visualize_graph(graph, os.getcwd() + '/fpga_modeling_reports/partition_graphs/' + name)

        print("Partition: {}: ".format(name))
        # optimizer = SimulatedAnnealing(graph, branch_mem=branch_buffer, partition_name=name, gap_approx=self.gap_approx)
        optimizer = SimulatedAnnealing(graph, partition_name=name, gap_approx=self.gap_approx)

        mwpc, solution_mem, solution_dp = optimizer.run_optimizer()
        if mwpc is None or solution_mem is None or solution_dp is None:
            raise Exception("Optimization failed")
        num_graphs = len(solution_mem)

        with open(self.partition_model_file, mode='a') as res_file:
            csv_writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            if num_graphs == 1:
                mem_config = (list(np.array(solution_mem[0][0]) * mwpc), list(np.array(solution_mem[0][1]) * mwpc))
                csv_row = [name, solution_dp[0]['latency(C)']-solution_dp[0]['depth'], solution_dp[0]['latency(C)'], solution_dp[0]['latency(S)'], solution_dp[0]['GOP/s'], solution_dp[0]['GOPs'], solution_dp[0]['vols/s'], solution_dp[0]['DSP'], solution_dp[0]['BRAM'], solution_dp[0]['rateIn'], solution_dp[0]['rateOut'], solution_dp[0]['depth'], solution_dp[0]['branch_depth'], solution_dp[0]['muls'], solution_dp[0]['adds'], solution_dp[0]['memWords'], solution_dp[0]['memKBs'], solution_dp[0]['dataSizeIn'], solution_dp[0]['dataSizeOut'], solution_dp[0]['memBoundedIn'], solution_dp[0]['memBoundedOut'], solution_dp[0]['config'], mem_config]
                csv_writer.writerow(csv_row)
            else:
                f_name = name
                f_latency_c = 0
                f_latency_s = 0
                f_dsps = 0
                f_brams = 0
                f_depth = 0
                f_muls = 0
                f_adds = 0
                f_mem_words = 0
                f_mem_kbs = 0
                f_total_ops = 0
                f_size_in = 0
                f_size_out = 0
                sub_rows = []
                for i in range(num_graphs):
                    mem_config = (list(np.array(solution_mem[i][0]) * mwpc), list(np.array(solution_mem[i][1]) * mwpc))
                    csv_row = [name+'_{}'.format(i), solution_dp[i]['latency(C)']-solution_dp[i]['depth'], solution_dp[i]['latency(C)'], solution_dp[i]['latency(S)'], solution_dp[i]['GOP/s'], solution_dp[i]['GOPs'], solution_dp[i]['vols/s'], solution_dp[i]['DSP'], solution_dp[i]['BRAM'], solution_dp[i]['rateIn'], solution_dp[i]['rateOut'], solution_dp[i]['depth'], solution_dp[i]['branch_depth'], solution_dp[i]['muls'], solution_dp[i]['adds'], solution_dp[i]['memWords'], solution_dp[i]['memKBs'], solution_dp[i]['dataSizeIn'], solution_dp[i]['dataSizeOut'], solution_dp[i]['memBoundedIn'], solution_dp[i]['memBoundedOut'], solution_dp[i]['config'], mem_config]
                    sub_rows.append(csv_row)
                    f_latency_c += solution_dp[i]['latency(C)']
                    f_latency_s += solution_dp[i]['latency(S)']
                    f_dsps += solution_dp[i]['DSP']
                    f_brams += solution_dp[i]['BRAM']
                    f_depth += solution_dp[i]['depth']
                    f_muls += solution_dp[i]['muls']
                    f_adds += solution_dp[i]['adds']
                    f_mem_words += solution_dp[i]['memWords']
                    f_mem_kbs += solution_dp[i]['memKBs']
                    f_total_ops += solution_dp[i]['GOPs']
                    f_size_in += solution_dp[i]['dataSizeIn']
                    f_size_out += solution_dp[i]['dataSizeOut']

                csv_row = [f_name, f_latency_c-f_depth, f_latency_c, f_latency_s, f_total_ops/f_latency_s, f_total_ops, 1/f_latency_s, f_dsps, f_brams, '', '', f_depth, f_muls, f_adds, f_mem_words, f_mem_kbs, f_size_in, f_size_out, '', '', '', '']
                csv_writer.writerow(csv_row)
                for sub_row in sub_rows:
                    csv_writer.writerow(sub_row)

    def model_layer(self, layer, layer_description):
        print("Modeling {} layer...".format(layer))
        throughput_gops, throughput_vols, latency, dsp_util, bram_util = layer_compose(layer, layer_description, self.layer_model_file, self.singlethreaded)

    def idetify_duplicates(self):
        partitions = {}
        for i, partition in enumerate(self.model_descriptor.partitions):
            graph = self.create_graph(partition)
            nodes_list = []
            for node in graph.nodes():
                hw = graph.nodes[node]['hw']
                if graph.nodes[node]['type'] == 'Conv':
                    nodes_list.append([hw.input_shape, hw.output_shape, hw.kernel_shape, hw.padding, hw.stride, hw.groups])
                elif graph.nodes[node]['type'] == 'ElementWise':
                    nodes_list.append([hw.input_shape_1, hw.input_shape_2, hw.output_shape])
                else:
                    nodes_list.append([hw.input_shape, hw.output_shape])
            partitions[i] = nodes_list

        prev_partition = None
        remove_duplicates = []
        for i, partition in enumerate(self.model_descriptor.partitions):
            v = partitions[i]
            if prev_partition is not None:
                if v == prev_partition:
                    remove_duplicates.append(i)
            prev_partition = v
        
        for index in sorted(remove_duplicates, reverse=True):
            del self.model_descriptor.partitions[index]

    def parse(self):
        self.idetify_duplicates()

        self.partition_model_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', self.model_name + '_partitions.csv')

        with open(self.partition_model_file, mode='w') as partition_dp:
            csv_writer = csv.writer(partition_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Part", "Latency(C)-No-Depth", "Latency(C)", "Latency(S)", "GOP/s", "GOPs", "volumes/s", "DSP(%)", "BRAM(%)", "RateIn", "RateOut", "Depth", "Branch Depth", "Muls", "Adds", "Mem(W)", "Mem(KB)", "DataSizeIn(MB)", "DataSizeOut(MB)", "MemBoundIn", "MemBoundOut", "config", "memconfig"])

        start = time.time()
        for i, partition in enumerate(self.model_descriptor.partitions):
            part_name = 'part_{}'.format(i)
            self.model_partition(partition, name=part_name)
        end = time.time()
        print("Partition modeling took {:.2f} seconds".format(end - start))
        
    def model_individual_layers(self):

        with open(self.layer_model_file, mode='w') as layer_dp:
            csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Layer", "Latency(C)", "Latency(S)", "GOP/s", "volumes/s", "DSP(%)", "BRAM(%)", "RateIn", "RateOut", "Depth", "Muls", "Adds", "Mem(W)", "Mem(KB)", "MemBoundIn", "MemBoundOut", "config"])
        with open(self.layer_model_file_par, mode='w') as layer_dp:
            csv_writer = csv.writer(layer_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Layer", "Latency(C)", "Latency(S)", "GOP/s", "volumes/s", "DSP(%)", "BRAM(%)", "RateIn", "RateOut", "Depth", "Muls", "Adds", "Mem(W)", "Mem(KB)", "MemBoundIn", "MemBoundOut", "config"])

        for name, descriptor in self.model_descriptor.layers.items():
            self.model_layer(name, descriptor)

        # utils.drop_duplicates_csv(self.layer_model_file)
        # utils.get_paretto_csv(self.layer_model_file_par, self.layer_model_file)
        # if self.per_layer_plot:
        #     utils.plot_layers_csv(self.layer_model_file_par, self.model_name)

    def model_custom_partition(self):
        self.partition_model_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', self.model_name + '_custom_partitions.csv')

        with open(self.partition_model_file, mode='w') as partition_dp:
            csv_writer = csv.writer(partition_dp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Part", "Latency(C)-No-Depth", "Latency(C)", "Latency(S)", "GOP/s", "GOPs", "volumes/s", "DSP(%)", "BRAM(%)", "RateIn", "RateOut", "Depth", "Branch Depth", "Muls", "Adds", "Mem(W)", "Mem(KB)", "DataSizeIn(MB)", "DataSizeOut(MB)", "MemBoundIn", "MemBoundOut", "config", "memconfig"])


        custom_partition = ['Custom_Conv_1']
        # self.model_descriptor.layers['Custom_Gap_1'] = {'operation': 'GlobalAveragePool',
        #                                                 'shape_in': [[1, 24, 16, 32, 32]],
        #                                                 'shape_out': [1, 24, 1, 1, 1],
        #                                                 'node_in': ['606'],
        #                                                 'node_out': '608',
        #                                                 'branching': False}
        self.model_descriptor.layers['Custom_Conv_1'] = {'operation': 'Conv',
                                                                'shape_in': [[1, 12, 16, 16, 16]],
                                                                'shape_out': [1, 12, 16, 16, 16],
                                                                'node_in': ['2'],
                                                                'node_out': '3',
                                                                'branching': False,
                                                                'kernel': [12, 1, 3, 3, 3],
                                                                'bias': [],
                                                                'padding': [1, 1, 1],
                                                                'stride': [1, 1, 1],
                                                                'groups': 12,
                                                                'dilation': [1, 1, 1]}
        self.model_descriptor.layers['Custom_Conv_2'] = {'operation': 'Conv',
                                                                'shape_in': [[1, 12, 16, 16, 16]],
                                                                'shape_out': [1, 24, 16, 16, 16],
                                                                'node_in': ['2'],
                                                                'node_out': '3',
                                                                'branching': False,
                                                                'kernel': [24, 12, 1, 1, 1],
                                                                'bias': [],
                                                                'padding': [0, 0, 0],
                                                                'stride': [1, 1, 1],
                                                                'groups': 1,
                                                                'dilation': [1, 1, 1]}
        # self.model_partition(custom_partition, name="Single_Layer")
        # exit()


        custom_partition = ['Custom_Relu', 'Custom_Conv_1', 'Custom_Swish', 'Custom_Add']
        self.model_descriptor.layers['Custom_Relu'] = {'operation': 'Relu',
                                                         'shape_in': [[1, 16, 8, 12, 12]],
                                                         'shape_out': [1, 16, 8, 12, 12],
                                                         'node_in': ['1'],
                                                         'node_out': '2',
                                                         'branching': False}
        self.model_descriptor.layers['Custom_Conv_1'] = {'operation': 'Conv',
                                                         'shape_in': [[1, 16, 8, 12, 12]],
                                                         'shape_out': [1, 16, 8, 12, 12],
                                                         'node_in': ['2'],
                                                         'node_out': '3',
                                                         'branching': False,
                                                         'kernel': [16, 1, 3, 3, 3],
                                                         'bias': [],
                                                         'padding': [1, 1, 1],
                                                         'stride': [1, 1, 1],
                                                         'groups': 16,
                                                         'dilation': [1, 1, 1]}
        self.model_descriptor.layers['Custom_Conv_2'] = {'operation': 'Conv',
                                                         'shape_in': [[1, 16, 8, 12, 12]],
                                                         'shape_out': [1, 16, 8, 12, 12],
                                                         'node_in': ['2'],
                                                         'node_out': '3',
                                                         'branching': False,
                                                         'kernel': [16, 16, 1, 1, 1],
                                                         'bias': [],
                                                         'padding': [0, 0, 0],
                                                         'stride': [1, 1, 1],
                                                         'groups': 1,
                                                         'dilation': [1, 1, 1]}
        self.model_descriptor.layers['Custom_Swish'] = {'operation': 'Swish',
                                                         'shape_in': [[1, 16, 8, 12, 12]],
                                                         'shape_out': [1, 16, 8, 12, 12],
                                                         'node_in': ['3'],
                                                         'node_out': '4',
                                                         'branching': False}
        self.model_descriptor.layers['Custom_Add'] = {'operation': 'Add',
                                                      'shape_in': [[1, 16, 8, 12, 12], [1, 16, 8, 12, 12]],
                                                      'shape_out': [1, 16, 8, 12, 12],
                                                      'node_in': ['2', '4'],
                                                      'node_out': '5',
                                                      'branching': False}
        # self.model_partition(custom_partition, name="Single_Layer_Branch")
        # exit()


        custom_partition = ['Relu_22', 'Conv_23', 'Relu_25', 'Conv_26', 'Swish_28', 'Conv_30', 'Add_32']
        # custom_partition = ['Relu_22', 'Conv_23', 'Relu_25', 'Swish_28', 'Conv_30', 'Add_32']

        self.model_descriptor.layers['Relu_22']['shape_in'] = [[1, 8, 6, 12, 12]]
        self.model_descriptor.layers['Relu_22']['shape_out'] = [1, 8, 6, 12, 12]

        self.model_descriptor.layers['Conv_23']['shape_in'] = [[1, 8, 6, 12, 12]]
        self.model_descriptor.layers['Conv_23']['shape_out'] = [1, 12, 6, 12, 12]
        self.model_descriptor.layers['Conv_23']['kernel'] = [12, 8, 1, 1, 1]
        self.model_descriptor.layers['Conv_23']['bias'] = [12]

        self.model_descriptor.layers['Relu_25']['shape_in'] = [[1, 12, 6, 12, 12]]
        self.model_descriptor.layers['Relu_25']['shape_out'] = [1, 12, 6, 12, 12]

        self.model_descriptor.layers['Conv_26']['shape_in'] = [[1, 12, 6, 12, 12]]
        self.model_descriptor.layers['Conv_26']['shape_out'] = [1, 12, 6, 12, 12]
        self.model_descriptor.layers['Conv_26']['kernel'] = [12, 1, 3, 3, 3]
        self.model_descriptor.layers['Conv_26']['bias'] = [12]
        self.model_descriptor.layers['Conv_26']['groups'] = 12

        self.model_descriptor.layers['Swish_28']['shape_in'] = [[1, 12, 6, 12, 12]]
        self.model_descriptor.layers['Swish_28']['shape_out'] = [1, 12, 6, 12, 12]
        # self.model_descriptor.layers['Swish_28']['node_in'] = ['594']

        self.model_descriptor.layers['Conv_30']['shape_in'] = [[1, 12, 6, 12, 12]]
        self.model_descriptor.layers['Conv_30']['shape_out'] = [1, 8, 6, 12, 12]
        self.model_descriptor.layers['Conv_30']['kernel'] = [8, 12, 1, 1, 1]
        self.model_descriptor.layers['Conv_30']['bias'] = [8]

        self.model_descriptor.layers['Add_32']['shape_in'] = [[1, 8, 6, 12, 12], [1, 8, 6, 12, 12]]
        self.model_descriptor.layers['Add_32']['shape_out'] = [1, 8, 6, 12, 12]

        self.model_partition(custom_partition, name="X3D_M_Layer_Type_3_RS")


        custom_partition = ['Relu_33', 'Conv_34', 'Relu_36', 'Conv_37', 'GlobalAveragePool_39', 'Conv_40', 'Relu_41', 'Conv_42', 'Sigmoid_43', 'Mul_44', 'Swish_45', 'Conv_47', 'Add_49']
        # custom_partition = ['Conv_37', 'GlobalAveragePool_39', 'Conv_40', 'Relu_41', 'Conv_42', 'Sigmoid_43', 'Mul_44']

        self.model_descriptor.layers['Relu_33']['shape_in'] = [[1, 8, 6, 12, 12]]
        self.model_descriptor.layers['Relu_33']['shape_out'] = [1, 8, 6, 12, 12]

        self.model_descriptor.layers['Conv_34']['shape_in'] = [[1, 8, 6, 12, 12]]
        self.model_descriptor.layers['Conv_34']['shape_out'] = [1, 12, 6, 12, 12]
        self.model_descriptor.layers['Conv_34']['kernel'] = [12, 8, 1, 1, 1]
        self.model_descriptor.layers['Conv_34']['bias'] = [12]

        self.model_descriptor.layers['Relu_36']['shape_in'] = [[1, 12, 6, 12, 12]]
        self.model_descriptor.layers['Relu_36']['shape_out'] = [1, 12, 6, 12, 12]

        self.model_descriptor.layers['Conv_37']['shape_in'] = [[1, 12, 6, 12, 12]]
        self.model_descriptor.layers['Conv_37']['shape_out'] = [1, 12, 6, 12, 12]
        self.model_descriptor.layers['Conv_37']['kernel'] = [12, 1, 3, 3, 3]
        self.model_descriptor.layers['Conv_37']['bias'] = [12]
        self.model_descriptor.layers['Conv_37']['groups'] = 12

        self.model_descriptor.layers['GlobalAveragePool_39']['shape_in'] = [[1, 12, 6, 12, 12]]
        self.model_descriptor.layers['GlobalAveragePool_39']['shape_out'] = [1, 12, 1, 1, 1]

        self.model_descriptor.layers['Conv_40']['shape_in'] = [[1, 12, 1, 1, 1]]
        self.model_descriptor.layers['Conv_40']['shape_out'] = [1, 8, 1, 1, 1]
        self.model_descriptor.layers['Conv_40']['kernel'] = [8, 12, 1, 1, 1]
        self.model_descriptor.layers['Conv_40']['bias'] = [8]

        self.model_descriptor.layers['Relu_41']['shape_in'] = [[1, 8, 1, 1, 1]]
        self.model_descriptor.layers['Relu_41']['shape_out'] = [1, 8, 1, 1, 1]

        self.model_descriptor.layers['Conv_42']['shape_in'] = [[1, 8, 1, 1, 1]]
        self.model_descriptor.layers['Conv_42']['shape_out'] = [1, 12, 1, 1, 1]
        self.model_descriptor.layers['Conv_42']['kernel'] = [12, 8, 1, 1, 1]
        self.model_descriptor.layers['Conv_42']['bias'] = [12]

        self.model_descriptor.layers['Sigmoid_43']['shape_in'] = [[1, 12, 1, 1, 1]]
        self.model_descriptor.layers['Sigmoid_43']['shape_out'] = [1, 12, 1, 1, 1]

        self.model_descriptor.layers['Mul_44']['shape_in'] = [[1, 12, 6, 12, 12], [1, 12, 1, 1, 1]]
        self.model_descriptor.layers['Mul_44']['shape_out'] = [1, 12, 6, 12, 12]
        
        self.model_descriptor.layers['Swish_45']['shape_in'] = [[1, 12, 6, 12, 12]]
        self.model_descriptor.layers['Swish_45']['shape_out'] = [1, 12, 6, 12, 12]

        self.model_descriptor.layers['Conv_47']['shape_in'] = [[1, 12, 6, 12, 12]]
        self.model_descriptor.layers['Conv_47']['shape_out'] = [1, 8, 6, 12, 12]
        self.model_descriptor.layers['Conv_47']['kernel'] = [8, 12, 1, 1, 1]
        self.model_descriptor.layers['Conv_47']['bias'] = [8]

        self.model_descriptor.layers['Add_49']['shape_in'] = [[1, 8, 6, 12, 12], [1, 8, 6, 12, 12]]
        self.model_descriptor.layers['Add_49']['shape_out'] = [1, 8, 6, 12, 12]

        self.model_partition(custom_partition, name="X3D_M_Layer_Type_2_RS")