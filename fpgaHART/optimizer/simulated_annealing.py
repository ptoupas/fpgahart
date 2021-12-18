from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.batchnorm_3d import BatchNorm3DLayer
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..layers.gap import GAPLayer
from ..layers.elemwise import ElementWiseLayer
from ..layers.activation import ActivationLayer
from ..layers.memory_interface import MemoryNode
from ..layers.fully_connected import FCLayer
from ..layers.base_layer import BaseLayer
from ..partitions.partition_compose import PartitionComposer
from ..utils import utils
from collections import deque
import scipy.constants as sc
import networkx as nx
import numpy as np
import itertools
import random
import copy
import math
import os


class SimulatedAnnealing(BaseLayer):
    def __init__(self, graph, branch_mem, t_min=0.0001, t_max=25, iterationPerTemp=20, cooling_rate=0.99, partition_name=''):
        super().__init__()
        self.part_name = partition_name

        self.partition_composer = PartitionComposer()
        self.graph = graph

        mem_kb = (branch_mem * self.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.bram_Kbytes)
        self.branch_bram_util = (mem_bram / self.bram) * 100
        self.branch_mem = branch_mem

        # Simulate Annealing Variables
        self.k          = sc.Boltzmann
        self.t_min      = t_min
        self.t_max      = t_max
        self.cooling_rate       = cooling_rate
        self.iterationPerTemp   = iterationPerTemp
    
    def has_gap(self):
        result = False
        for node in self.graph.nodes:
            op_type = self.graph.nodes[node]['type']
            hw = self.graph.nodes[node]['hw']
            if isinstance(hw, GAPLayer):
                result = True
                break
        return result

    def split_graph(self):
        """
            Create a 1st break point of the graph right before the mul operation of squeeze excitation module.
            Then search (using simulated annealing) from which point in the graph we will start the 2nd phase of the execution (or we will reconfigure the whole FPGA). Depending on the point we might need to store extra intermediate results of phase1 and read them from off-chip memory during phase 2. The other alternative will be to re-compute some layers but reduce the off-chip memory transfers.
        """
        mem_in_1, mem_out_1, mem_in_2, mem_out_2 = [] , [], [], []

        merge_nodes = [n for n in self.graph.nodes if self.graph.in_degree[n] > 1]
        split_nodes = [n for n in self.graph.nodes if self.graph.out_degree[n] > 1]
        break_node_gap = ''
        for node in self.graph.nodes:
            op_type = self.graph.nodes[node]['type']
            hw = self.graph.nodes[node]['hw']
            if op_type == 'ElementWise' and self.graph.in_degree[node] > 1 and hw.type == 'Mul':
                break_node_gap = node

        phase_1 = deque()
        check_node = break_node_gap
        predec_nodes = [n for n in self.graph.predecessors(check_node)]
        while predec_nodes:
            if len(predec_nodes) > 1:
                for pn in predec_nodes:
                    op_type = self.graph.nodes[pn]['type']
                    if op_type == 'Activation':
                        phase_1.appendleft(pn)
                        check_node = pn
                        predec_nodes = [n for n in self.graph.predecessors(check_node)]
                        break
            else:
                curr_node = predec_nodes[0]
                phase_1.appendleft(curr_node)
                check_node = curr_node
                predec_nodes = [n for n in self.graph.predecessors(check_node)]

        phase_1_graph_frozen = self.graph.subgraph(list(phase_1))
        phase_1_graph = nx.DiGraph(phase_1_graph_frozen)
        phase_1_edges = list(phase_1_graph.edges)
        
        mem_in_1_count = 1
        for node in list(phase_1_graph.nodes()):
            if node in merge_nodes and not phase_1_graph.in_degree[node] > 1:
                mem_in_1.append(node)
            if phase_1_graph.in_degree[node] == 0:
                mem_in_1.append(node)
            if phase_1_graph.out_degree[node] == 0:
                mem_out_1.append(node)


        phase_2_graph = self.graph.copy()
        gap_index = [i for i, n in enumerate(list(phase_1)) if 'Global' in n][0]
        phase_2_graph.remove_nodes_from(list(phase_1)[gap_index:])
        phase_2_graph.remove_edges_from(phase_1_edges[gap_index-1:])

        phase_2_read_point = random.choice([*[-1], *list(range(gap_index))])
        split_graph_start_point = 0
        split_nodes_ind = [i for i, n in enumerate(list(phase_1)) if n in split_nodes]
        for i in split_nodes_ind:
            if i < phase_2_read_point:
                split_graph_start_point = i
                #TODO: What if we have more than 1 point before the phase_2_read_point node?
                break
        if phase_2_read_point >= 0:
            phase_2_read_node = [n for i, n in enumerate(list(phase_1)) if phase_2_read_point==i][0]
            mem_out_1.append(phase_2_read_node)

        if split_graph_start_point > 0:
            phase_2_graph.remove_nodes_from(list(phase_1)[split_graph_start_point+1:phase_2_read_point+1])
            phase_2_graph.remove_edges_from(phase_1_edges[split_graph_start_point+1:phase_2_read_point+1])
        else:
            phase_2_graph.remove_nodes_from(list(phase_1)[:phase_2_read_point+1])
            phase_2_graph.remove_edges_from(phase_1_edges[:phase_2_read_point+1])

        mem_in_2_count = 1
        for node in list(phase_2_graph.nodes()):
            if node in merge_nodes and not phase_2_graph.in_degree[node] > 1:
                mem_in_2.append(node)
            if phase_2_graph.in_degree[node] == 0:# and node in [n for n in self.graph.successors(phase_2_read_node)]:
                mem_in_2.append(node) 
            if phase_2_graph.out_degree[node] == 0:
                mem_out_2.append(node)

        branch_edges_1 = utils.get_branch_edges(phase_1_graph)
        branch_edges_2 = utils.get_branch_edges(phase_2_graph)
        # Worst case scenario
        branch_buffer_1 = 0
        for edge in branch_edges_1:
            assert phase_1_graph.nodes[edge[0]]['hw'].output_shape == phase_1_graph.nodes[edge[1]]['hw'].input_shape_1, "Layers input and output shapes does not match"
            branch_buffer_1 += np.prod(np.array(phase_1_graph.nodes[edge[0]]['hw'].output_shape[1:]))
        branch_buffer_2 = 0
        for edge in branch_edges_2:
            assert phase_2_graph.nodes[edge[0]]['hw'].output_shape == phase_2_graph.nodes[edge[1]]['hw'].input_shape_1, "Layers input and output shapes does not match"
            branch_buffer_2 += np.prod(np.array(phase_2_graph.nodes[edge[0]]['hw'].output_shape[1:]))

        mem_kb = ((branch_buffer_1 + branch_buffer_2) * self.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.bram_Kbytes)
        branch_bram_util = (mem_bram / self.bram) * 100
        if branch_bram_util > 80.:
            raise ValueError("BRAM utilization is {}%. Buffering cant be used in one of the splitted graphs.".format(branch_bram_util))
        return phase_1_graph, phase_2_graph, branch_buffer_1, branch_buffer_2, mem_in_1, mem_out_1, mem_in_2, mem_out_2

    @staticmethod
    def validate_configs(graph_1_dp, graph_2_dp):
        g_1_dsp_util = graph_1_dp['DSP']
        g_2_dsp_util = graph_2_dp['DSP']

        g_1_bram_util = graph_1_dp['BRAM']
        g_2_bram_util = graph_2_dp['BRAM']

        if g_1_dsp_util + g_2_dsp_util >= 90.:
            return False
        if g_1_bram_util + g_2_bram_util >= 90.:
            return False

        return True

    def run_optimizer_double_graph(self):
        graph_1, graph_2, bb1, bb2, mem_in_1, mem_out_1, mem_in_2, mem_out_2 = self.split_graph()

        nIN1 = len(mem_in_1)
        nOUT1 = len(mem_out_1)
        assert nIN1 > 0 and nOUT1 > 0, 'No memory in/out nodes found'
        config_1, mem_bw_1 = self.generate_random_config(target_graph=graph_1, n_in=nIN1, n_out=nOUT1)
        cost_1, dp_info_1 = self.get_cost(config_1, mem_bw_1, target_graph=graph_1, branch_mem_update=bb1, mem_in_conns=mem_in_1, mem_out_conns=mem_out_1)

        nIN2 = len(mem_in_2)
        nOUT2 = len(mem_out_2)
        assert nIN2 > 0 and nOUT2 > 0, 'No memory in/out nodes found'
        config_2, mem_bw_2 = self.generate_random_config(target_graph=graph_2, n_in=nIN2, n_out=nOUT2)
        cost_2, dp_info_2 = self.get_cost(config_2, mem_bw_2, target_graph=graph_2, branch_mem_update=bb2, mem_in_conns=mem_in_2, mem_out_conns=mem_out_2)

        if cost_1 is None or cost_2 is None:
            for _ in range(100):
                graph_1, graph_2, bb1, bb2, mem_in_1, mem_out_1, mem_in_2, mem_out_2 = self.split_graph()

                nIN1 = len(mem_in_1)
                nOUT1 = len(mem_out_1)
                assert nIN1 > 0 and nOUT1 > 0, 'No memory in/out nodes found'
                config_1, mem_bw_1 = self.generate_random_config(target_graph=graph_1, n_in=nIN1, n_out=nOUT1)
                cost_1, dp_info_1 = self.get_cost(config_1, mem_bw_1, target_graph=graph_1, branch_mem_update=bb1, mem_in_conns=mem_in_1, mem_out_conns=mem_out_1)
                
                nIN2 = len(mem_in_2)
                nOUT2 = len(mem_out_2)
                assert nIN2 > 0 and nOUT2 > 0, 'No memory in/out nodes found'
                config_2, mem_bw_2 = self.generate_random_config(target_graph=graph_2, n_in=nIN2, n_out=nOUT2)
                cost_2, dp_info_2 = self.get_cost(config_2, mem_bw_2, target_graph=graph_2, branch_mem_update=bb2, mem_in_conns=mem_in_2, mem_out_conns=mem_out_2)
                if cost_1 is not None and cost_2 is not None and self.validate_configs(dp_info_1, dp_info_2):
                    break
            if cost_1 is None or cost_2 is None:
                return None, None, None

        prev_state_1 = config_1
        prev_state_2 = config_2
        prev_cost = cost_1 + cost_2
        solution_dp_1 = dp_info_1
        solution_dp_2 = dp_info_2
        solution_mem_1 = mem_bw_1
        solution_mem_2 = mem_bw_2

        current_temp = self.t_max

        print(f"Temperature  |  Latency")
        split_graph_count = 0
        while current_temp > self.t_min:
            for i in range(self.iterationPerTemp):
                if split_graph_count % 1000 == 0:
                    graph_1, graph_2, bb1, bb2, mem_in_1, mem_out_1, mem_in_2, mem_out_2 = self.split_graph()

                prev_state_1 = self.fix_inconsistent_config(prev_state_1, graph_1)
                prev_state_2 = self.fix_inconsistent_config(prev_state_2, graph_2)

                nIN1 = len(mem_in_1)
                nOUT1 =len(mem_out_1)
                assert nIN1 > 0 and nOUT1 > 0, 'No memory in/out nodes found'
                new_state_1, new_mem_bw_1 = self.generate_random_config(target_graph=graph_1, neighbours=True, prev_state=prev_state_1, n_in=nIN1, n_out=nOUT1)
                new_cost_1, new_dp_info_1 = self.get_cost(new_state_1, new_mem_bw_1, target_graph=graph_1, branch_mem_update=bb1, mem_in_conns=mem_in_1, mem_out_conns=mem_out_1)

                nIN2 = len(mem_in_2)
                nOUT2 = len(mem_out_2)
                assert nIN2 > 0 and nOUT2 > 0, 'No memory in/out nodes found'
                new_state_2, new_mem_bw_2 = self.generate_random_config(target_graph=graph_2, neighbours=True, prev_state=prev_state_2, n_in=nIN2, n_out=nOUT2)
                new_cost_2, new_dp_info_2 = self.get_cost(new_state_2, new_mem_bw_2, target_graph=graph_2, branch_mem_update=bb2, mem_in_conns=mem_in_2, mem_out_conns=mem_out_2)

                if new_cost_1 is None or new_cost_2 is None or not self.validate_configs(new_dp_info_1, new_dp_info_2):
                    continue
                split_graph_count += 1

                new_cost = new_cost_1 + new_cost_2
                cost_diff = prev_cost - new_cost
                if cost_diff >= 0:
                    prev_state_1 = copy.deepcopy(new_state_1)
                    prev_state_2 = copy.deepcopy(new_state_2)
                    prev_cost = copy.deepcopy(new_cost)
                    solution_mem_1, solution_dp_1 = copy.deepcopy(new_mem_bw_1), copy.deepcopy(new_dp_info_1)
                    solution_mem_2, solution_dp_2 = copy.deepcopy(new_mem_bw_2), copy.deepcopy(new_dp_info_2)
                    self.visualize_graph(graph_1, os.getcwd() + '/fpga_modeling_reports/partition_graphs/' + self.part_name + '_phase_1')
                    self.visualize_graph(graph_2, os.getcwd() + '/fpga_modeling_reports/partition_graphs/' + self.part_name + '_phase_2')
                else:
                    if random.uniform(0, 1) < math.exp((cost_diff/(self.k*current_temp))):
                        prev_state_1 = copy.deepcopy(new_state_1)
                        prev_state_2 = copy.deepcopy(new_state_2)
                        prev_cost = copy.deepcopy(new_cost)
                        solution_mem_1, solution_dp_1 = copy.deepcopy(new_mem_bw_1), copy.deepcopy(new_dp_info_1)
                        solution_mem_2, solution_dp_2 = copy.deepcopy(new_mem_bw_2), copy.deepcopy(new_dp_info_2)
                        self.visualize_graph(graph_1, os.getcwd() + '/fpga_modeling_reports/partition_graphs/' + self.part_name + '_phase_1')
                        self.visualize_graph(graph_2, os.getcwd() + '/fpga_modeling_reports/partition_graphs/' + self.part_name + '_phase_2')

            current_temp *= self.cooling_rate
            # current_temp -= self.cooling_rate
            # current_temp = current_temp/(1 + self.cooling_rate*current_temp)
            print(f"{current_temp:.5e}\t{prev_cost:.5e}", end='\r')

        print(f"\n\nLatency: {prev_cost}. volumes/s: {1/prev_cost}.\nSolution 1: Memory IN {list(np.array(solution_mem_1[0]) * self.mem_words_per_cycle)}, Memory OUT {list(np.array(solution_mem_1[1]) * self.mem_words_per_cycle)}\nSolution 2: Memory IN {list(np.array(solution_mem_2[0]) * self.mem_words_per_cycle)}, Memory OUT {list(np.array(solution_mem_2[1]) * self.mem_words_per_cycle)}.")
        print("Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}\nPartition Configuration: {}".format(solution_dp_1['latency(C)'], solution_dp_1['latency(S)'], solution_dp_1['GOP/s'], solution_dp_1['vols/s'], solution_dp_1['DSP'], solution_dp_1['BRAM'], solution_dp_1['rateIn'], solution_dp_1['rateOut'], solution_dp_1['depth'], solution_dp_1['muls'], solution_dp_1['adds'], solution_dp_1['memWords'], solution_dp_1['memKBs'], solution_dp_1['memBoundedIn'], solution_dp_1['memBoundedOut'], solution_dp_1['config']))
        print("Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}\nPartition Configuration: {}".format(solution_dp_2['latency(C)'], solution_dp_2['latency(S)'], solution_dp_2['GOP/s'], solution_dp_2['vols/s'], solution_dp_2['DSP'], solution_dp_2['BRAM'], solution_dp_2['rateIn'], solution_dp_2['rateOut'], solution_dp_2['depth'], solution_dp_2['muls'], solution_dp_2['adds'], solution_dp_2['memWords'], solution_dp_2['memKBs'], solution_dp_2['memBoundedIn'], solution_dp_2['memBoundedOut'], solution_dp_2['config']))
        print("*"*40)
        return self.mem_words_per_cycle, [solution_mem_1, solution_mem_2], [solution_dp_1, solution_dp_2]

    def run_optimizer(self):
        if self.has_gap() and self.branch_bram_util > 80:
            return self.run_optimizer_double_graph()

        config, mem_bw = self.generate_random_config()
        cost, dp_info = self.get_cost(config, mem_bw)

        if cost is None:
            for _ in range(100):
                config, mem_bw = self.generate_random_config()
                cost, dp_info = self.get_cost(config, mem_bw)
                if cost is not None:
                    break
            if cost is None:
                return None, None, None

        prev_state = config
        prev_cost = cost
        solution_dp = dp_info
        solution_mem = mem_bw

        current_temp = self.t_max

        print(f"Temperature  |  Latency")
        while current_temp > self.t_min:
            
            for i in range(self.iterationPerTemp):
                new_state, new_mem_bw = self.generate_random_config(neighbours=True, prev_state=prev_state)
                new_cost, new_dp_info = self.get_cost(new_state, new_mem_bw)

                if new_cost is None:
                    continue

                cost_diff = prev_cost - new_cost
                if cost_diff >= 0:
                    prev_state = copy.deepcopy(new_state)
                    prev_cost = copy.deepcopy(new_cost)
                    solution_mem, solution_dp = copy.deepcopy(new_mem_bw), copy.deepcopy(new_dp_info)
                else:
                    if random.uniform(0, 1) < math.exp((cost_diff/(self.k*current_temp))):
                        prev_state = copy.deepcopy(new_state)
                        prev_cost = copy.deepcopy(new_cost)
                        solution_mem, solution_dp = copy.deepcopy(new_mem_bw), copy.deepcopy(new_dp_info)

            current_temp *= self.cooling_rate
            print(f"{current_temp:.5e}\t{prev_cost:.5e}", end='\r')
        
        print(f"\n\nLatency: {prev_cost}.\nFinal Memory IN {list(np.array(solution_mem[0]) * self.mem_words_per_cycle)}, Memory OUT {list(np.array(solution_mem[1]) * self.mem_words_per_cycle)}.")
        print("Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}\nPartition Configuration: {}".format(solution_dp['latency(C)'], solution_dp['latency(S)'], solution_dp['GOP/s'], solution_dp['vols/s'], solution_dp['DSP'], solution_dp['BRAM'], solution_dp['rateIn'], solution_dp['rateOut'], solution_dp['depth'], solution_dp['muls'], solution_dp['adds'], solution_dp['memWords'], solution_dp['memKBs'], solution_dp['memBoundedIn'], solution_dp['memBoundedOut'], solution_dp['config']))
        print("*"*40)
        return self.mem_words_per_cycle, [solution_mem], [solution_dp]

    def get_cost(self, config, mem_bw, target_graph=None, branch_mem_update=None, mem_in_conns=[], mem_out_conns=[]):
        """
            We should be able to choose whether we want to optimize for latency or throughput
        """
        if target_graph is None:
            graph = self.graph.copy()
        else:
            graph = target_graph
        if branch_mem_update is None:
            branch_mem = self.branch_mem
        else:
            branch_mem = branch_mem_update

        comb_config = {}
        for k, v in config.items():
            if v['op_type'] == 'GlobalAveragePool':
                comb_config[k] = [v['coarse_in'], v['coarse_out']]
            elif v['op_type'] == 'Conv':
                comb_config[k] = [v['fine'], v['coarse_in'], v['coarse_out']]
            elif v['op_type'] == 'Activation':
                comb_config[k] = [v['coarse_in']]
            elif v['op_type'] == 'ElementWise':
                comb_config[k] = [v['coarse_in_1'], v['coarse_in_2'], v['coarse_out']]
            elif v['op_type'] == 'BatchNormalization':
                comb_config[k] = [v['coarse_in']]
            elif v['op_type'] == 'Gemm':
                comb_config[k] = [v['coarse_in'], v['coarse_out']]
            else:
                assert False, "Not supported layer"
        
        mem_in = mem_in_conns
        mem_out = mem_out_conns
        read_points, write_points = self.add_off_chip_connections(graph, mem_in, mem_out)
        # self.visualize_graph(graph, os.getcwd() + '/fpga_modeling_reports/partition_graphs/' + self.part_name + '_int')
        dp_info = self.partition_composer.get_design_point(graph.copy(), comb_config, mem_bw[0], mem_bw[1], read_points, write_points, branch_mem)
        if dp_info['config']:
            return dp_info['latency(S)'], dp_info
        return None, None

    @staticmethod
    def visualize_graph(graph, path):
        PG = nx.nx_pydot.to_pydot(graph)
        PG.write_png(path + ".png")

    def add_off_chip_connections(self, graph, in_connections, out_connections):
        read_points = []
        write_points = []

        mem_in_count = 1
        mem_out_count = 1
        for n in graph.nodes():
            edges_in = graph.in_edges(n)
            edges_out = graph.out_edges(n)
            if not edges_in:
                input_node = n
            if not edges_out:
                output_node = n

        if not in_connections and not out_connections:
            read_points.append(input_node)
            self.add_node_to_position(G=graph, new_node='Mem_in{}'.format(mem_in_count), connect_node=input_node, connect_pos='pre')
            mem_in_count += 1  
            write_points.append(output_node)
            self.add_node_to_position(G=graph, new_node='Mem_out{}'.format(mem_out_count), connect_node=output_node, connect_pos='post')
            mem_out_count += 1

        for con_in in in_connections:
            self.add_node_to_position(G=graph, new_node='Mem_in{}'.format(mem_in_count), connect_node=con_in, connect_pos='pre')
            read_points.append(con_in)
            mem_in_count += 1

        for con_out in out_connections:
            self.add_node_to_position(G=graph, new_node='Mem_out{}'.format(mem_out_count), connect_node=con_out, connect_pos='post')
            write_points.append(con_out)
            mem_out_count += 1
        
        return read_points, write_points

    @staticmethod
    def add_node_to_position(G, new_node, connect_node, connect_pos, is_input=False, is_output=False):
        if connect_pos == 'pre':
            edge = (new_node, connect_node)
        elif connect_pos == 'post':
            edge = (connect_node, new_node)
        else:
            raise Exception(f"Invalid connect_pos {connect_pos}")
        old_nodes = G.copy().nodes()
        nodes = list(G.nodes())
        edges = list(G.edges())
        if is_input:
            new_nodes = [new_node] + nodes
            new_edges = [edge] + edges
        elif is_output:
            new_nodes = nodes + [new_node]
            new_edges = edges + [edge]
        else:
            node_idx = nodes.index(connect_node)
            new_nodes = nodes.copy()
            if connect_pos == 'pre':
                new_nodes.insert(node_idx, new_node)
            elif connect_pos == 'post':
                new_nodes.insert(node_idx + 1, new_node)
            new_edges = edges.copy()
            new_edges.append(edge)
        G.remove_nodes_from(nodes)
        G.remove_edges_from(edges)
        G.add_nodes_from(new_nodes)
        G.add_edges_from(new_edges)

        for node in G.nodes():
            if node in old_nodes:
                G.nodes._nodes[node] = old_nodes[node]
            else:
                if 'Mem_in' in node:
                    if old_nodes[edge[1]]['type'] == 'ElementWise':
                        extra_inputs = [n for n in G.predecessors(edge[1]) if not n==node]
                        for e in extra_inputs:
                            assert not old_nodes[e]['type'] == 'ElementWise', "Current graph sequence cannot be handled."
                            shape_e = old_nodes[e]['hw'].output_shape
                            if shape_e == old_nodes[edge[1]]['hw'].input_shape_1:
                                shape = old_nodes[edge[1]]['hw'].input_shape_2
                            elif shape_e == old_nodes[edge[1]]['hw'].input_shape_2:
                                shape = old_nodes[edge[1]]['hw'].input_shape_1
                            else:
                                raise Exception("Invalid shape for Memory in node")
                        if not extra_inputs:
                            shape = old_nodes[edge[1]]['hw'].input_shape_1
                    else:
                        shape = old_nodes[edge[1]]['hw'].input_shape
                    G.nodes._nodes[node] = {'type': 'mem_in', 'hw': MemoryNode('in', shape)}
                elif 'Mem_out' in node:
                    shape = old_nodes[edge[0]]['hw'].output_shape
                    G.nodes._nodes[node] = {'type': 'mem_out', 'hw': MemoryNode('out', shape)}
        del old_nodes

    @staticmethod
    def get_mem_bw_feasible(n_in=0, n_out=0):
        rand_vals = []
        for i in range(n_in+n_out):
            rand_vals.append(random.randint(1,100))

        total_sum = np.sum(np.array(rand_vals))

        perc = []
        for i in range(n_in+n_out):
            perc.append(rand_vals[i] / total_sum)

        assert math.isclose(np.sum(np.array(perc)), 1.0), "Sum of mem_in_1_perc, mem_in_2_perc and mem_out_perc should be 1"

        perc_in = perc[:n_in]
        perc_out = perc[n_in:]

        return perc_in, perc_out

    @staticmethod
    def fix_inconsistent_config(config, graph):
        new_config = config.copy()
        for node in config.keys():
            if not graph.has_node(node):
                del new_config[node]
        for node in graph.nodes():
            if not node in config.keys():
                op_type = graph.nodes[node]['type']
                hw = graph.nodes[node]['hw']
                if isinstance(hw, GAPLayer):
                    channels = hw.channels
                    filters = hw.filters
                    coarse_in_feasible = utils.get_factors(channels)
                    coarse_out_feasible = utils.get_factors(filters)
                    coarse_in_factor = random.choice(coarse_in_feasible)/channels
                    coarse_out_factor = random.choice(coarse_out_feasible)/filters
                    new_config[node] = {'op_type': op_type,
                                'coarse_in': coarse_in_factor,
                                'coarse_out': coarse_out_factor}
                elif isinstance(hw, Convolutional3DLayer):
                    channels = hw.channels
                    filters = hw.filters
                    kernel_size = hw.kernel_shape
                    coarse_in_feasible = utils.get_factors(channels)
                    coarse_out_feasible = utils.get_factors(filters)
                    fine_feasible = utils.get_fine_feasible(kernel_size)
                    coarse_in_factor = random.choice(coarse_in_feasible)/channels
                    coarse_out_factor = random.choice(coarse_out_feasible)/filters
                    fine_factor = random.choice(fine_feasible)/np.prod(np.array(kernel_size))
                    new_config[node] = {'op_type': op_type,
                                'fine': fine_factor,
                                'coarse_in': coarse_in_factor,
                                'coarse_out': coarse_out_factor}
                elif isinstance(hw, ActivationLayer):
                    channels = hw.channels
                    coarse_in_feasible = utils.get_factors(channels)
                    coarse_in_factor = random.choice(coarse_in_feasible)/channels
                    new_config[node] = {'op_type': op_type,
                        'coarse_in': coarse_in_factor}
                elif isinstance(hw, ElementWiseLayer):
                    channels_1 = hw.channels_1
                    channels_2 = hw.channels_2
                    filters = hw.filters
                    coarse_in1_feasible = utils.get_factors(channels_1)
                    coarse_in2_feasible = utils.get_factors(channels_2)
                    coarse_out_feasible = utils.get_factors(filters)
                    coarse_in1_factor = random.choice(coarse_in1_feasible)/channels_1
                    coarse_in2_factor = random.choice(coarse_in2_feasible)/channels_2
                    coarse_out_factor = random.choice(coarse_out_feasible)/filters
                    new_config[node] = {'op_type': op_type,
                                'coarse_in_1': coarse_in1_factor,
                                'coarse_in_2': coarse_in2_factor,
                                'coarse_out': coarse_out_factor}
                elif isinstance(hw, BatchNorm3DLayer):
                    channels = hw.channels
                    coarse_in_feasible = utils.get_factors(channels)
                    coarse_in_factor = random.choice(coarse_in_feasible)/channels
                    new_config[node] = {'op_type': op_type,
                        'coarse_in': coarse_in_factor}
                elif isinstance(hw, SqueezeExcitationLayer):
                    assert False, "Not supported layer (SqueezeExcitationLayer)"
                elif isinstance(hw, FCLayer):
                    dim_in = hw.dim_in
                    dim_out = hw.dim_out
                    coarse_in_feasible = utils.get_factors(dim_in)
                    coarse_out_feasible = utils.get_factors(dim_out)
                    coarse_in_factor = random.choice(coarse_in_feasible)/dim_in
                    coarse_out_factor = random.choice(coarse_out_feasible)/dim_out
                    new_config[node] = {'op_type': op_type,
                            'coarse_in': coarse_in_factor,
                            'coarse_out': coarse_out_factor}
                elif isinstance(hw, MemoryNode):
                    continue
                else:
                    assert False, "Not supported layer"
        return new_config

    def generate_random_config(self, target_graph=None, neighbours=False, prev_state=None, n_in=1, n_out=1):
        if target_graph is None:
            graph = self.graph.copy()
        else:
            graph = target_graph

        if neighbours:
            number_of_new_configs = math.ceil(graph.order() * 0.25)

            # without replacement
            config_nodes = random.sample(list(graph.nodes), number_of_new_configs)
            # with replacement
            # config_nodes = random.choices(list(graph.nodes), k=number_of_new_configs)

            config = prev_state
        else:
            config_nodes = list(graph.nodes)
            config = {}

        for node in config_nodes:
            op_type = graph.nodes[node]['type']
            hw = graph.nodes[node]['hw']
            if isinstance(hw, GAPLayer):
                channels = hw.channels
                filters = hw.filters
                coarse_in_feasible = utils.get_factors(channels)
                coarse_out_feasible = utils.get_factors(filters)
                coarse_in_factor = random.choice(coarse_in_feasible)/channels
                coarse_out_factor = random.choice(coarse_out_feasible)/filters
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove('op_type')
                    apply_transform = random.choice(transformations)
                    if apply_transform == 'coarse_in':
                        config[node][apply_transform] = coarse_in_factor
                    elif apply_transform == 'coarse_out':
                        config[node][apply_transform] = coarse_out_factor
                else:
                    config[node] = {'op_type': op_type,
                                'coarse_in': coarse_in_factor,
                                'coarse_out': coarse_out_factor}
            elif isinstance(hw, Convolutional3DLayer):
                channels = hw.channels
                filters = hw.filters
                kernel_size = hw.kernel_shape
                coarse_in_feasible = utils.get_factors(channels)
                coarse_out_feasible = utils.get_factors(filters)
                fine_feasible = utils.get_fine_feasible(kernel_size)
                coarse_in_factor = random.choice(coarse_in_feasible)/channels
                coarse_out_factor = random.choice(coarse_out_feasible)/filters
                fine_factor = random.choice(fine_feasible)/np.prod(np.array(kernel_size))
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove('op_type')
                    apply_transform = random.choice(transformations)
                    if apply_transform == 'coarse_in':
                        config[node][apply_transform] = coarse_in_factor
                    elif apply_transform == 'coarse_out':
                        config[node][apply_transform] = coarse_out_factor
                    elif apply_transform == 'fine':
                        config[node][apply_transform] = fine_factor
                else:
                    config[node] = {'op_type': op_type,
                                'fine': fine_factor,
                                'coarse_in': coarse_in_factor,
                                'coarse_out': coarse_out_factor}
            elif isinstance(hw, ActivationLayer):
                channels = hw.channels
                coarse_in_feasible = utils.get_factors(channels)
                coarse_in_factor = random.choice(coarse_in_feasible)/channels
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove('op_type')
                    apply_transform = random.choice(transformations)
                    if apply_transform == 'coarse_in':
                        config[node][apply_transform] = coarse_in_factor
                else:
                    config[node] = {'op_type': op_type,
                        'coarse_in': coarse_in_factor}
            elif isinstance(hw, ElementWiseLayer):
                channels_1 = hw.channels_1
                depth_1 = hw.depth_in_1
                channels_2 = hw.channels_2
                depth_2 = hw.depth_in_2
                filters = hw.filters
                depth_out = hw.depth_out
                coarse_in1_feasible = utils.get_conbinations(utils.get_factors(channels_1), utils.get_factors(depth_1))
                coarse_in2_feasible = utils.get_conbinations(utils.get_factors(channels_2), utils.get_factors(depth_2))
                coarse_out_feasible = utils.get_conbinations(utils.get_factors(filters), utils.get_factors(depth_out))
                coarse_in1_factor = random.choice(coarse_in1_feasible)/(channels_1*depth_1)
                coarse_in2_factor = random.choice(coarse_in2_feasible)/(channels_2*depth_2)
                coarse_out_factor = random.choice(coarse_out_feasible)/(filters*depth_out)
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove('op_type')
                    apply_transform = random.choice(transformations)
                    if apply_transform == 'coarse_in_1':
                        config[node][apply_transform] = coarse_in1_factor
                    elif apply_transform == 'coarse_in_2':
                        config[node][apply_transform] = coarse_in2_factor
                    elif apply_transform == 'coarse_out':
                        config[node][apply_transform] = coarse_out_factor
                else:
                    config[node] = {'op_type': op_type,
                                'coarse_in_1': coarse_in1_factor,
                                'coarse_in_2': coarse_in2_factor,
                                'coarse_out': coarse_out_factor}
            elif isinstance(hw, BatchNorm3DLayer):
                channels = hw.channels
                coarse_in_feasible = utils.get_factors(channels)
                coarse_in_factor = random.choice(coarse_in_feasible)/channels
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove('op_type')
                    apply_transform = random.choice(transformations)
                    if apply_transform == 'coarse_in':
                        config[node][apply_transform] = coarse_in_factor
                else:
                    config[node] = {'op_type': op_type,
                        'coarse_in': coarse_in_factor}
            elif isinstance(hw, SqueezeExcitationLayer):
                assert False, "Not supported layer (SqueezeExcitationLayer)"
            elif isinstance(hw, FCLayer):
                dim_in = hw.dim_in
                dim_out = hw.dim_out
                coarse_in_feasible = utils.get_factors(dim_in)
                coarse_out_feasible = utils.get_factors(dim_out)
                coarse_in_factor = random.choice(coarse_in_feasible)/dim_in
                coarse_out_factor = random.choice(coarse_out_feasible)/dim_out
                if neighbours and node in config.keys():
                    transformations = list(config[node].keys())
                    transformations.remove('op_type')
                    apply_transform = random.choice(transformations)
                    if apply_transform == 'coarse_in':
                        config[node][apply_transform] = coarse_in_factor
                    elif apply_transform == 'coarse_out':
                        config[node][apply_transform] = coarse_out_factor
                else:
                    config[node] = {'op_type': op_type,
                            'coarse_in': coarse_in_factor,
                            'coarse_out': coarse_out_factor}
            elif isinstance(hw, MemoryNode):
                continue
            else:
                assert False, "Not supported layer"

        mem_config_in, mem_config_out = self.get_mem_bw_feasible(n_in=n_in, n_out=n_out)

        return config, [mem_config_in, mem_config_out]
    
    # TODO: Revise that to follow the changes on run_optimizer
    def run_optimizer_layer(self, layer):
        config, mem_bw = self.generate_random_config_layer(layer)
        cost, dp_info = self.get_cost_layer(config, mem_bw, layer)

        if cost is None:
            while cost is None:
                config, mem_bw = self.generate_random_config_layer(layer)
                cost, dp_info = self.get_cost_layer(config, mem_bw, layer)

        prev_state = config
        solution_dp = dp_info
        prev_cost = cost

        current_temp = self.t_max

        print(f"Temperature  |  Latency")
        while current_temp > self.t_min:
            
            for i in range(self.iterationPerTemp):
                new_state, new_mem_bw = self.generate_random_config_layer(layer)
                new_cost, new_dp_info = self.get_cost_layer(new_state, new_mem_bw, layer)

                if new_cost is None:
                    continue

                cost_diff = prev_cost - new_cost
                if cost_diff >= 0:
                    prev_state = copy.deepcopy(new_state)
                    prev_cost = copy.deepcopy(new_cost)
                    solution_mem, solution_dp = copy.deepcopy(new_mem_bw), copy.deepcopy(new_dp_info)
                else:
                    if random.uniform(0, 1) < math.exp((cost_diff/(self.k*current_temp))):
                        prev_state = copy.deepcopy(new_state)
                        prev_cost = copy.deepcopy(new_cost)
                        solution_mem, solution_dp = copy.deepcopy(new_mem_bw), copy.deepcopy(new_dp_info)

            current_temp *= self.cooling_rate
            print(f"{current_temp:.5e}\t{prev_cost:.5e}",end='\r')

        print(f"\n\nLatency: {prev_cost}.\nFinal Memory IN {list(np.array(solution_mem[0]) * self.mem_words_per_cycle)}, Memory OUT {list(np.array(solution_mem[1]) * self.mem_words_per_cycle)}.")
        print("Latency(C)={}, Latency(S)={:.6f}, GOP/s={:.2f}, volumes/s={:.2f}, DSP(%)={:.2f}, BRAM(%)={:.2f}, rateIn={}, RateOut={}, Depth={}, Muls={}, Adds={}, Mem(W)={}, Mem(KB)={}, MemBoundIn={}, MemBoundOut={}\nPartition Configuration: {}".format(solution_dp['latency(C)'], solution_dp['latency(S)'], solution_dp['GOP/s'], solution_dp['vols/s'], solution_dp['DSP'], solution_dp['BRAM'], solution_dp['rateIn'], solution_dp['rateOut'], solution_dp['depth'], solution_dp['muls'], solution_dp['adds'], solution_dp['memWords'], solution_dp['memKBs'], solution_dp['memBoundedIn'], solution_dp['memBoundedOut'], solution_dp['config']))
        print("*"*40)

    def get_cost_layer(self, config, mem_bw, layer):
        hw = self.graph.nodes[layer]['hw']
        if isinstance(hw, GAPLayer):
            dp_info = hw.get_design_point(config[0], config[1], hw.mem_words_per_cycle*mem_bw[0][0], hw.mem_words_per_cycle*mem_bw[1][0])
        elif isinstance(hw, Convolutional3DLayer):
            dp_info = hw.get_design_point(config[0], config[1], config[2], hw.mem_words_per_cycle*mem_bw[0][0], hw.mem_words_per_cycle*mem_bw[1][0])
        elif isinstance(hw, ActivationLayer):
            dp_info = hw.get_design_point(config[0], hw.mem_words_per_cycle*mem_bw[0][0], hw.mem_words_per_cycle*mem_bw[1][0])
        elif isinstance(hw, ElementWiseLayer):
            dp_info = hw.get_design_point(config[0], config[1], config[2], hw.mem_words_per_cycle*mem_bw[0][0], hw.mem_words_per_cycle*mem_bw[0][1], hw.mem_words_per_cycle*mem_bw[1][0])
        elif isinstance(hw, BatchNorm3DLayer):
            dp_info = hw.get_design_point(config[0], hw.mem_words_per_cycle*mem_bw[0][0], layhwer.mem_words_per_cycle*mem_bw[1][0])
        elif isinstance(hw, SqueezeExcitationLayer):
            assert False, "Not supported layer (SqueezeExcitationLayer)"
        elif isinstance(hw, FCLayer):
            dp_info = hw.get_design_point(config[0], config[1], hw.mem_words_per_cycle*mem_bw[0][0], hw.mem_words_per_cycle*mem_bw[1][0])
        else:
            assert False, "Not supported layer"

        if dp_info['config']:
            return dp_info['latency(S)'], dp_info
        return None, None

    def generate_random_config_layer(self, l):
        config = []
        hw = self.graph.nodes[l]['hw']
        if isinstance(hw, GAPLayer):
            channels = hw.channels
            filters = hw.filters
            coarse_in_feasible = utils.get_factors(channels)
            coarse_out_feasible = utils.get_factors(filters)
            coarse_in_factor = random.choice(coarse_in_feasible)/channels
            coarse_out_factor = random.choice(coarse_out_feasible)/filters
            config = [coarse_in_factor, coarse_out_factor]
        elif isinstance(hw, Convolutional3DLayer):
            channels = hw.channels
            filters = hw.filters
            kernel_size = hw.kernel_shape
            coarse_in_feasible = utils.get_factors(channels)
            coarse_out_feasible = utils.get_factors(filters)
            fine_feasible = utils.get_fine_feasible(kernel_size)
            coarse_in_factor = random.choice(coarse_in_feasible)/channels
            coarse_out_factor = random.choice(coarse_out_feasible)/filters
            fine_factor = random.choice(fine_feasible)/np.prod(np.array(kernel_size))
            config = [coarse_in_factor, coarse_out_factor, fine_factor]
        elif isinstance(hw, ActivationLayer):
            channels = hw.channels
            coarse_in_feasible = utils.get_factors(channels)
            coarse_in_factor = random.choice(coarse_in_feasible)/channels
            config = [coarse_in_factor]
        elif isinstance(hw, ElementWiseLayer):
            channels_1 = hw.channels_1
            depth_1 = hw.depth_in_1
            channels_2 = hw.channels_2
            depth_2 = hw.depth_in_2
            filters = hw.filters
            depth_out = hw.depth_out
            coarse_in1_feasible = utils.get_conbinations(utils.get_factors(channels_1), utils.get_factors(depth_1))
            coarse_in2_feasible = utils.get_conbinations(utils.get_factors(channels_2), utils.get_factors(depth_2))
            coarse_out_feasible = utils.get_conbinations(utils.get_factors(filters), utils.get_factors(depth_out))
            coarse_in1_factor = random.choice(coarse_in1_feasible)/(channels_1*depth_1)
            coarse_in2_factor = random.choice(coarse_in2_feasible)/(channels_2*depth_2)
            coarse_out_factor = random.choice(coarse_out_feasible)/(filters*depth_out)
            config = [coarse_in1_factor, coarse_in2_factor, coarse_out_factor]
        elif isinstance(hw, BatchNorm3DLayer):
            channels = hw.channels
            coarse_in_feasible = utils.get_factors(channels)
            coarse_in_factor = random.choice(coarse_in_feasible)/channels
            config = [coarse_in_factor, coarse_out_factor]
        elif isinstance(hw, SqueezeExcitationLayer):
            assert False, "Not supported layer (SqueezeExcitationLayer)"
        elif isinstance(hw, FCLayer):
            dim_in = hw.dim_in
            dim_out = hw.dim_out
            coarse_in_feasible = utils.get_factors(dim_in)
            coarse_out_feasible = utils.get_factors(dim_out)
            coarse_in_factor = random.choice(coarse_in_feasible)/dim_in
            coarse_out_factor = random.choice(coarse_out_feasible)/dim_out
            config = [coarse_in_factor, coarse_out_factor]
        else:
            assert False, "Not supported layer"

        if isinstance(hw, ElementWiseLayer):
            mem_config_in, mem_config_out = self.get_mem_bw_feasible(n_in=2, n_out=1)
        else:
            mem_config_in, mem_config_out = self.get_mem_bw_feasible(n_in=1, n_out=1)

        return config, [mem_config_in, mem_config_out]
