from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.batchnorm_3d import BatchNorm3DLayer
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..layers.gap import GAPLayer
from ..layers.elemwise import ElementWiseLayer
from ..layers.activation import ActivationLayer
from ..layers.fully_connected import FCLayer
from ..layers.base_layer import BaseLayer
from collections import deque
import numpy as np
import math

np.set_printoptions(precision=5, suppress=True, linewidth=250)
np.seterr(divide='ignore', invalid='ignore')

DEBUG=False
class PartitionComposer(BaseLayer):
    def __init__(self):
        super().__init__()

    def update_layer(self):
        self.full_rate_in_1 = 0
        self.full_rate_in_2 = 0
        self.full_rate_out = 0
        self.max_parallel_muls = 0
        self.max_parallel_adds = 0
        self.memory = 0
        self.memoryKB = 0
        self.depth = 0
        self.mem_bd_in_1 = False
        self.mem_bd_in_2 = False
        self.mem_bd_out = False
        self.config = []
        self.dsps_util = 0
        self.bram_util = 0
        self.latency_sec = 0
        self.latency_cycles = 0
        self.throughput_ops = 0
        self.throughput_vols = 0

    def get_total_workload(self, graph):
        total_wl = 0
        for node in graph.nodes:
            op_type = graph.nodes[node]['type']
            hw = graph.nodes[node]['hw']
            
            if not op_type == 'mem_in' and not op_type == 'mem_out':
                total_wl += hw.get_total_workload()

        return total_wl

    def get_dp_info(self):
        dp_info = {}

        dp_info['latency(C)'] = self.latency_cycles
        dp_info['latency(S)'] = self.latency_sec
        dp_info['GOP/s'] = self.throughput_ops*1e-9
        dp_info['vols/s'] = self.throughput_vols
        dp_info['DSP'] = self.dsps_util
        dp_info['BRAM'] = self.bram_util
        dp_info['rateIn1'] = self.full_rate_in_1
        dp_info['rateIn2'] = self.full_rate_in_2
        dp_info['rateOut'] = self.full_rate_out
        dp_info['depth'] = self.depth
        dp_info['muls'] = self.max_parallel_muls
        dp_info['adds'] = self.max_parallel_adds
        dp_info['memWords'] = self.memory
        dp_info['memKBs'] = self.memoryKB
        dp_info['memBoundedIn1'] = self.mem_bd_in_1
        dp_info['memBoundedIn2'] = self.mem_bd_in_2
        dp_info['memBoundedOut'] = self.mem_bd_out
        dp_info['config'] = self.config
        
        return dp_info

    @staticmethod
    def find_node_idx(graph, request_node):
        for n, node in enumerate(graph.nodes):
            if node == request_node:
                return n
        return -1

    def get_design_point(self, graph, comb: dict(), mem_bw_in: list(), mem_bw_out: list, read_mem_points: list, write_mem_points: list, branch_mem=0):
        assert len(mem_bw_in) == len(read_mem_points), "Input memory break points and memory configuration does not match."
        assert len(mem_bw_out) == len(write_mem_points), "Output memory break points and memory configuration does not match."

        self.update_layer()

        off_chip_mem_in = deque()
        off_chip_mem_out = deque()
        for i in range(len(mem_bw_in)):
            off_chip_mem_in.appendleft(mem_bw_in[i]*self.mem_words_per_cycle)
        for i in range(len(mem_bw_out)):
            off_chip_mem_out.appendleft(mem_bw_out[i]*self.mem_words_per_cycle)


        num_layers = graph.number_of_nodes()

        gamma_matrix = np.zeros( shape=(num_layers-1, num_layers) , dtype=float )

        graph_idx = {}
        for i, n in enumerate(graph.nodes()):
            graph_idx[n] = i

        total_muls = 0
        total_adds = 0
        total_memory = branch_mem
        total_depth = 0
        for n, node in enumerate(graph.nodes()):
            op_type = graph.nodes[node]['type']
            hw = graph.nodes[node]['hw']

            if op_type == 'mem_in':
                assert not node in comb.keys(), f"Memory IN node: {node} cannot have configuration."
                gamma_matrix[n, n] = off_chip_mem_in.pop()
                graph.nodes[node]['prod_rate'] = gamma_matrix[n, n]
                continue
            
            if op_type == 'mem_out':
                assert not node in comb.keys(), f"Memory OUT node: {node} cannot have configuration."
                gamma_matrix[n - 1, n] = -off_chip_mem_out.pop()
                curr_layer_rate = gamma_matrix[n - 1, n]
                graph.nodes[node]['cons_rate'] = gamma_matrix[n - 1, n]
                continue
            
            assert node in comb.keys(), f"Node: {node} does not have configuration. Please check the graph creation."
            c = comb[node]

            curr_layer_rate = 1000000
            node_predecessors = list(graph.predecessors(node))
            if graph.in_degree(node) == 1:
                assert len(node_predecessors) == 1, f"Node: {node} must have exactly one predecessor. Graph inconsistency."
                prev_layer_rate_1 = graph.nodes[node_predecessors[0]]['prod_rate']
                prev_layer_rate_2 = None
            elif graph.in_degree(node) == 2:
                assert len(node_predecessors) == 2, f"Node: {node} must have exactly two predecessors. Graph inconsistency."
                prev_layer_rate_1 = graph.nodes[node_predecessors[0]]['prod_rate']
                prev_layer_rate_2 = graph.nodes[node_predecessors[1]]['prod_rate']
            else:
                raise Exception(f"Node: {node} has more than 2 predecessors. This kind of connection is not yet supported.")

            if isinstance(hw, GAPLayer):
                dp_info = hw.get_design_point(coarse_in=c[0], coarse_out=c[1], mem_bw_in=prev_layer_rate_1, mem_bw_out=curr_layer_rate)
            elif isinstance(hw, Convolutional3DLayer):
                dp_info = hw.get_design_point(f_fine=c[0], f_coarseIn=c[1], f_coarseOut=c[2], mem_bw_in=prev_layer_rate_1, mem_bw_out=curr_layer_rate)
            elif isinstance(hw, ActivationLayer):
                dp_info = hw.get_design_point(coarse_inout=c[0], mem_bw_in=prev_layer_rate_1, mem_bw_out=curr_layer_rate)
            elif isinstance(hw, ElementWiseLayer):
                if hw.broadcasting:
                    prev_nodes = [pn for pn in graph.predecessors(node)]
                    prev_nodes_out_shapes = [graph.nodes[pn]['hw'].output_shape for pn in prev_nodes]
                    node_fs = prev_nodes[prev_nodes_out_shapes.index(max(prev_nodes_out_shapes))]
                    node_rs = prev_nodes[prev_nodes_out_shapes.index(min(prev_nodes_out_shapes))]
                    prev_layer_rate_1 = graph.nodes[node_fs]['prod_rate']
                    prev_layer_rate_2 = graph.nodes[node_rs]['prod_rate']
                if branch_mem == 0:
                    dp_info = hw.get_design_point(coarse_in1=c[0], coarse_in2=c[1], coarse_out=c[2], mem_bw_in_1=prev_layer_rate_1, mem_bw_in_2=prev_layer_rate_2, mem_bw_out=curr_layer_rate)
                else:
                    dp_info = hw.get_design_point(coarse_in1=c[0], coarse_in2=c[1], coarse_out=c[2], mem_bw_in_1=curr_layer_rate, mem_bw_in_2=prev_layer_rate_2, mem_bw_out=curr_layer_rate)
            elif isinstance(hw, BatchNorm3DLayer):
                dp_info = hw.get_design_point(coarse_inout=c[0], mem_bw_in=prev_layer_rate_1, mem_bw_out=curr_layer_rate)
            elif isinstance(hw, SqueezeExcitationLayer):
                dp_info = hw.get_design_point(f_gap_coarsein=c[0], f_gap_coarseout=c[1], f_fine_1=c[2], f_coarseIn_1=c[3], f_coarseOut_1=c[4], f_relu_cinout=c[5], f_fine_2=c[6], f_coarseIn_2=c[7], f_coarseOut_2=c[8], f_sigm_cinout=c[9], f_mul_coarsein1=c[10], f_mul_coarsein2=c[11], f_mul_coarseout=c[12], mem_bw_in=prev_layer_rate_1, mem_bw_out=curr_layer_rate)
            elif isinstance(hw, FCLayer):
                dp_info = hw.get_design_point(coarse_in=c[0], coarse_out=c[1], mem_bw_in=prev_layer_rate_1, mem_bw_out=curr_layer_rate)
            else:
                assert False, "Not supported layer"

            if isinstance(hw, ElementWiseLayer):
                full_rate_in_1, full_rate_in_2, full_rate_out, muls, adds, memory, depth, mem_bd_in_1, mem_bd_in_2, mem_bd_out = dp_info['rateIn1'], dp_info['rateIn2'], dp_info['rateOut'], dp_info['muls'], dp_info['adds'], dp_info['memWords'], dp_info['depth'], dp_info['memBoundedIn1'], dp_info['memBoundedIn2'], dp_info['memBoundedOut']
                if hw.broadcasting:
                    cp1 = graph_idx[node_fs]
                    cp2 = graph_idx[node_rs]
                else:
                    cp1 = graph_idx[node_predecessors[0]]
                    cp2 = graph_idx[node_predecessors[1]]
                gamma_matrix[cp1, n] = -full_rate_in_1
                gamma_matrix[cp2, n] = -full_rate_in_2
                gamma_matrix[n, n] = full_rate_out
                graph.nodes[node]['cons_rate_1'] = full_rate_in_1
                graph.nodes[node]['cons_rate_2'] = full_rate_in_2
                graph.nodes[node]['prod_rate'] = full_rate_out
            else:
                full_rate_in, full_rate_out, muls, adds, memory, depth, mem_bd_in, mem_bd_out = dp_info['rateIn1'], dp_info['rateOut'], dp_info['muls'], dp_info['adds'], dp_info['memWords'], dp_info['depth'], dp_info['memBoundedIn1'], dp_info['memBoundedOut']
                cp = graph_idx[node_predecessors[0]]
                gamma_matrix[cp, n] = -full_rate_in
                gamma_matrix[n, n] = full_rate_out
                graph.nodes[node]['cons_rate'] = full_rate_in
                graph.nodes[node]['prod_rate'] = full_rate_out

            total_muls += muls
            total_adds += adds
            total_memory += memory
            total_depth += depth

            mem_kb = (total_memory * self.word_bytes) / 1e3
            mem_bram = math.ceil(mem_kb / self.bram_Kbytes)
            curr_bram_util = (mem_bram / self.bram) * 100
            curr_dsps_util = (total_muls/self.dsp)*100

            if not dp_info['config'] or curr_dsps_util >= 90. or curr_bram_util >= 90.:
                self.update_layer()
                if DEBUG:
                    print("Discarding design point.")
                return self.get_dp_info()

        assert len(off_chip_mem_in) == 0, "Off-chip memory IN points left hanging. Wrong configuration of the graph."
        assert len(off_chip_mem_out) == 0, "Off-chip memory OUT points left hanging. Wrong configuration of the graph."

        # gamma_matrix[0, 0] = mem_bw_in_1
        # gamma_matrix[-1, -1] = -mem_bw_out

        # total_muls = 0
        # total_adds = 0
        # total_memory = branch_mem
        # total_depth = 0
        # first_layer_bw_in = False
        # prev_layer_rate = mem_bw_in_1

        # for n, node in enumerate(graph.nodes):
        #     if node in comb.keys():
        #         c = comb[node]
        #     else:
        #         print("Off chip memory node. Skipping...")
        #         continue

        #     op_type = graph.nodes[node]['type']
        #     hw = graph.nodes[node]['hw']

        #     input_node = False
        #     output_node = False
        #     if graph.in_degree[node] == 0:
        #         input_node = True
        #     if graph.out_degree[node] == 0:
        #         output_node = True

        #     if output_node:
        #         curr_layer_rate = mem_bw_out
        #     else:
        #         curr_layer_rate = 10000000

        #     if isinstance(hw, GAPLayer):
        #         dp_info = hw.get_design_point(c[0], c[1], prev_layer_rate, curr_layer_rate)
        #     elif isinstance(hw, Convolutional3DLayer):
        #         dp_info = hw.get_design_point(c[0], c[1], c[2], prev_layer_rate, curr_layer_rate)
        #     elif isinstance(hw, ActivationLayer):
        #         dp_info = hw.get_design_point(c[0], prev_layer_rate, curr_layer_rate)
        #     elif isinstance(hw, ElementWiseLayer):
        #         #TODO: Check this how to deal when the input comes from another layer and not from off-chip mem
        #         if branch_mem == 0:
        #             dp_info = hw.get_design_point(c[0], c[1], c[2], mem_bw_in_2, prev_layer_rate, curr_layer_rate)
        #         else:
        #             dp_info = hw.get_design_point(c[0], c[1], c[2], curr_layer_rate, prev_layer_rate, curr_layer_rate)
        #     elif isinstance(hw, BatchNorm3DLayer):
        #         dp_info = hw.get_design_point(c[0], prev_layer_rate, curr_layer_rate)
        #     elif isinstance(hw, SqueezeExcitationLayer):
        #         dp_info = hw.get_design_point(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], prev_layer_rate, curr_layer_rate)
        #     elif isinstance(hw, FCLayer):
        #         dp_info = hw.get_design_point(c[0], c[1], prev_layer_rate, curr_layer_rate)
        #     else:
        #         assert False, "Not supported layer"
            
        #     if isinstance(hw, ElementWiseLayer):
        #         full_rate_in_1, full_rate_in_2, full_rate_out, muls, adds, memory, depth, mem_bd_in_1, mem_bd_in_2, mem_bd_out = dp_info['rateIn1'], dp_info['rateIn2'], dp_info['rateOut'], dp_info['muls'], dp_info['adds'], dp_info['memWords'], dp_info['depth'], dp_info['memBoundedIn1'], dp_info['memBoundedIn2'], dp_info['memBoundedOut']
        #         # if graph.in_degree[node] > 1:
        #         #     predec_nodes = [n for n in graph.predecessors(node)]
        #         #     for pn in predec_nodes:
        #         #         node_idx = self.find_node_idx(graph, pn)
        #         #         if not node_idx == n-1:
        #         #             gamma_matrix[node_idx, n+1] = -full_rate_in_1
        #         gamma_matrix[n, n+1] = -full_rate_in_2
        #         gamma_matrix[n+1, n+1] = full_rate_out
        #     else:
        #         full_rate_in, full_rate_out, muls, adds, memory, depth, mem_bd_in, mem_bd_out = dp_info['rateIn1'], dp_info['rateOut'], dp_info['muls'], dp_info['adds'], dp_info['memWords'], dp_info['depth'], dp_info['memBoundedIn1'], dp_info['memBoundedOut']
        #         gamma_matrix[n, n+1] = -full_rate_in
        #         gamma_matrix[n+1, n+1] = full_rate_out

        #     prev_layer_rate = full_rate_out

        #     total_muls += muls
        #     total_adds += adds
        #     total_memory += memory
        #     total_depth += depth

        #     mem_kb = (total_memory * self.word_bytes) / 1e3
        #     mem_bram = math.ceil(mem_kb / self.bram_Kbytes)
        #     curr_bram_util = (mem_bram / self.bram) * 100
        #     curr_dsps_util = (total_muls/self.dsp)*100

        #     if not dp_info['config'] or curr_dsps_util >= 90. or curr_bram_util >= 90.:
        #         self.update_layer()
        #         if DEBUG:
        #             print("Discarding design point.")
        #         return self.get_dp_info()

        if DEBUG:
            print("Γ:\n{}".format(gamma_matrix))
        # TODO: Integrate the new matrix balancing algorithm
        # gamma_matrix_balanced, mem_bounded_in, mem_bounded_out = self.balance_matrix(gamma_matrix.copy())
        gamma_matrix_balanced = gamma_matrix.copy()
        if DEBUG:
            print("Γ Balanced:\n{}".format(gamma_matrix_balanced))
        workload_matrix = self.get_workload_matrix(graph, num_layers)
        if DEBUG:
            print("WL:\n{}".format(workload_matrix))
        ii_matrix = np.nan_to_num(workload_matrix/gamma_matrix_balanced)
        if DEBUG:
            print("II:\n{}".format(ii_matrix))

        # TODO: Properly find whether the graph is memory bounding and in which input/output (from gama matrix balancing)
        # mem_bounded_in = mem_bounded_in or first_layer_bw_in
        mem_bounded_out = False
        mem_bounded_in = False
        batch_size = 1
        latency_sec, latency_cycles, thr_in, thr_out, dsps_util, bram_util, memKBs = self.get_performance(workload_matrix, ii_matrix, total_muls, total_adds, total_memory, total_depth, batch=batch_size)
        total_ops = self.get_total_workload(graph)*batch_size
        throughput_ops = total_ops/latency_sec
        thr_in /= workload_matrix[0,0]              # Volumes per second
        thr_out /= workload_matrix[-1,-1]           # Volumes per second
        assert math.isclose(thr_in, thr_out), "Thoughputs missmatch. IN = {}, OUT = {}.".format(thr_in, thr_out)

        if dsps_util < 90. and bram_util < 90.:
            #TODO: Add 2nd input
            self.full_rate_in_1 = gamma_matrix_balanced[0, 0]
            self.full_rate_in_2 = gamma_matrix_balanced[1, 1]
            self.full_rate_out = abs(gamma_matrix_balanced[-1, -1])
            self.max_parallel_muls = total_muls
            self.max_parallel_adds = total_adds
            self.memory = total_memory
            self.depth = total_depth
            #TODO: Add 2nd input
            self.mem_bd_in_1 = mem_bounded_in
            self.mem_bd_in_2 = mem_bounded_in
            self.mem_bd_out = mem_bounded_out

            #TODO: Add 2nd input
            config = comb
            self.config = config
            self.memoryKB = memKBs
            self.dsps_util = dsps_util
            self.bram_util = bram_util
            self.latency_sec = latency_sec
            self.latency_cycles = int(latency_cycles)
            self.throughput_ops = throughput_ops
            self.throughput_vols = thr_out

            if DEBUG:
                print("GOPs/s={:.2f}, DSPS={:.2f}, BRAM={:.2f}, depth={}, latency(s)={:.2f}, latency(c)={:.2f}, mem bounded in = {}, mem bounded out = {}".format(throughput_ops*1e-9, dsps_util, bram_util, total_depth, latency_sec, int(latency_cycles), mem_bounded_in, mem_bounded_out))
        else:
            self.update_layer()
            if DEBUG:
                print("Discarding design point.")

        return self.get_dp_info()

    def get_workload_matrix(self, graph, num_layers):
        graph_idx = {}
        for i, n in enumerate(graph.nodes()):
            graph_idx[n] = i

        workload_matrix = np.zeros( shape=(num_layers-1, num_layers) , dtype=float )

        for n, node in enumerate(graph.nodes):

            op_type = graph.nodes[node]['type']
            hw = graph.nodes[node]['hw']

            if op_type == 'mem_in':
                workload_matrix[n, n] = np.prod(np.array(hw.output_shape[1:]))
                continue
            
            if op_type == 'mem_out':
                workload_matrix[n - 1, n] = np.prod(np.array(hw.input_shape[1:]))
                continue
            
            if isinstance(hw, ElementWiseLayer):
                cp1 = graph_idx[list(graph.in_edges(node))[0][0]]
                cp2 = graph_idx[list(graph.in_edges(node))[1][0]]
                workload_matrix[cp1, n] = np.prod(np.array(hw.input_shape_1[1:]))
                workload_matrix[cp2, n] = np.prod(np.array(hw.input_shape_2[1:]))
                workload_matrix[n, n] = np.prod(np.array(hw.output_shape[1:]))
            else:
                cp = graph_idx[list(graph.in_edges(node))[0][0]]
                workload_matrix[cp, n] = np.prod(np.array(hw.input_shape[1:]))
                workload_matrix[n, n] = np.prod(np.array(hw.output_shape[1:]))

            # input_node = False
            # output_node = False
            # if graph.in_degree[node] == 0:
            #     input_node = True
            # if graph.out_degree[node] == 0:
            #     output_node = True

            # op_type = graph.nodes[node]['type']
            # hw = graph.nodes[node]['hw']

            # if input_node:
            #     workload_matrix[0, 0] = np.prod(np.array(hw.input_shape[1:]))
            # if output_node:
            #     workload_matrix[-1, -1] = np.prod(np.array(hw.output_shape[1:]))
            
            # if isinstance(hw, ElementWiseLayer):
            #     # if graph.in_degree[node] > 1:
            #     #     predec_nodes = [n for n in graph.predecessors(node)]
            #     #     for pn in predec_nodes:
            #     #         node_idx = self.find_node_idx(graph, pn)
            #     #         if not node_idx == n-1:
            #     #             workload_matrix[node_idx, n+1] = np.prod(np.array(hw.input_shape_1[1:]))
            #     workload_matrix[n, n+1] = np.prod(np.array(hw.input_shape_2[1:]))
            # else:
            #     workload_matrix[n, n+1] = np.prod(np.array(hw.input_shape[1:]))
            # workload_matrix[n+1, n+1] = np.prod(np.array(hw.output_shape[1:]))

        return workload_matrix

    def get_performance(self, workload_matrix, ii, muls, adds, mem, depth, batch=1):
        mem_kb = (mem * self.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.bram_Kbytes)
        bram_util = (mem_bram / self.bram) * 100

        dsps_util = (muls/self.dsp)*100

        latency_cycles = np.max(np.abs(ii))*batch + depth
        latency_sec = latency_cycles/self.cycles_per_sec

        thr_in = (batch*workload_matrix[0,0])/latency_sec       # Input words per second
        thr_out = (batch*workload_matrix[-1,-1])/latency_sec    # Output words per second

        return latency_sec, latency_cycles, thr_in, thr_out, dsps_util, bram_util, mem_kb