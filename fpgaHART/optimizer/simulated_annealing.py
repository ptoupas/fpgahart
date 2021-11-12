from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.batchnorm_3d import BatchNorm3DLayer
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..layers.gap import GAPLayer
from ..layers.elemwise import ElementWiseLayer
from ..layers.activation import ActivationLayer
from ..layers.fully_connected import FCLayer
from ..layers.base_layer import BaseLayer
from ..partitions.partition_compose import PartitionComposer
import random
from functools import reduce
import numpy as np
import copy
import math
import configparser
import os

class SimulatedAnnealing():
    def __init__(self, partition, t_min=0.0001, t_max=25, iterationPerTemp=20, cooling_rate=0.99):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), 'fpgaHART', 'config', 'config_fpga.ini'))
        self.word_length = int(config.get('FPGA Specifications', 'word_length'))
        self.clock_freq = int(config.get('FPGA Specifications', 'clock_freq'))
        self.mem_bw = float(config.get('FPGA Specifications', 'mem_bw')) * 1e9
        self.cycles_per_sec = self.clock_freq*1e6
        self.mem_words_per_cycle = (self.mem_bw / self.word_length) / self.cycles_per_sec

        self.partition_composer = PartitionComposer()
        self.partition = partition

        # Simulate Annealing Variables
        # self.T          = T
        # self.k          = k
        self.t_min      = t_min
        self.t_max      = t_max
        self.cooling_rate       = cooling_rate
        self.iterationPerTemp   = iterationPerTemp

    def run_optimizer(self):
        config, mem_bw = self.generate_random_config()
        cost = self.get_cost(config, mem_bw)

        if cost is None:
            # assert False, "No feasible design point found"
            while cost is None:
                config, mem_bw = self.generate_random_config()
                cost = self.get_cost(config, mem_bw)
        prev_state = config
        solution = config
        prev_cost = cost

        current_temp = self.t_max

        print(f"Temperature  |  Latency")
        while current_temp > self.t_min:
            
            for i in range(self.iterationPerTemp):
                # TODO: Make this configuration generation more efficient in a means of producing configuration that are close to the best configuration so far (i.e. not just random but close to the neighbor of the best configuration)
                new_state, new_mem_bw = self.generate_random_config()
                new_cost = self.get_cost(new_state, new_mem_bw)

                if new_cost is None:
                    continue
                    # assert False, "No feasible design point found"

                cost_diff = prev_cost - new_cost
                if cost_diff >= 0:
                    prev_state = copy.deepcopy(new_state)
                    prev_cost = copy.deepcopy(new_cost)
                    solution, mem_solution = copy.deepcopy(new_mem_bw), copy.deepcopy(new_state)
                else:
                    # print(f"Probability = {math.exp((cost_diff/current_temp))} of {(cost_diff/current_temp)}")
                    if random.uniform(0, 1) < math.exp((cost_diff/current_temp)):
                        prev_state = copy.deepcopy(new_state)
                        prev_cost = copy.deepcopy(new_cost)
                        solution, mem_solution = copy.deepcopy(new_mem_bw), copy.deepcopy(new_state)

            current_temp *= self.cooling_rate
            print(f"{current_temp:.5e}\t{prev_cost:.5e}",end='\r')

        print(f"\n\nFinal Solution: {solution} - {mem_solution}. Latency: {prev_cost}")

    def get_cost(self, config, mem_bw):
        comb_config = list()
        for k, v in config.items():
            if v['op_type'] == 'GlobalAveragePool':
                comb_config.append([v['coarse_in'], v['coarse_out']])
            elif v['op_type'] == 'Conv':
                comb_config.append([v['fine'], v['coarse_in'], v['coarse_out']])
            elif v['op_type'] == 'Activation':
                comb_config.append([v['coarse_in']])
            elif v['op_type'] == 'ElementWise':
                comb_config.append([v['coarse_in_1'], v['coarse_in_2'], v['coarse_out']])
            elif v['op_type'] == 'BatchNormalization':
                comb_config.append([v['coarse_in']])
            else:
                assert False, "Not supported layer"

        dp_info = self.partition_composer.get_design_point(self.partition, comb_config, mem_bw[0], mem_bw[1], mem_bw[2])
        if dp_info['config']:
            return dp_info['latency(S)']
        return None

    def get_mem_bw_feasible(self):
        first = random.randint(1,100)
        second = random.randint(1,100)
        third = random.randint(1,100)

        total_sum = first + second + third

        mem_in_1_perc = first / total_sum
        mem_in_2_perc = second / total_sum
        mem_out_perc = third / total_sum

        assert math.isclose(mem_in_1_perc + mem_in_2_perc + mem_out_perc, 1.0), "Sum of mem_in_1_perc, mem_in_2_perc and mem_out_perc should be 1"

        return mem_in_1_perc, mem_in_2_perc, mem_out_perc

    def generate_random_config(self):
        config = {}
        for k, l in self.partition.items():
            if isinstance(l, GAPLayer):
                channels = l.channels
                filters = l.filters
                coarse_in_feasible = self.get_factors(channels)
                coarse_out_feasible = self.get_factors(filters)
                coarse_in_factor = random.choice(coarse_in_feasible)/channels
                coarse_out_factor = random.choice(coarse_out_feasible)/filters
                config[k] = {'op_type': 'GlobalAveragePool',
                            'coarse_in': coarse_in_factor,
                            'coarse_out': coarse_out_factor}
            elif isinstance(l, Convolutional3DLayer):
                channels = l.channels
                filters = l.filters
                kernel_size = l.kernel_shape
                coarse_in_feasible = self.get_factors(channels)
                coarse_out_feasible = self.get_factors(filters)
                fine_feasible = self.get_fine_feasible(kernel_size)
                coarse_in_factor = random.choice(coarse_in_feasible)/channels
                coarse_out_factor = random.choice(coarse_out_feasible)/filters
                fine_factor = random.choice(fine_feasible)/np.prod(np.array(kernel_size))
                config[k] = {'op_type': 'Conv',
                            'fine': fine_factor,
                            'coarse_in': coarse_in_factor,
                            'coarse_out': coarse_out_factor}
            elif isinstance(l, ActivationLayer):
                channels = l.channels
                coarse_in_feasible = self.get_factors(channels)
                coarse_in_factor = random.choice(coarse_in_feasible)/channels
                config[k] = {'op_type': 'Activation',
                    'coarse_in': coarse_in_factor}
            elif isinstance(l, ElementWiseLayer):
                #TODO: Check this how to deal when the input comes from another layer and not from off-chip mem
                channels_1 = l.channels_1
                channels_2 = l.channels_2
                filters = l.filters
                coarse_in1_feasible = self.get_factors(channels_1)
                coarse_in2_feasible = self.get_factors(channels_2)
                coarse_out_feasible = self.get_factors(filters)
                coarse_in1_factor = random.choice(coarse_in1_feasible)/channels_1
                coarse_in2_factor = random.choice(coarse_in2_feasible)/channels_2
                coarse_out_factor = random.choice(coarse_out_feasible)/filters
                config[k] = {'op_type': 'ElementWise',
                            'coarse_in_1': coarse_in1_factor,
                            'coarse_in_2': coarse_in2_factor,
                            'coarse_out': coarse_out_factor,
                            'fine': fine_factor}
            elif isinstance(l, BatchNorm3DLayer):
                channels = l.channels
                coarse_in_feasible = self.get_factors(channels)
                coarse_in_factor = random.choice(coarse_in_feasible)/channels
                config[k] = {'op_type': 'BatchNormalization',
                    'coarse_in': coarse_in_factor}
            elif isinstance(l, SqueezeExcitationLayer):
                assert False, "Not supported layer (SqueezeExcitationLayer)"
            elif isinstance(l, FCLayer):
                assert False, "Not supported layer (FCLayer)"
            else:
                assert False, "Not supported layer"

        mem_in_1_perc, mem_in_2_perc, mem_out_perc = self.get_mem_bw_feasible()

        return config, [mem_in_1_perc, mem_in_2_perc, mem_out_perc]

    def get_factors(self, n):
        """
        Parameters
        ----------
        n: int
        
        Returns
        -------
        list
            list of integers that are factors of `n`
        """
        return list(set(reduce(list.__add__, 
                    ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

    def get_fine_feasible(self, kernel_size):
        if kernel_size[0] != kernel_size[1] and kernel_size[1] == kernel_size[2]:
            if kernel_size[0] == 1:
                return [1, kernel_size[1], kernel_size[1]*kernel_size[2]]
            elif kernel_size[1] == 1:
                return [1, kernel_size[0]]
            else:
                return [1, kernel_size[0], kernel_size[1], kernel_size[0]*kernel_size[1], kernel_size[1]*kernel_size[2], kernel_size[0]*kernel_size[1]*kernel_size[2]]
        elif kernel_size[0] == kernel_size[1] and kernel_size[1] == kernel_size[2]:
            if kernel_size[0] == 1:
                return [1]
            else:
                return [1, kernel_size[0], kernel_size[0]*kernel_size[1], kernel_size[0]*kernel_size[1]*kernel_size[2]]
        else:
            return [ 1, kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[0]*kernel_size[1], kernel_size[0]*kernel_size[2], kernel_size[1]*kernel_size[2], kernel_size[0]*kernel_size[1]*kernel_size[2] ]