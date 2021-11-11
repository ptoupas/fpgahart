from ..layers.convolutional_3d import Convolutional3DLayer
from ..layers.batchnorm_3d import BatchNorm3DLayer
from ..layers.squeeze_excitation import SqueezeExcitationLayer
from ..layers.gap import GAPLayer
from ..layers.elemwise import ElementWiseLayer
from ..layers.activation import ActivationLayer
from ..layers.fully_connected import FCLayer
from ..layers.base_layer import BaseLayer
import random

class SimulatedAnnealing():
  def __init__(self, partition):
    self.partition = partition

    # Simulate Annealing Variables
    # self.T          = T
    # self.k          = k
    # self.T_min      = T_min
    # self.cool       = cool
    # self.iterations = iterations

  def run_optimizer(self):
    config = self.generate_random_config()

    cost = self.get_cost(config)


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
                       'coarse_in': coarse_in_factor,
                       'coarse_out': coarse_out_factor,
                       'fine': fine_factor}
      elif isinstance(l, ActivationLayer):
          channels = l.channels
          coarse_in_feasible = self.get_factors(channels)
          coarse_in_factor = random.choice(coarse_in_feasible)
      elif isinstance(l, ElementWiseLayer):
          #TODO: Check this how to deal when the input comes from another layer and not from off-chip mem
          channels_1 = l.channels_1
          channels_2 = l.channels_2
          filters = l.filters
          coarse_in1_feasible = self.get_factors(channels_1)
          coarse_in2_feasible = self.get_factors(channels_2)
          coarse_out_feasible = self.get_factors(filters)
          coarse_in1_factor = random.choice(coarse_in1_feasible)
          coarse_in2_factor = random.choice(coarse_in2_feasible)
          coarse_out_factor = random.choice(coarse_out_feasible)
      elif isinstance(l, BatchNorm3DLayer):
          channels = l.channels
          coarse_in_feasible = self.get_factors(channels)
          coarse_in_factor = random.choice(coarse_in_feasible)
      elif isinstance(l, SqueezeExcitationLayer):
          assert False, "Not supported layer (SqueezeExcitationLayer)"
      elif isinstance(l, FCLayer):
          assert False, "Not supported layer (FCLayer)"
      else:
          assert False, "Not supported layer"

    return config

  def get_factors(n):
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