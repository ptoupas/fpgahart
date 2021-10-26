import numpy as np
import math
import scipy.optimize as optimize
from scipy.optimize import NonlinearConstraint, Bounds
from .base_layer import BaseLayer
np.set_printoptions(precision=5, suppress=True, linewidth=150)
np.seterr(divide='ignore', invalid='ignore')

DEBUG=False

class FCLayer(BaseLayer):
    def __init__(self, description, optimization):
        super().__init__()

        self.optimization = optimization
        self.input_shape = description['shape_in'][0]
        self.output_shape = description['shape_out']