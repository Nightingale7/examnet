import os, sys

from stgem.algorithm.hyperheuristic.algorithm import HyperHeuristic
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS
from stgem.algorithm.wogan.algorithm import WOGAN
from stgem.algorithm.wogan.model import WOGAN_Model
from stgem.generator import Search
from stgem.selector import WeightSelector
from stgem.sut.matlab.sut import Matlab

from f16.algorithm_parameters import *

sys.path.append(os.path.split(os.path.dirname(__file__)))
from f16.benchmark import build_specification, step_factory, get_step_factory

def get_sut(mode=None):
    from math import pi

    # TODO: Use the mode argument to select the ranges.
    # ARCH-COMP
    roll_range = [0.2*pi, 0.2833*pi]
    pitch_range = [-0.4*pi, -0.35*pi]
    yaw_range = [-0.375*pi, -0.125*pi]
    # PART-X
    """
    roll_range = [0.2*pi, 0.2833*pi]
    pitch_range = [-0.5*pi, -0.54*pi]
    yaw_range = [0.25*pi, 0.375*pi]
    """
    # FULL
    """
    roll_range = [-pi, pi]
    pitch_range = [-pi, pi]
    yaw_range = [-pi, pi]
    """

    sut_parameters = {"model_file": "f16n/run_f16",
                      "init_model_file": "f16n/init_f16",
                      "input_type": "vector",
                      "output_type": "signal",
                      "inputs": ["ROLL", "PITCH", "YAW"],
                      "outputs": ["ALTITUDE"],
                      "input_range": [roll_range, pitch_range, yaw_range],
                      "output_range": [[0, 2338]], # Starting altitude defined in init_f16.m.
                      "simulation_time": 15
                     }

    return Matlab(sut_parameters)

