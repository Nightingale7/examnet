import os, sys
from algorithms.examnet.algorithm import Examnet
from algorithms.examnet.model import Examnet_Model

from stgem.algorithm.hyperheuristic.algorithm import HyperHeuristic
from stgem.algorithm.diffusion.algorithm import Diffusion
from stgem.algorithm.diffusion.model import Diffusion_Model
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS
from stgem.algorithm.wogan.algorithm import WOGAN
from stgem.algorithm.wogan.model import WOGAN_Model, WOGAN_RandomModel
from stgem.generator import Search
from stgem.selector import WeightSelector
from stgem.sut.matlab.sut import Matlab

from discriminator_score_step import ScoreStep

from f16.algorithm_parameters import *

sys.path.append(os.path.split(os.path.dirname(__file__))[-1])
from f16_python_sut import F16GCAS_PYTHON2, F16GCAS_PYTHON3

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

    sut_parameters = {"model_file": "f16/run_f16",
                      "init_model_file": "f16/init_f16",
                      "input_type": "vector",
                      "output_type": "signal",
                      "inputs": ["ROLL", "PITCH", "YAW"],
                      "outputs": ["ALTITUDE"],
                      "input_range": [roll_range, pitch_range, yaw_range],
                      "output_range": [[0, 4040]], # Starting altitude defined in init_f16.m.
                      "initial_altitude": 4040, # Used by the Python SUTs.
                      "simulation_time": 15
                     }

    # Uncomment to use the Python SUTs.
    """
    # Python 2
    sys.path.append(os.path.split(os.path.dirname(__file__))[-1])
    from f16_python_sut import F16GCAS_PYTHON2, F16GCAS_PYTHON3
    return F16GCAS_PYTHON2(sut_parameters)
    """

    """
    # Python 3
    sys.path.append(os.path.split(os.path.dirname(__file__))[-1])
    from f16_python_sut import F16GCAS_PYTHON2, F16GCAS_PYTHON3
    return F16GCAS_PYTHON3(sut_parameters)
    """

    return Matlab(sut_parameters)

def build_specification(selected_specification):
    # Notice that here the input is a vector.
    if selected_specification == "F16":
        specification = "always[0,15] ALTITUDE > 0"

        specifications = [specification]
        strict_horizon_check = True
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return specifications, strict_horizon_check

def step_factory(N_objectives=1, setup="default"):
    mode = "stop_at_first_objective"
    #mode = "exhaust_budget"

    if setup.lower().startswith("arch23"):
        # The rationale here is that the total budget is 1500 executions, and
        # we would like to use 25% (375) of it for random search. However, we
        # want to minimize the number of executions required for a
        # falsification, so we do not want to spend all the initial 375
        # executions for random search and we want to start OGAN earlier. Thus
        # we use 75 initial random executions and then we do a random execution
        # in place of OGAN with probability 0.21. This amounts, on average, to
        # 375 executions during 1500 total executions.
        step_1 = Search(mode=mode,
                        budget_threshold={"executions": 75},
                        algorithm=Random(models=LHS({"samples": 75}))
                       )      
        step_2 = Search(mode=mode,
                        budget_threshold={"executions": 1500},
                        algorithm=HyperHeuristic(
                            algorithms=[
                                OGAN(models=OGAN_Model(ogan_model_parameters["dense"]), parameters=ogan_parameters),
                                Random(models=Uniform())
                            ],
                            generation_selector=WeightSelector([1 - 0.21, 0.21])
                        ),
                        results_include_models=False
                       )
    elif setup.lower().startswith("wogan"):
        mode = "stop_at_first_objective"
        model_class = WOGAN_Model
        if setup.lower().endswith("perfect"):
            wogan_model_parameters["analyzer"] = "PerfectAnalyzer"
        if setup.lower().endswith("random_sampler"):
            wogan_parameters["sampler_id"] = "Random_Sampler"
        if setup.lower().endswith("random_analyzer"):
            wogan_model_parameters["analyzer"] = "RandomAnalyzer"
        if setup.lower().endswith("random_wgan"):
            model_class = WOGAN_RandomModel

        # See the above for the rationale for the selector.
        step_1 = Search(mode=mode,
                        budget_threshold={"executions": 75},
                        algorithm=Random(models=Uniform())
                       )      
        step_2 = Search(mode=mode,
                        budget_threshold={"executions": 1500},
                        algorithm=HyperHeuristic(
                            algorithms=[
                                WOGAN(models=model_class(wogan_model_parameters), parameters=wogan_parameters),
                                Random(models=Uniform())
                            ],
                            generation_selector=WeightSelector([1 - 0.21, 0.21])
                        ),
                        results_include_models=False
                       )
    elif setup.lower() == "examnet":
        # See the above for the rationale for the selector.
        step_1 = Search(mode=mode,
                        budget_threshold={"executions": 1500},
                        algorithm=Examnet(models=Examnet_Model()),
                        results_include_models=False
                       )
    elif setup.lower().startswith("diffusion_fals"):
        mode = "stop_at_first_objective"
        random_budget = 50
        total_budget = 1500
        random_model = LHS({"samples": random_budget}) if setup.lower().endswith("lhs") else Uniform()
        step_1 = Search(mode=mode,
                        budget_threshold={"executions": random_budget},
                        algorithm=Random(models=random_model)
                       )
        step_2 = Search(mode=mode,
                        budget_threshold={"executions": total_budget},
                        algorithm=Diffusion(models=Diffusion_Model(diffusion_model_parameters), parameters=diffusion_parameters),
                        results_include_models=False
                       )
    else:
        if setup == "random_wogan":
            mode = "exhaust_budget"
            random_search_budget = 1500
            total_budget = 1500
        elif setup == "random":
            random_search_budget = 300
            total_budget = 300
        else:
            random_search_budget = 75
            total_budget = 300

        if setup.startswith("discriminator_score"):
            if setup == "discriminator_score":
                random_mode = "uniform"
                model = Uniform()
            else:
                random_mode = "lhs"
                model = LHS({"samples": total_budget})
            step_1 = Search(mode=mode,
                            budget_threshold={"executions": random_search_budget},
                            algorithm=Random(models=model)
                           )
            step_2 = ScoreStep(mode=mode,
                               random_mode=random_mode,
                               budget_threshold={"executions": total_budget},
                               algorithm=OGAN(models=OGAN_Model(ogan_model_parameters["dense"]), parameters=ogan_parameters)
                              )
        else:
            if setup == "default" or setup == "wogan_random":
                model = Uniform()
            else:
                model = LHS({"samples": random_search_budget})
            step_1 = Search(mode=mode,
                            budget_threshold={"executions": random_search_budget},
                            algorithm=Random(models=model)
                           )
            step_2 = Search(mode=mode,
                            budget_threshold={"executions": total_budget},
                            algorithm=OGAN(models=OGAN_Model(ogan_model_parameters["dense"]), parameters=ogan_parameters)
                           )

    if setup == "random" or setup.lower().startswith("examnet"):
        steps = [step_1]
    else:
        steps = [step_1, step_2]
    return steps

def get_step_factory():
    return step_factory

