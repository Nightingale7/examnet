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
from stgem.selector.mab import MABSelector
from stgem.sut.matlab.sut import Matlab_Simulink

from pm.algorithm_parameters import *

from discriminator_score_step import ScoreStep

def get_sut(mode=None):
    # The input range is given as [50,90] but actually the input is always
    # rounded to the nearest integer (this integer signal is the output LRL
    # signal). The range for PERIOD is unknown.
    sut_parameters = {"type": "simulink",
                      "model_file": "pm/Model1_Scenario1_Correct",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["LRL"],
                      "outputs": ["PERIOD", "LRL", "PACE_COUNT"],
                      "input_range": [[50, 90]],
                      "output_range": [[0, 1000], [50, 90], [0, 16]],
                      "simulation_time": 10,
                      "time_slices": [2],
                      "sampling_step": 0.01
                     }

    return Matlab_Simulink(sut_parameters)

def build_specification(selected_specification):
    if selected_specification == "PM":
        specification = "always[0,10](PACE_COUNT <= 15) and eventually[0,10](PACE_COUNT >= 8)"

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
        model_parameters = ogan_model_parameters["convolution"]
        if N_objectives > 1:
            # Use MAB selector for multiple objectives.
            algorithms = [OGAN(models=OGAN_Model(model_parameters), parameters=ogan_parameters) for _ in range(N_objectives)]
            algorithm1 = HyperHeuristic(
                algorithms=algorithms,
                training_selector=MABSelector({"warm_up": 30})
            )
        else:
            algorithm1 = OGAN(models=OGAN_Model(model_parameters), parameters=ogan_parameters)

        step_2 = Search(mode=mode,
                        budget_threshold={"executions": 1500},
                        algorithm=HyperHeuristic(
                            algorithms=[
                                algorithm1,
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
                               algorithm=OGAN(models=OGAN_Model(ogan_model_parameters["convolution"]), parameters=ogan_parameters)
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
                            algorithm=OGAN(models=OGAN_Model(ogan_model_parameters["convolution"]), parameters=ogan_parameters)
                           )

    if setup == "random" or setup.lower().startswith("examnet"):
        steps = [step_1]
    else:
        steps = [step_1, step_2]
    return steps

def get_step_factory():
    return step_factory

