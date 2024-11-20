import numpy as np

from algorithms.examnet.algorithm import Examnet
from algorithms.examnet.model import Examnet_Model


from stgem.algorithm import Model
from stgem.algorithm.hyperheuristic.algorithm import HyperHeuristic
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.wogan.algorithm import WOGAN
from stgem.algorithm.wogan.model import WOGAN_Model

from stgem.exceptions import GenerationException
from stgem.generator import STGEM, Search
from stgem.selector.iterative import IterativeSelector


class FalsificationTask:

    def __init__(self, problem):
        self.problem = problem

    def get_description(self):
        return self.__class__.__name__

    def get_steps(self):
        raise NotImplementedError

    def extract_models(self, result):
        raise NotImplementedError

    def run_replica(self, seed, models=None, use_gpu=True, silent=False, prev_tests=None):
        """Attempts to solve the falsification task with the given seed.
        Returns the number of executed tests."""
        generator = STGEM(
            description="{} {}".format(self.get_description(), self.problem.get_description()),
            sut=self.problem.get_sut(),
            objectives=self.problem.get_objectives(),
            steps=self.get_steps(),
        )

        #generator.setup(seed=seed, use_gpu=use_gpu, prev_tests=prev_tests)
        generator.setup(seed=seed, use_gpu=use_gpu)
        # Edit the models if needed.
        if models is not None:
            #print("Reusing models...")
            for i, step in enumerate(generator.steps):
                if models[i] is None: continue
                step.algorithm.transfer_model(models[i])
        else:
            #print("Not reusing models...")
            pass

        try:
            result = generator._run(silent=silent)
        except GenerationException:
            # We take this as an indication that the mutation has caused the
            # test generation algorithm to behave erratically meaning that the
            # SUT is somehow strange. This can happen for instance with the
            # AmbieGen problem and WOGAN: WOGAN cannot learn to produce valid
            # roads.
            return None

        return result
        
    def update_task(self, prev_task=None):
    	pass	

class Random_FalsificationTask(FalsificationTask):
    """A falsification task that uses uniform random search for falsification."""

    step_parameters = {
        "mode": "stop_at_first_objective",
        "max_executions": 500
    }

    def get_steps(self):
        model = Uniform()
        step1 = Search(mode=self.step_parameters["mode"],
                       budget_threshold={"executions": self.step_parameters["max_executions"]},
                       algorithm=Random(models=model)
        )
        steps = [step1]
        return steps

    def extract_models(self, result):
        model1_skeleton = result.step_results[0].parameters["model"]
        return [model1_skeleton]

class OGAN_FalsificationTask(FalsificationTask):
    """A falsification task that uses FOGAN for falsification."""

    step_parameters = {
        "mode": "stop_at_first_objective",
        "max_executions": 500
    }

    algorithm_parameters = {
        "fitness_coef": 0.95,
        "train_delay": 1,
        "N_candidate_tests": 1,
        "reset_each_training": False
    }

    model_parameters = {
        "convolution": {
            "optimizer": "Adam",
            "discriminator_lr": 0.001,
            "discriminator_betas": [0.9, 0.999],
            "generator_lr": 0.0001,
            "generator_betas": [0.9, 0.999],
            "noise_batch_size": 8192,
            "generator_loss": "MSE,Logit",
            "discriminator_loss": "MSE,Logit",
            "generator_mlm": "GeneratorNetwork",
            "generator_mlm_parameters": {
                "noise_dim": 20,
                "hidden_neurons": [128,128,128],
                "hidden_activation": "leaky_relu"
            },
            "discriminator_mlm": "DiscriminatorNetwork1dConv",
            "discriminator_mlm_parameters": {
                "feature_maps": [16, 16],
                "kernel_sizes": [[2,2], [2,2]],
                "convolution_activation": "leaky_relu",
                "dense_neurons": 128
            },
            "train_settings_init": {"epochs": 1, "discriminator_epochs": 15, "generator_batch_size": 32},
            "train_settings": {"epochs": 1, "discriminator_epochs": 15, "generator_batch_size": 32}
        },
        "dense": {
            "optimizer": "Adam",
            "discriminator_lr": 0.001,
            "discriminator_betas": [0.9, 0.999],
            "generator_lr": 0.0001,
            "generator_betas": [0.9, 0.999],
            "noise_batch_size": 8192,
            "generator_loss": "MSE,Logit",
            "discriminator_loss": "MSE,Logit",
            "generator_mlm": "GeneratorNetwork",
            "generator_mlm_parameters": {
                "noise_dim": 20,
                "hidden_neurons": [128,128,128],
                "hidden_activation": "leaky_relu"
            },
            "discriminator_mlm": "DiscriminatorNetwork",
            "discriminator_mlm_parameters": {
                "hidden_neurons": [128,128,128],
                "hidden_activation": "leaky_relu"
            },
            "train_settings_init": {"epochs": 1, "discriminator_epochs": 15, "generator_batch_size": 32},
            "train_settings": {"epochs": 1, "discriminator_epochs": 15, "generator_batch_size": 32}
        }
    }

    def _get_random_budget(self):
        return 40#int(0.1*self.step_parameters["max_executions"])

    def _get_model_parameters(self):
        return self.model_parameters["convolution"]

    def get_steps(self):
        model1 = Uniform()
        model2 = OGAN_Model(self._get_model_parameters())
        step1 = Search(mode=self.step_parameters["mode"],
                       budget_threshold={"executions": self._get_random_budget()},
                       algorithm=Random(models=[model1])
        )
        step2 = Search(mode=self.step_parameters["mode"],
                       budget_threshold={"executions": self.step_parameters["max_executions"]},
                       algorithm=OGAN(models=[model2], parameters=self.algorithm_parameters)
        )
        steps = [step1, step2]
        return steps

    # Legacy function, I think it's used but not applied right now
    def extract_models(self, result):
        model1_skeleton = result.step_results[0].parameters["model"]
        model2_skeleton = result.step_results[1].parameters["model"]
        return [model1_skeleton, model2_skeleton]

class Examnet_FalsificationTask(FalsificationTask):
    """A falsification task that uses Diffusion for falsification."""
    step_parameters = {
        "mode": "stop_at_first_objective",
        "max_executions": 500
    }

    def get_steps(self):        
        model = Examnet_Model()
        
        step_1 = Search(mode=self.step_parameters["mode"],
                       budget_threshold={"executions": self.step_parameters["max_executions"]},
                       algorithm=Examnet(models=[model])
        )
        steps = [step_1]
        return steps

    # Legacy function, I think it's used but not applied right now
    def extract_models(self, result):
        model1_skeleton = result.step_results[0].parameters["model"]
        model2_skeleton = result.step_results[1].parameters["model"]
        return [model1_skeleton, model2_skeleton]

class WOGAN_FalsificationTask(FalsificationTask):
    """A falsification task that uses WOGAN for falsification."""

    step_parameters = {
        "mode": "stop_at_first_objective",
        "max_executions": 500
    }

    algorithm_parameters = None
    init_alg_params = {
        "bins": 10,
        "wgan_batch_size": 32,
        "fitness_coef": 0.95,
        "train_delay": 3,
        "N_candidate_tests": 1,
        "shift_function": "linear",
        "shift_function_parameters": {"initial": 0, "final": 3},
    }

    model_parameters = {
        "critic_optimizer": "Adam",
        "critic_lr": 0.00005,
        "critic_betas": [0, 0.9],
        "generator_optimizer": "Adam",
        "generator_lr": 0.00005,
        "generator_betas": [0, 0.9],
        "noise_batch_size": 32,
        "gp_coefficient": 10,
        "eps": 1e-6,
        "report_wd": True,
        "analyzer": "Analyzer_NN",
        "analyzer_parameters": {
            "optimizer": "Adam",
            "lr": 0.001,
            "betas": [0, 0.9],
            "loss": "MSE,logit",
            "l2_regularization_coef": 0.01,
            "analyzer_mlm": "AnalyzerNetwork",
            "analyzer_mlm_parameters": {
                "hidden_neurons": [32,32],
                "layer_normalization": False
            }
        },
        "generator_mlm": "GeneratorNetwork",
        "generator_mlm_parameters": {
            "noise_dim": 10,
            "hidden_neurons": [128, 128],
            "hidden_activation": "relu",
            "batch_normalization": True,
            "layer_normalization": False
        },
        "critic_mlm": "CriticNetwork",
        "critic_mlm_parameters": {
            "hidden_neurons": [128, 128],
            "hidden_activation": "leaky_relu",
        },
        "train_settings_init": {
            "epochs": 3,
            "analyzer_epochs": 20,
            "critic_steps": 5,
            "generator_steps": 1
        },
        "train_settings": {
            "epochs": 2,
            "analyzer_epochs": 10,
            "critic_steps": 5,
            "generator_steps": 1
        },
    }

    def _get_random_budget(self):
        return 75#int(0.02*self.step_parameters["max_executions"])

    def _get_model_parameters(self):
        return self.model_parameters

    def get_steps(self):
        model1 = Uniform()
        model2 = WOGAN_Model(self._get_model_parameters())
        step1 = Search(mode=self.step_parameters["mode"],
                       budget_threshold={"executions": self._get_random_budget()},
                       algorithm=Random(models=model1)
        )
        step2 = Search(mode=self.step_parameters["mode"],
                       budget_threshold={"executions": self.step_parameters["max_executions"]},
                       algorithm=WOGAN(models=model2, parameters=self.algorithm_parameters)
        )
        steps = [step1, step2]
        return steps

    def extract_models(self, result):
        model1_skeleton = result.step_results[0].parameters["model"]
        model2_skeleton = result.step_results[1].parameters["model"]
        return [model1_skeleton, model2_skeleton]


    

