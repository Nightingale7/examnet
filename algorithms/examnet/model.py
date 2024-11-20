import copy, importlib

import numpy as np
import torch
import math

from stgem import algorithm
from stgem.algorithm import Model, ModelSkeleton
from stgem.exceptions import AlgorithmException
import random

class Examnet_ModelSkeleton(ModelSkeleton):

    def __init__(self, parameters):
        super().__init__(parameters)
        self.modelM = None
        self.modelD = None
        self.modelV = None

    def to_model(self):
        return Examnet_Model.from_skeleton(self)

    def _mutate_tests(self, tests, count=1, device=None):
        modelM = self.modelM
        
        if modelM is None:
            raise Exception("No machine learning models available. Has the model been setup correctly?")

        if count < 0:
            raise ValueError("The number of tests should be positive.")

        training_M = modelM.training
        modelM.train(False)
        result = tests.to(device)
        
        for i in range(count):
            result = modelM(result)
        
        if torch.any(torch.isinf(result)) or torch.any(torch.isnan(result)):
            raise AlgorithmException("Mutator produced a test with inf or NaN entries.")

        modelM.train(training_M)
        return result.cpu().detach().numpy()

    def mutate_tests(self, tests, count=1, device=None):
        """Generate N random tests.

        Args:
          N (int):      Number of tests to be generated.
          device (obj): CUDA device or None.

        Returns:
          output (np.ndarray): Array of shape (N, self.input_ndimension).

        Raises:
        """
        
        try:
            return self._mutate_tests(tests, count, device)
        except:
            raise

    def _predict_objective(self, test, device=None):
        if self.modelD is None:
            raise Exception("No machine learning models available. Has the model been setup correctly?")

        test_tensor = torch.from_numpy(test).float().to(device)
        return self.modelD(test_tensor).cpu().detach().numpy().reshape(-1)

    def predict_objective(self, test, device=None):
        """Predicts the objective function value of the given tests.

        Args:
          test (np.ndarray): Array of shape (N, self.input_ndimension).
          device (obj):      CUDA device or None.

        Returns:
          output (np.ndarray): Array of shape (N, 1).

        Raises:
        """

        try:
            return self._predict_objective(test, device)
        except:
            raise

class Examnet_Model(Model,Examnet_ModelSkeleton):
    """Implements the OGAN model."""
    
    default_parameters = {
        "optimizer": "Adam",
        "discriminator_lr": 0.001,
        "discriminator_betas": [0.9, 0.999],
        "discriminator_loss": "MSE,Logit",
        "mutator_lr": 0.0001,
        "mutator_betas": [0.9, 0.999],
        "mutator_loss": "mse",
        "validator_lr": 0.0001,
        "validator_betas": [0.9, 0.999],
        "validator_loss": "bce",
        "mutator_mlm": "MutatorNetwork",
        "mutator_mlm_parameters": {
            "hidden_neurons": [128,128,128],
            "hidden_activation": "leaky_relu"
        },
        "validator_mlm": "ValidatorNetwork",
        "validator_mlm_parameters": {
            "hidden_neurons": [128,128,128],
            "hidden_activation": "leaky_relu"
        },
        "discriminator_mlm": "DiscriminatorNetwork",
        "discriminator_mlm_parameters": {
            "hidden_neurons": [128,128,128,128],
            "hidden_activation": "leaky_relu"
        },
        "train_settings_init": {
            "epochs": 1,
            "discriminator_epochs": 24,
            "validator_epochs": 8,
            "mutator_epochs": 24,            
        },
        "train_settings": {
            "epochs": 1,
            "discriminator_epochs": 16,
            "validator_epochs": 8,
            "mutator_epochs": 16,
        },
        "mutator_loss_type": "msrel",
    }

    def __init__(self, parameters=None):
        Model.__init__(self, parameters)
        Examnet_ModelSkeleton.__init__(self, parameters)
        self.modelM = None
        self.modelD = None
        self.modelV = None

    def setup(self, search_space, device, logger=None, use_previous_rng=False):
        super().setup(search_space, device, logger, use_previous_rng)

        # Infer input and output dimensions for ML models.
        self.parameters["mutator_mlm_parameters"]["input_shape"] = self.search_space.input_dimension
        self.parameters["discriminator_mlm_parameters"]["input_shape"] = self.search_space.input_dimension
        self.parameters["validator_mlm_parameters"]["input_shape"] = self.search_space.input_dimension

        # Save current RNG state and use previous.
        if use_previous_rng:
            current_rng_state = torch.random.get_rng_state()
            torch.random.set_rng_state(self.previous_rng_state["torch"])
        else:
            self.previous_rng_state = {}
            self.previous_rng_state["torch"] = torch.random.get_rng_state()

        self._initialize()

        # Restore RNG state.
        if use_previous_rng:
            torch.random.set_rng_state(current_rng_state)        

    def _initialize(self, hard_reset=False):
        # The amount to scale, -1 for uninited, 0 for no scaling
        self.adaptive_scale = -1.0
        # Enabling or disabling adaptive scaling depending on how this is trained
        self.ascale_enabled = True
        # In case mutator is reset mid training (for example if scale is changedd), we want to train for a few extra epochs
        self.ascale_epoch_extra = 0
        
        # Load the specified mutator and discriminator machine learning
        # models unles they are already loaded.
        module = importlib.import_module("algorithms.examnet.mlm")
        mutator_class = getattr(module, self.mutator_mlm)
        discriminator_class = getattr(module, self.discriminator_mlm)
        validator_class = getattr(module, self.validator_mlm)

        if self.modelM is None or hard_reset:
            self.modelM = mutator_class(**self.mutator_mlm_parameters).to(self.device)
        else:
            self.modelM = self.modelM.to(self.device)
        if self.modelD is None or hard_reset:
            self.modelD = discriminator_class(**self.discriminator_mlm_parameters).to(self.device)
        else:
            self.modelD = self.modelD.to(self.device)
        if self.modelV is None or hard_reset:
            self.modelV = validator_class(**self.validator_mlm_parameters).to(self.device)
        else:
            self.modelV = self.modelV.to(self.device)
            
        self.modelM.train(False)
        self.modelD.train(False)
        self.modelV.train(False)

        # Load the specified optimizers.
        module = importlib.import_module("torch.optim")
        optimizer_class = getattr(module, self.optimizer)
        mutator_parameters = {k[10:]:v for k, v in self.parameters.items() if k.startswith("mutator")}
        self.optimizerM = optimizer_class(self.modelM.parameters(), **algorithm.filter_arguments(mutator_parameters, optimizer_class))
        discriminator_parameters = {k[14:]:v for k, v in self.parameters.items() if k.startswith("discriminator")}
        self.optimizerD = optimizer_class(self.modelD.parameters(), **algorithm.filter_arguments(discriminator_parameters, optimizer_class))
        validator_parameters = {k[14:]:v for k, v in self.parameters.items() if k.startswith("validator")}
        self.optimizerV = optimizer_class(self.modelV.parameters(), **algorithm.filter_arguments(validator_parameters, optimizer_class))

        # Loss functions.
        def get_loss(loss_s):
            loss_s = loss_s.lower()
            if loss_s == "mse":
                loss = torch.nn.MSELoss()
            elif loss_s == "l1":
                loss = torch.nn.L1Loss()
            elif loss_s == "bce":
                loss = torch.nn.BCELoss()
            elif loss_s == "mse,logit" or loss_s == "l1,logit":
                # When doing regression with values in [0, 1], we can use a
                # logit transformation to map the values from [0, 1] to \R
                # to make errors near 0 and 1 more drastic. Since logit is
                # undefined in 0 and 1, we actually first transform the values
                # to the interval [0.01, 0.99].
                L = 0.001
                g = torch.logit
                if loss_s == "mse,logit":
                    def f(X, Y):
                        return ((g(0.98*X+0.01) - g(0.98*Y+0.01))**2 + L*(g((1+X-Y)/2))**2).mean()
                else:
                    def f(X, Y):
                        return (torch.abs(g(0.98*X+0.01) - g(0.98*Y+0.01)) + L*torch.abs(g((1+X-Y)/2))).mean()
                loss = f
            else:
                raise Exception("Unknown loss function '{}'.".format(loss_s))

            return loss

        try:
            self.lossM = get_loss(self.mutator_loss)
            self.lossD = get_loss(self.discriminator_loss)
            self.lossV = get_loss(self.validator_loss)
        except:
            raise

    @classmethod
    def from_skeleton(C, skeleton):
        model = C(skeleton.parameters)
        model.modelM = copy.deepcopy(skeleton.modelM)
        model.modelD = copy.deepcopy(skeleton.modelD)
        model.modelV = copy.deepcopy(skeleton.modelV)

        return model

    def skeletonize(self):
        skeleton = Examnet_ModelSkeleton(self.parameters)
        skeleton.modelM = copy.deepcopy(self.modelM).to("cpu")
        skeleton.modelD = copy.deepcopy(self.modelD).to("cpu")
        skeleton.modelV = copy.deepcopy(self.modelV).to("cpu")

        return skeleton

    def reset(self):
        self._initialize(hard_reset=True)
    
    def reset_discriminator(self):
        module = importlib.import_module("algorithms.examnet.mlm")
        discriminator_class = getattr(module, self.discriminator_mlm)
        self.modelD = self.modelD.to(self.device)
        self.modelD.train(False)
        module = importlib.import_module("torch.optim")
        optimizer_class = getattr(module, self.optimizer)
        discriminator_parameters = {k[14:]:v for k, v in self.parameters.items() if k.startswith("discriminator")}
        self.optimizerD = optimizer_class(self.modelD.parameters(), **algorithm.filter_arguments(discriminator_parameters, optimizer_class))
        
    def reset_mutator(self):
        module = importlib.import_module("algorithms.examnet.mlm")
        mutator_class = getattr(module, self.mutator_mlm)
        self.modelM = mutator_class(**self.mutator_mlm_parameters).to(self.device)
        self.modelM.train(False)
        module = importlib.import_module("torch.optim")
        optimizer_class = getattr(module, self.optimizer)
        mutator_parameters = {k[10:]:v for k, v in self.parameters.items() if k.startswith("mutator")}
        self.optimizerM = optimizer_class(self.modelM.parameters(), **algorithm.filter_arguments(mutator_parameters, optimizer_class))
        
    def train_discriminator(self, dataX, dataY, train_settings=None):
        if self.modelD is None:
            raise Exception("No machine learning models available. Has the model been setup correctly?")

        if train_settings is None:
            train_settings = self.default_parameters["train_settings"]

        if len(dataY) < len(dataX):
            raise ValueError("There should be at least as many training outputs as there are inputs.")
            
        if train_settings is None:
            train_settings = self.default_parameters["train_settings"]
            
        dataX = torch.from_numpy(dataX).float().to(self.device)
        dataY = torch.from_numpy(dataY).float().to(self.device)

        # Unpack values from the train_settings dictionary.
        discriminator_epochs = train_settings["discriminator_epochs"] if "discriminator_epochs" in train_settings else 32
        return self._train_discriminator(dataX, dataY, discriminator_epochs)
        
    def train_mutator(self, dataX, train_settings=None):
        if train_settings is None:
            train_settings = self.default_parameters["train_settings"]
        
        modelM = self.modelM
        if modelM is None:
            raise Exception("No machine learning models available. Has the model been setup correctly?")
        
        #dataX = torch.from_numpy(dataX).float().to(self.device)
        
        # Unpack values from the train_settings dictionary.
        mutator_epochs = train_settings["mutator_epochs"] if "mutator_epochs" in train_settings else 16
        miter = train_settings["mutate_iterations"] if "mutate_iterations" in train_settings else 1
        venf = train_settings["validator_enforcement"] if "validator_enforcement" in train_settings else 1.0
        loss_alpha = train_settings["loss_alpha"] if "loss_alpha" in train_settings else 0.9
        return self._train_mutator(dataX, mutator_epochs, loss_alpha, venf, miter)
        
    def train_validator(self, dataX, validity, train_settings=None):
        if self.modelV is None:
            raise Exception("No machine learning models available. Has the model been setup correctly?")

        if train_settings is None:
            train_settings = self.default_parameters["train_settings"]
    
        dataX = torch.from_numpy(dataX).float().to(self.device)
        dataY = torch.from_numpy(validity).float().to(self.device)
        
        # Unpack values from the train_settings dictionary.
        validator_epochs = train_settings["validator_epochs"] if "validator_epochs" in train_settings else 8

        return self._train_validator(dataX, dataY, validator_epochs)
        
    def _train_discriminator(self, dataX, dataY, discriminator_epochs=1):        
        # Save the training modes for restoring later.
        training_D = self.modelD.training
        
        # Train the discriminator.
        # ---------------------------------------------------------------------
        # We want the discriminator to learn the mapping from tests to test
        # outputs.
        self.modelD.train(True)
        D_losses = []
        for _ in range(discriminator_epochs):
            D_loss = self.lossD(self.modelD(dataX), dataY)
            D_losses.append(D_loss.cpu().detach().numpy().item())
            c_loss = D_losses[-1]
            self.optimizerD.zero_grad()
            D_loss.backward()
            self.optimizerD.step()
        
        m = np.mean(D_losses)
        if discriminator_epochs > 0:
            self.log("Discriminator epochs {}, Loss: {} -> {} (mean {})".format(discriminator_epochs, D_losses[0], D_losses[-1], m))

        # Visualize the computational graph.
        # print(make_dot(D_loss, params=dict(self.modelD.named_parameters())))

        # Restore the training modes.
        self.modelD.train(training_D)
        
        return D_losses

    def _train_validator(self, dataX, validity, validator_epochs=1):        
        # Save the training modes for restoring later.
        training_V = self.modelV.training

        # Train the discriminator.
        # ---------------------------------------------------------------------
        # We want the discriminator to learn the mapping from tests to test
        # outputs.
        self.modelV.train(True)
        V_losses = []
        for _ in range(validator_epochs):
            V_loss = self.lossV(self.modelV(dataX), validity)
            V_losses.append(V_loss.cpu().detach().numpy().item())
            self.optimizerV.zero_grad()
            V_loss.backward()
            self.optimizerV.step()

        m = np.mean(V_losses)
        if validator_epochs > 0:
            self.log("Validator epochs {}, Loss: {} -> {} (mean {})".format(validator_epochs, V_losses[0], V_losses[-1], m))

        # Visualize the computational graph.
        # print(make_dot(D_loss, params=dict(self.modelD.named_parameters())))

        # Restore the training modes.
        self.modelV.train(training_V)

        return V_losses
    
    def _train_mutator(self, dataX, mutator_epochs, loss_alpha=0.9, validator_enforcement=1.0,mutate_iterations=1):
        loss_type = self.parameters["mutator_loss_type"] if "mutator_loss_type" in self.parameters else ""

        if(mutate_iterations > 0) and self.ascale_epoch_extra > 0:
            mutator_epochs += self.ascale_epoch_extra
            self.ascale_epoch_extra = 0

        # Train the mutator on the discriminator.
        # -----------------------------------------------------------------------
        # We want to train the mutator so that it learns to mutate the test so
        # that the mutated test is more likely to fail. 
        modelM = self.modelM
        optimizerM = self.optimizerM
        
        # Save the training modes for restoring later.
        training_M = modelM.training
        modelM.train(True)
        zeros_label = torch.zeros(size=(dataX.shape[0], 1)).to(self.device)
        ones_label = torch.ones(size=(dataX.shape[0], 1)).to(self.device)
        
        # Currently there are 3 different loss functions under test to figure out which one is the best
        # Weights could be used as hyperparameters to possibly increase performance.
        # To disable the validity checking from funnel network set val_w to 0.0, otherwise it is added
        # at the end of each algorithm to guide it away from failing cases with binary cross entropy.
        # Loss functions:
        # default - loss is equal to the score received from discriminator network. No additional calculations
        #    Note: It seems that this loss causes network to always converge on same output no matter the input.
        #          This loss type seems to be equivalent to what ogan would produce
        # rel - given original proposal and mutated test, loss equals alpha*dissimilarity(orig,mut)+(1-alpha)*(d(mut)-d(orig)+0,5)
        #	We measure dissimilarity between the 2 tests and add it to the estimated improvement achieved by mutation
        # msrel - same as rel, but the difference between the discriminator scores is scaled up to improve the training signal.
        #	  Scaling is as big as possible, but in case error gets smaller than 0 or higher than 1, the scaling is reduced.
        ori_w = loss_alpha
        mut_w = 1.0-loss_alpha
        val_w = validator_enforcement if loss_type != "gan" else 0.0
        self.log("Training mutator with alpha value: {}".format(loss_alpha))
        total_w = (ori_w + mut_w + val_w)*mutate_iterations
        M_losses = []        
        total_target_scale = 0.0
        
        for _ in range(mutator_epochs):
            din = dataX
            for i in range(mutate_iterations):
                outputs = modelM(din)
                v_score = self.modelV(outputs)
                d_score = self.modelD(outputs)
                
                if loss_type == "dist":
                    mut_loss = self.lossM(d_score, zeros_label)
                    dist_loss = self.lossM(dataX, outputs) 
                    agg_loss = mut_w * mut_loss
                    if ori_w > 0:
                        agg_loss += ori_w * dist_loss 
                    scores[2] = dist_loss
                elif loss_type == "rel":
                    #In case lossD is used for relative loss, the eps needs to be a bit bigger than zero to avoid infinite loss scores
                    eps = 0.0#1e-7
                    rel_loss = self.lossM((d_score-self.modelD(din)+0.5).clamp(min=eps,max=1.0-eps),zeros_label)
                    dist_loss = self.lossM(din, outputs)
                    agg_loss = mut_w * rel_loss
                    if ori_w > 0:
                        agg_loss += ori_w * dist_loss
                elif loss_type == "msrel":
                    MIN_SCALE = 2.0
                    dif = torch.abs(d_score-self.modelD(din))
                    tmv = torch.mean(dif)
                    if self.adaptive_scale < 0:
                        target_scale = max(int(abs(math.log2(tmv.item())))-MIN_SCALE, 0.0) if tmv > 0.0 else 0.0 
                    else:
                        target_scale = int(abs(math.log2(tmv.item()))) if tmv > 0.0 else 0.0
                    total_target_scale += target_scale                        
                    if self.adaptive_scale <= 0.0:
                        current_scale = 1.0
                    else:
                        current_scale = 2**self.adaptive_scale
                    
                    eps = 0.0#1e-7
                    ldif = (d_score*current_scale-self.modelD(din)*current_scale+0.5).clamp(min=eps,max=1.0-eps)
                    rel_loss = self.lossM(ldif,zeros_label)    
                    dist_loss = self.lossM(din, outputs)                   
                    agg_loss = mut_w * rel_loss
                    if ori_w > 0:
                        agg_loss += ori_w * dist_loss 
                else:
                    mut_loss = self.lossM(d_score, zeros_label)
                    agg_loss = mut_w * mut_loss
                
                M_loss = agg_loss if i == 0 else M_loss+agg_loss
                
                # Include the validity loss in calculations
                v_loss = self.lossV(v_score, ones_label)
                if val_w > 0.0:
                    M_loss += val_w * v_loss
                
                M_losses.append(M_loss.cpu().detach().numpy().item())
                if i < mutate_iterations - 1:
                    din = outputs.detach().clone()
            M_loss = M_loss/total_w
            optimizerM.zero_grad()
            M_loss.backward()
            optimizerM.step()                

        if self.ascale_enabled and loss_type == "larel":
            self.adaptive_scale = target_scale

        m = np.mean(M_losses)
        if mutator_epochs > 0:
            #TODO: Logging won't work as intedded in case mutating for more than one iteration.
            self.log("Mutator epochs {}, Loss: {} -> {}, mean {}".format(mutator_epochs, M_losses[0], M_losses[-1], m))
            # For msrel loss, when training mutator during the test generation step, make the scaling smaller if necessary
            if self.ascale_enabled and loss_type == "msrel":
                tts = int(total_target_scale/mutator_epochs + 0.5)
                new_scale = max(min(self.adaptive_scale, tts),0.0) if self.adaptive_scale >= 0.0 else tts
                print("----------------- Target scale for mutator: {}".format(self.adaptive_scale))
                if new_scale != self.adaptive_scale:
                    self.adaptive_scale = new_scale
                    

        #Restore the training modes.
        modelM.train(training_M)

        # Visualize the computational graph.
        # print(make_dot(G_loss, params=dict(self.modelM.named_parameters())))
        
        return M_losses

    def train_with_batch(self, dataX, dataY, train_settings=None,full_val_train=False):
        """Train the OGAN with a batch of training data.

        Args:
          dataX (np.ndarray): Array of tests of shape
                              (N, self.input_dimension).
          dataY (np.ndarray): Array of test outputs of shape (N, 1).
          train_settings (dict): A dictionary setting up the number of training
                                 epochs for various parts of the model. The
                                 keys are as follows:

                                   discriminator_epochs: How many times the
                                   discriminator is trained per call.


                                 The default for each missing key is 1. Keys
                                 not found above are ignored.

        Returns:
            D_losses (list): List of discriminator losses observed.
            M_losses (list): List of mutator losses observed."""
        
        modelM = self.modelM
        if modelM is None or self.modelD is None:
            raise Exception("No machine learning models available. Has the model been setup correctly?")

        if train_settings is None:
            train_settings = self.default_parameters["train_settings"]

        if len(dataY) < len(dataX):
            raise ValueError("There should be at least as many training outputs as there are inputs.")

        dataX = torch.from_numpy(dataX).float().to(self.device)
        dataY = torch.from_numpy(dataY).float().to(self.device)

        mutator_epochs = train_settings["mutator_epochs"] if "mutator_epochs" in train_settings else 16
        discriminator_epochs = train_settings["discriminator_epochs"] if "discriminator_epochs" in train_settings else 32

        D_losses = self._train_discriminator(dataX, dataY, discriminator_epochs)
        
        miter = train_settings["mutate_iterations"] if "mutate_iterations" in train_settings else 1
        venf = train_settings["validator_enforcement"] if "validator_enforcement" in train_settings else 1.0
        loss_alpha = train_settings["loss_alpha"] if "loss_alpha" in train_settings else 0.9
        self.ascale_enabled = False
        M_losses = self._train_mutator(dataX, mutator_epochs,loss_alpha,venf, miter)
        self.ascale_enabled = True
        return D_losses, M_losses
        
    def mutate_tests(self, tests=None,count=1):
        """Generate N random tests.

        Args:
          tests (np.ndarray):      Tests to mutate.
          count (int):                 Number of times to mutate each of the tests.

        Returns:
          output (np.ndarray): Array of shape (tests.shape).

        Raises:
        """

        try:
            return self._mutate_tests(tests, count, device=self.device)
        except:
            raise

    def predict_objective(self, test):
        """Predicts the objective function value of the given tests.

        Args:
          test (np.ndarray): Array of shape (N, self.input_ndimension).

        Returns:
          output (np.ndarray): Array of shape (N, 1).

        Raises:
        """

        try:
            return self._predict_objective(test, self.device)
        except:
            raise

