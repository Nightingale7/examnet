from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS
from stgem.algorithm import Algorithm
from stgem.exceptions import GenerationException
from collections import deque 
import heapq
import random
import torch
import os
import random
import numpy as np
import itertools
import math

class Examnet(Algorithm):
    """Implements the online mutator-type algorithm for test generation. """
#TODO: Implement approach where instead of discriminator we learn comparator  that tells us whether new test is better
    # Do not change the defaults
    default_parameters = {
        "fitness_coef": 0.99,
        "train_delay": 1,
        "N_candidate_tests": 50,
        "invalid_threshold": 10000,
        "mutation_count": 1,
        "enable_mutation": 40,
        "exploration_ratio": 0.9,
        "train_mode": "Default", #"Reset", "Default", "Balanced"
        "insert_random_at": 10,
        "train_memory": 0,
        "test_source": "Generator",
        "init_alg": "pex",
        "insert_alg": "uni",
        "batch_alg": "rnd",
        "std_memory": 5,
        "std_range": [0.0,1.0],
        "mutator_alpha_range": [0.01, 0.99],
        "max_precision": 5,
        "generation_iterations": 10,
        "scale_output": False,
        "range_expand": 1.0,
        "improvement_req": 0.05
    }

    def setup(self, search_space, device=None, logger=None):
        super().setup(search_space, device, logger)
        init_exp_size = self.enable_mutation
        self.test_gen = TestGenerator(alg=self.init_alg, N=init_exp_size, batch_count=self.N_candidate_tests)
        self.test_gen.setup(search_space, device, logger)
        #self.enable_mutation_on = max(len(self.test_gen.test_queue), self.enable_mutation_on)
        self.model_trained = 0 # keeps track how many tests were generated when a model was previously trained
        if len(self.minimized_objectives) > 1:
            raise ValueError("Examnet only supports one objective. Check the minimized_objectives parameter.")
        self.objective_idx = self.minimized_objectives[0]
        self.queued_tests = []
        self.mut_enabled = False
        self.batch_gen = True
        self.log_scale = 0.0
        self.trail_scale = 0.0
        self.test_std = 0.0
        self.adaptive_alpha = 0.5
        self.std_avg = 1.0
        self.SCL_CONST = 0.1
        self.investigated = []
        self.test_precision = self.max_precision
        self.min_fitness = 1.0
        self.std_ratio_penalty = 0.0
        
    def init_train(self, model, dataX, dataY):
        epochs = model.train_settings_init["epochs"]
        train_settings = model.train_settings_init
        train_settings["loss_alpha"] = self.get_mutator_alpha(self.mutator_alpha_range[-1]/2.0+self.mutator_alpha_range[0]/2.0)
        train_settings["validator_enforcement"] = 0.0
        train_settings["mutate_iterations"] = max(1, self.mutation_count)
        for epoch in range(epochs):
            D_losses, M_losses = model.train_with_batch(dataX,
                                                        dataY,
                                                        train_settings=train_settings,
                                                        full_val_train=True
                                                        )
    # Calculate std from latest [std_memory] tests that weren't raddomly selected
    def get_nonrand_std(self, data):
        total_std = self.std_avg
        if total_std == 0.0:
            return 1.0
        rv = [i for i in range(len(data)) if (i+1)%self.insert_random_at == 0]
        data = np.delete(data, rv)
        #Divide by initial random tests in attempt to normalize the distribution 
        test_std = np.std(data[-min(len(data),self.std_memory):])/total_std
        target_std = round(1.0 - min(1.0,max(0.0,(test_std - self.std_range[0])/(self.std_range[1]-self.std_range[0]))),3)
        #print("Scaled STD: current {} - total {}, target {}".format(test_std, total_std, target_std))
        return target_std
    
    def update_std_avg(self, history):
        rv1 = [d for i, d in enumerate(history[self.enable_mutation:]) if (i+1)%self.insert_random_at == 0]
        rv2 = [d for d in history[self.test_gen.pex_budget:self.enable_mutation]]
        rv = rv1+rv2
        self.std_avg = np.std(rv)
        if self.std_ratio_penalty > 0.0:
            ratio = self.std_ratio_penalty/100.0
            fstd = np.std(history)
            self.std_avg = self.std_avg*(1.0-ratio) + fstd * ratio        

    # Check if the proposed test has been explored. The condition is that it closer than provided distance
    def is_unexplored(self, test, thr_dist):
        if len(self.investigated) == 0:
            return True
        for exp_test in self.investigated:
             if np.linalg.norm(test-exp_test) <= thr_dist:
                 return False
        return True
        
    # Returns True if test is valid and unexplored
    def check_test_validity(self, test, mdist):
        if not self.search_space.is_valid(test):
            return False
        if self.investigated is None or mdist == 0.0:
            return True
        return self.is_unexplored(test, mdist)

    def do_train(self, test_repository, budget_remaining):
        # PerformanceRecordHandler for the current test.
        performance = test_repository.performance(test_repository.current_test)
        discriminator_losses = []
        mutator_losses = []
        performance.record("discriminator_loss", discriminator_losses)
        performance.record("mutator_loss", mutator_losses)

        # Currently we support only a single Mutator model.
        model = self.get_models(0)[0]

        # Take into account how many tests a previous step (usually random
        # search) has generated.
        self.tests_generated = test_repository.tests
        
        X, _, Y = test_repository.get()
        if len(X) > 0:
            dataX = np.asarray([sut_input.inputs for sut_input in X])/self.range_expand
            dataY = np.array(Y)[:, self.objective_idx].reshape(-1, 1)
            if self.init_alg == "pex":
                self.test_gen.update_hist_data(dataX,dataY)
            self.min_fitness = min(dataY)

            # If output scaling is performed, scale the output and make sure it is in the allowed range
            # Currently disabled
            if self.log_scale > 1:
                scaled_data = (dataY-self.trail_scale)*10**(self.log_scale-1)+self.SCL_CONST
                while np.min(scaled_data) < 0.0:
                    self.log_scale = max(self.log_scale-1, 0)
                    self.trail_scale = int(self.trail_scale*10**self.log_scale)/10**self.log_scale
                    scaled_data = (dataY-self.trail_scale)*10**(self.log_scale-1)+self.SCL_CONST
                    model.reset_discriminator()
                    model.reset_mutator()
                    self.std_avg = np.std(scaled_data[:self.enable_mutation+1])

                dataY = np.clip(scaled_data, 0.0,1.0)
            self.investigated = dataY
            
            # If batch gen is false we will only select one random unmutated test for the next test proposal
            # Essentially we insert random tests at fixed interval                        
            self.batch_gen = True
            if self.insert_random_at >= 0 and budget_remaining > 0:
                self.batch_gen = (len(dataY)+1)%self.insert_random_at != 0    
           
        # Perform initial train after set number of random attempts and when done, enable mutation
        if not self.mut_enabled and len(test_repository.get()[0]) >= self.enable_mutation:
            self.update_std_avg(dataY)
            avgy = np.average(dataY)
            dify = np.max(np.abs(dataY - avgy))
            self.log_scale = max(0.0,int(abs(math.log10(dify)))) if self.scale_output else 0.0
            if self.log_scale > 0:
                self.trail_scale = int(avgy*10**self.log_scale)/10**self.log_scale
                scaled_data = (dataY-self.trail_scale)*10**(self.log_scale-1)
                self.SCL_CONST = max(self.SCL_CONST, round(self.trail_scale,1))
                scaled_data += self.SCL_CONST
                dataY = np.clip(scaled_data, 0.0,1.0)
            self.std_avg = np.std(dataY)
            self.init_train(model, dataX, dataY)
            self.model_trained = self.tests_generated
            self.mut_enabled = True
            validity = np.array([[1] for i in range(len(dataX))])
            model.train_validator(dataX, validity)                        
        
        # Train the model with the initial tests.
        if self.mut_enabled:
            # Calculate the desired alpha based on std of recent tests and adjust alpha to be closer to it
            self.update_std_avg(dataY)
            target_std = self.get_nonrand_std(dataY)
            if self.adaptive_alpha <= self.std_range[0]:
            	self.std_ratio_penalty = max(0.0, self.std_ratio_penalty-5)
            if self.adaptive_alpha >= 0.9:
            	self.std_ratio_penalty = min(100.0, self.std_ratio_penalty+5)

            # variance is low for too long, even when producing semi-random tests, then it means either tests are not random enough or that threshold for making alpha lower is too big -> decrease the maximum by 0.001
            # variance is too high for too long then it means focusing is not enough, ie. cannot train
            self.log("Last Test: {}, ASTD:{}, Target: {}".format(dataY[-1],self.std_avg, target_std))
            ALPHA_STEP = 0.1
            if abs(target_std - self.adaptive_alpha) <= ALPHA_STEP:
                self.adaptive_alpha = target_std
            else:
                self.adaptive_alpha += ALPHA_STEP if (target_std-self.adaptive_alpha) > 0.0 else -ALPHA_STEP
            self.adaptive_alpha = round(self.adaptive_alpha,3)
            
            self.log("Training the Mutator model...")
            
            # If enabled, we reset discriminator each training and train on all the tests, but do not reset validator and mutator models
            if self.train_mode == "Reset":
                model.reset_discriminator()
                        
            epochs = model.train_settings["epochs"] 
            train_settings = model.train_settings 
            
            if self.train_memory > 0 and len(dataX) > self.train_memory:
                dataX = dataX[-self.train_memory:]
                dataY = dataY[-self.train_memory:]
                            
            train_settings["loss_alpha"] = self.get_mutator_alpha(self.adaptive_alpha)
            train_settings["validator_enforcement"] = 1.0
            train_settings["mutate_iterations"] = max(1, self.mutation_count)
            for epoch in range(epochs):
                D_losses = model.train_discriminator(dataX, dataY,train_settings=train_settings)
                discriminator_losses.append(D_losses)
                
                self.model_trained = self.tests_generated
    
    def get_mutator_alpha(self, raw_scale):
        if len(self.mutator_alpha_range) == 1:
            return self.mutator_alpha_range[0]        
        mar_min = self.mutator_alpha_range[0]
        mar_max = self.mutator_alpha_range[1]
        return round(mar_min+raw_scale*(mar_max-mar_min),3)
        
    def do_generate_next_test(self, test_repository, budget_remaining):
        heap = []
        target_fitness = 0
        entry_count = 0  # this is to avoid comparing tests when two tests added to the heap have the same predicted objective
        N_generated = 0
        N_invalid = 0

        # Currently we support only a single Mutator model.
        model = self.get_models(0)[0]
        train_settings = model.train_settings
        
        mdist = self.search_space.input_dimension / 10**self.max_precision
        
        self.log("Generating a test with Examnet model. Batch Gen: {}, Budget: {}, AdaptiveAlpha: {}".format(self.batch_gen,budget_remaining, self.adaptive_alpha))
     
        # PerformanceRecordHandler for the current test.
        performance = test_repository.performance(test_repository.current_test)
        generate_attempts = self.generation_iterations
        
        while generate_attempts > 0:
            generate_attempts -= 1
            found_valid = False
            while not found_valid:
                # If we have already generated many tests and all have been
                # invalid, we give up and hope that the next training phase
                # will fix things.
                if N_invalid >= self.invalid_threshold:
                    raise GenerationException("Could not generate a valid test within {} tests.".format(N_invalid))
                    
                test_source = self.test_source #"Generator"
                data = None
                mcount = 0 if not self.mut_enabled else self.mutation_count
                 
                # Use recent tests as inputs, if not enough tests, give as many as possible
                if test_source == "Generator":
                    attempts = 1000
                    min_valid = 0.5 
                    while(attempts > 0):
                        if not self.mut_enabled:
                            input_tests = self.test_gen.get_next_test(alg="", batch_gen=False)
                        elif self.batch_gen:
                            input_tests = self.test_gen.get_next_test(alg=self.batch_alg, batch_gen=True)
                        else:
                            input_tests = self.test_gen.get_next_test(alg=self.insert_alg, batch_gen=False)
                        
                        val_to_check = input_tests.cpu().detach().numpy()
                        v = [1 for i in range(len(input_tests)) if self.check_test_validity(val_to_check[i], mdist=0.0) == True]
                        if sum(v)/len(val_to_check) > min_valid:
                            attempts = 0
                        attempts -= 1
                    if input_tests.shape[0] == 1:
                        mcount = 0
                else:
                    # As data is None, generate_input_test will simply generate requested number of random tests
                    input_tests = model.generate_input_test(data, N=self.N_candidate_tests)
                # Continue training mutator with the new information
                train_settings["loss_alpha"] = self.get_mutator_alpha(self.adaptive_alpha)
                train_settings["validator_enforcement"] = min(10.0, max(0.0, N_invalid/self.N_candidate_tests))/10.0
                train_settings["mutate_iterations"] = max(1, mcount)
                if mcount > 0:
                    model.train_mutator(input_tests,train_settings)
                    
                self.log("Generating the test with input size - {} and mcount - {}".format(input_tests.shape[0], mcount))
                # Generate several tests and pick the one with best predicted
                # objective function component. We do this as long as we find
                # at least one valid test.
                try:
                    candidate_tests = model.mutate_tests(input_tests, count=mcount)
                except:
                    raise GenerationException("Error while attempting to mutate!")

                
                candidate_tests = np.around(candidate_tests,self.test_precision)    
                # Pick only the valid tests.
                cexp = np.clip(candidate_tests*self.range_expand, -1.0, 1.0)
                valid_idx = [i for i in range(len(candidate_tests)) if self.check_test_validity(cexp[i],mdist) == True]
                invalid_idx = [i for i in range(len(candidate_tests)) if i not in valid_idx]
                invalid_tests = candidate_tests[invalid_idx]
                orig_noise = input_tests[valid_idx]
                valid_tests = candidate_tests[valid_idx]
                N_generated += len(candidate_tests)
                N_invalid += len(candidate_tests) - len(valid_idx)
                train_settings["validator_enforcement"] = min(10.0, max(0.0, N_invalid/self.N_candidate_tests))/10.0
                
                # If any of the generated tests were invalid, finetune the validator
                if len(invalid_tests) > 0:                    
                    validity = np.array([[1] if i in valid_idx else [0] for i in range(len(candidate_tests))])
                    # Train the validation network on the new tests that we got
                    model.train_validator(candidate_tests, validity,train_settings)
                                
                if valid_tests.shape[0] > 0:
                    # Estimate objective function values and add the tests to heap.
                    tests_predicted_objective = model.predict_objective(valid_tests)
                    for i in range(len(tests_predicted_objective)):
                        heapq.heappush(heap, (tests_predicted_objective[i], entry_count, valid_tests[i], orig_noise[i]))
                        entry_count += 1
                    found_valid = True
                else:
                    generate_attempts += 1
                    print("All mutants are invalid")

            # To accept the test we either need to iterate maximum number of times or the fitness of the best test needs to improve on previous fitness enough
            target_fitness = self.min_fitness/(1.0+generate_attempts*self.improvement_req)

            # Check if the best predicted test is good enough.
            eps = 1e-4
            if heap[0][0] - eps <= target_fitness or mcount == 0: break
            #if heap[0][0] <= target_fitness or mcount == 0: break
            #if mcount == 0: break

        # Save information on how many tests needed to be generated etc.
        # -----------------------------------------------------------------
        performance.record("N_tests_generated", N_generated)
        performance.record("N_invalid_tests_generated", N_invalid)

        htest = heap[0]        
        best_test = htest[2]
        best_estimated_objective = htest[0]
        best_test = np.clip(best_test*self.range_expand, -1.0, 1.0)

        self.log("Chose test {} with predicted minimum objective {}. Generated total {} tests of which {} were invalid.".format(best_test, best_estimated_objective, N_generated, N_invalid))
        return best_test


# Test generator to feed random tests into the algorithm.
# Implementation needs to ensure 2 modes: tests during the initial step and separately for tests inserted mid execution
# One of the features achieved this way is that algorithm can test some of the obvious extreme values before starting the learning algorithm.
class TestGenerator():
    def __init__(self, alg="rnd", N=0, batch_count=1):
        self.batch_count = batch_count
        self.test_queue = deque([])
        self.N = N
        self.gen_alg = alg
        self.uniform = Uniform()
        self.iterated = 0                   
        self.current_step = 0
        # How many coverage array samples to generate
        self.pex_budget = 20
        # How many tests each step generates
        self.pex_step = 5
        # How many times are we allowed to increase the power of the covering array that we use (3 means we use covering arrays of powers 2,3,4,5)
        self.pex_imax = 3
        # When calculating the score as to which tests from new covering array candidates should be executed, this is how many closest candidates we use for that score        
        self.pex_clen = 3
        self.history = []
    
    #  Set up different modes of test generation    
    def setup(self, search_space, device, logger=None, use_previous_rng=False):
        self.input_shape = search_space.input_dimension
        self.device = device
        
        if self.gen_alg == "lhs":
            self.lhs_crit = None# "center", "maximin", "centermaximin", "correlation"
            self.test_gen = LHS({"samples": self.N}) 
        else:
            self.test_gen = Uniform()     
        self.uniform.setup(search_space, self.device)   
        self.test_gen.setup(search_space, self.device)
        self.generate_queue(self.N)
        if self.gen_alg == "ext":
            self.gen_alg = "uni"

    def init_queue_finished(self):
        return self.iterated
    
    # 
    def generate_queue(self, N=1):
        tests = []
        if N > 0:
            tests = [torch.from_numpy(np.asarray(t)).to(self.device).type(torch.float32).unsqueeze(0) for t in self._generate(N)]
        self.test_queue = deque(tests)

    def get_next_test(self, alg="", batch_gen=True):
        if self.gen_alg == "pex" and self.iterated >= 1 and self.iterated <= self.pex_imax:
            self.current_step += 1
            if self.current_step >= self.pex_step:
                self.current_step = 0
                self.test_queue = deque([])
        while len(self.test_queue) == 0:
            self.iterated += 1
            self.generate_queue(N=self.N)
        if self.iterated <= self.pex_imax:
            valid_tests = sum([1 for v, s in self.history if s <= 1.0 ])
            if valid_tests >= self.pex_budget:
                self.iterated = self.pex_imax + 1
                self.test_queue = deque([])
        valid_tests = sum([1 for v, s in self.history if s <= 1.0 ])
        if ((self.gen_alg != "pex" and self.iterated > 0) or self.iterated > self.pex_imax) and batch_gen:
            return self.generate_tests(alg=alg, N=self.batch_count)
        if len(self.test_queue) == 0:
            self.generate_queue(N=1)        
        return self.test_queue.popleft().to(self.device).type(torch.float32)
    
    # Covering arrays stored in csv files according to the naming convention. This function loads them and scales to the range of [-1.0, 1.0]            
    def load_ca(self, length, strength=4,add_full_ext=False):
        if length > 64 or length < 1:
            raise GenerationException("No precalculated coverage array given for length {}".format(length))
        if length <= strength:
            rv = np.array([v for v in itertools.product([-1.0, 1.0], repeat=length)])
        else:
            ca_str = strength
            fn = "{}/covering_array/ca.{}.2^{}.txt".format(os.path.dirname(__file__),ca_str,length)
            with open(fn) as f:
                lines = f.readlines()[1:]
                tests = len(lines)
                add_ones = True
                add_minus = True
                
                rv = -1.0*np.ones((tests,length))
                for idx, line in enumerate(lines):
                    vals = line.strip().split(" ")
                    for idx2, v in enumerate(vals):  
                        if v == "1":
                            rv[idx,idx2] = 1.0
                            
                    if add_full_ext:
                        s = sum(rv[idx])
                        if s == length:
                            add_ones = False
                        if s == -length:
                            add_minus = False

                if add_full_ext:
                    if add_ones:
                        rv = np.concatenate((rv,np.ones((1,length))),axis=0) 
                    if add_minus:
                        rv = np.concatenate((rv,-1*np.ones((1,length))),axis=0) 
        return rv

    # Extreme tests 
    def generate_extremes(self, length, strength, add_full_ext):
        return self.load_ca(self.input_shape, strength, add_full_ext=add_full_ext)
    
    # Function to update the tests that were executed so that new tests would be generated
    # Currently assumes that every update with dataY=None is a new set of extremes that will be executed
    # Assumes that update history is called each time new test is executed
    #TODO: It would make ode more readable if this was separated into 2 functions
    def update_history(self,dataX, dataY=None):
        if self.iterated > self.pex_imax or dataX.shape[0] == 0:
            return
        eps = 1e-9
        if dataY is None:
            for x in dataX:
                contains = False
                for h in self.history:
                    d = np.linalg.norm(h[0]-x)
                    if d < eps:
                        contains = True
                        break
                if not contains:
                    self.history.append((x,1.0))
        else:
            x = dataX[-1]
            for i in range(len(self.history),0,-1):
                h = self.history[i-1]
                d = np.linalg.norm(h[0]-x)
                if d < eps:
                    self.history[i-1] = (h[0], dataY[-1])
                    break
                    
    def add_next_tests(self, data):
        INIT_VALUE = 10.0
        if self.iterated > self.pex_imax or data.shape[0] == 0:
            return
        eps = 1e-9
        for x in data:
            contains = False
            for h in self.history:
                d = np.linalg.norm(h[0]-x)
                if d < eps:
                    contains = True
                    break
            if not contains:
                self.history.append((x, 10.0))
    def update_hist_data(self, dataX, dataY):
        if self.iterated > self.pex_imax or dataX.shape[0] == 0:
            return
        eps = 1e-9
        x = dataX[-1]
        for i in range(len(self.history),0,-1):
            h = self.history[i-1]
            d = np.linalg.norm(h[0]-x)
            if d < eps:
                if dataY is None:
                    self.history[i-1] = (h[0], 1.0)
                else:
                    self.history[i-1] = (h[0], dataY[-1][0])
                break
    
    # Function to sort the extreme candidates according to how close they are to low fitness tests that were executed
    # only top [self.cdist] candidates are used for calculating the score
    def sort_excand(self, candidates):
        if self.iterated == 0 or sum(v for a,v in self.history) == 10.0*len(self.history):
            if self.iterated > self.pex_imax:
                return np.array([])
            return candidates
        eps = 1e-9
        oa = []
        for c in candidates:
            cdist = []
            same = False
            for h in self.history:
                nd = np.linalg.norm(h[0]-c)/2
                if nd < eps and h[1] <= 1.0:
                    same = True
                    break
                cdist += [ nd * h[1] ]
            if not same:
                cdist.sort()
                cscore = sum(cdist[0:min(len(cdist), self.pex_clen)])
                oa += [(c, cscore)]
        oa.sort(key=lambda x: x[1], reverse=False)
        oa = np.array([x[0] for x in oa])
        return oa
    
    # Generate tests according the algorithm selected
    # "pex" is hybrid approach which first generates tests from extremes and then uses uniformly random tests                
    def _generate(self, N=1):
        if self.gen_alg == "lhs":
            return 2*(self.test_gen.lhs(self.input_shape, samples=N, criterion=self.lhs_crit) - 0.5)
        elif self.gen_alg == "ext":
            et = self.generate_extremes(length = self.input_shape, strength = 4)
            if et.shape[0] < N:
                et = np.concatenate((et,self.uniform.generate_test(N-et.shape[0])),axis=0) 
            return et        
        elif self.gen_alg == "pex":
            if self.iterated > self.pex_imax:
                et = self.uniform.generate_test(N)
            else:
                fe = True if self.iterated == 0 else False                                                        
                et = self.generate_extremes(length = self.input_shape, strength = self.iterated+2, add_full_ext=fe)
                et = self.sort_excand(et)
                self.add_next_tests(et)
                if et.shape[0] == 0:
                    et = self.uniform.generate_test(N)
                elif et.shape[0] < N and self.iterated == self.pex_imax or et.shape[0] == 0:
                    et = np.concatenate((et,self.uniform.generate_test(N-et.shape[0])),axis=0)
            return et        
        return self.test_gen.generate_test(N)
    
    def generate_tests(self, alg="", N=1):
        if alg=="":
            tests = torch.from_numpy(np.asarray(self._generate(N))).to(self.device).type(torch.float32)
        elif alg == "uni":
            tests = torch.from_numpy(self.uniform.generate_test(N)).to(self.device).type(torch.float32)
        else:
            tests = (torch.rand(size=(N, self.input_shape))*2 - 1).to(self.device).type(torch.float32)     
        return tests
        
    def generate_gaussian_scatter(self, orig_test, N, variance):
        test = torch.from_numpy(np.asarray(orig_test)).to(self.device).type(torch.float32).repeat(N,1)
        if variance > 0.0:
            noise = ((variance**0.5)*torch.randn(N, self.input_shape)).to(self.device).type(torch.float32)
            test += noise  
        return test
   
