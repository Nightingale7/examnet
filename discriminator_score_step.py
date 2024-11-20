import numpy as np

from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS
from stgem.generator import Search
from stgem.sut import SearchSpace, SUT, SUTInput

class ScoreStep(Search):

    def __init__(self, algorithm, budget_threshold, mode="exhaust_budget", random_mode="uniform", results_include_models=False, results_checkpoint_period=1):
        super().__init__(algorithm, budget_threshold, mode=mode, results_include_models=results_include_models, results_checkpoint_period=results_checkpoint_period)
        self.random_mode = random_mode

        if not self.random_mode in ["uniform", "lhs"]:
            raise ValueError("Unknown random mode '{}'.".format(self.random_mode))

    def run(self):
        def all_zeros(l):
            for x in l:
                if x > 0:
                    return False
            return True

        self.budget.update_threshold(self.budget_threshold)

        # This stores the test indices that were executed during this step.
        test_idx = []
        # A list for saving model skeletons.
        model_skeletons = []

        # Allow the algorithm to initialize itself.
        self.algorithm.initialize()

        if self.random_mode == "lhs":
            # Here we continue the LHS of the initial step meaning that we need
            # to get a new RNG that has the same seed as the initial step and
            # use it to sample 75 tests to get it into the same state.
            rng = np.random.RandomState(seed=self.search_space.sut.seed)

            search_space = SearchSpace()
            search_space.setup(sut=self.search_space.sut, objectives=self.objective_funcs, rng=rng)

            executions = self.budget.budget_ranges["executions"][1]
            random = Random(model=LHS({"samples": executions}))
            random.setup(search_space, self.device, self.logger)

            for i in range(self.budget.budget_ranges["executions"][0]):
                random.generate_next_test(self.test_repository, self.budget.remaining())
        else:
            # We just want independent random samples, so using the same RNG as the search so far is OK.
            random = Random(model=Uniform())
            random.setup(self.search_space, self.device, self.logger)

        if (self.mode != "stop_all_objectives" and self.test_repository.minimum_objective <= 0.0) or \
           (self.mode == "stop_all_objectives" and all_zeros(self.test_repository.minimum_objectives)):
            self.success = True

        if self.mode != "stop_at_first_objective" or not self.success:
            # Below we omit including a test into the test repository if the
            # budget was exhausted during training, generation, or test
            # execution. We do not care for the special case where the budget
            # is exactly 0 as this is unlikely.

            i = 0
            minimum = self.test_repository.minimum_objective
            while self.budget.remaining() > 0:
                self.log("Budget remaining {}.".format(self.budget.remaining()))

                # Create a new test repository record to be filled.
                performance = self.test_repository.new_record()

                self.algorithm.train(self.test_repository, self.budget.remaining())
                self.budget.consume("training_time", performance.obtain("training_time"))
                if not self.budget.remaining() > 0:
                    self.log("Ran out of budget during training. Discarding the test.")
                    self.test_repository.discard_record()
                    break

                self.log("Starting to generate test {}.".format(self.test_repository.tests + 1))
                could_generate = True
                try:
                    next_test = self.algorithm.generate_next_test(self.test_repository, self.budget.remaining())
                except AlgorithmException:
                    # We encountered an algorithm error. There might be many
                    # reasons such as explosion of gradients. We take this as
                    # an indication that the algorithm is unable to keep going,
                    # so we exit.
                    break
                except GenerationException:
                    # We encountered a generation error. We take this as an
                    # indication that another training phase could correct the
                    # problem, so we do not exit completely.
                    could_generate = False

                self.budget.consume("generation_time", performance.obtain("generation_time"))
                if not self.budget.remaining() > 0:
                    self.log("Ran out of budget during test generation. Discarding the test.")
                    self.test_repository.discard_record()
                    break
                if could_generate:
                    self.log("Generated test {}.".format(next_test))
                    self.log("Executing the test...")

                    # OGAN test.
                    performance.timer_start("execution")
                    sut_input = SUTInput(next_test, None, None)
                    sut_output = self.sut.execute_test(sut_input)
                    performance.record("execution_time", performance.timer_reset("execution"))

                    # Random test.
                    random_test = random.generate_next_test(self.test_repository, self.budget.remaining())
                    self.log("Executing the random test {}.".format(random_test))
                    random_input = SUTInput(random_test, None, None)
                    random_output = self.sut.execute_test(random_input)

                    self.test_repository.record_input(random_input)
                    self.test_repository.record_output(random_output)

                    self.budget.consume("execution_time", performance.obtain("execution_time"))
                    self.budget.consume(sut_output)
                    if not self.budget.remaining() > 0:
                        self.log("Ran out of budget during test execution. Discarding the test.")
                        self.test_repository.discard_record()
                        break
                    self.budget.consume("executions")

                    self.log("Input to the SUT: {}".format(sut_input))

                    if sut_output.error is None:
                        self.log("Output from the SUT: {}".format(sut_output))

                        objectives = [objective(sut_input, sut_output) for objective in self.objective_funcs]
                        random_objectives = [objective(random_input, random_output) for objective in self.objective_funcs]
                        self.test_repository.record_objectives(random_objectives)

                        self.log("The actual objective: {}".format(objectives))
                        self.log("Random objective: {}".format(random_objectives))
                    else:
                        self.log("An error '{}' occurred during the test execution. No output available.".format(sut_output.error))
                        self.test_repository.record_objectives([])

                    idx = self.test_repository.finalize_record()
                    test_idx.append(idx)

                    if objectives[0] < minimum:
                        minimum = objectives[0]

                    if not self.success and (
                        (self.mode == "stop_at_first_objective" and minimum <= 0.0) or
                        (self.mode != "stop_at_first_objective" and all_zeros(self.test_repository.minimum_objectives))
                    ):
                        self.success = True
                        self.log("First success at test {}.".format(i + 1))
                else:
                    self.log("Encountered a problem with test generation. Skipping to next training phase.")

                # Save the model if requested.
                if self.results_include_models and self.results_checkpoint_period != 0 and i % self.results_checkpoint_period == 0:
                    model_skeletons.append(self.algorithm.model.skeletonize() if self.algorithm.model is not None else None)
                else:
                    model_skeletons.append(None)

                i += 1

                if self.success and self.mode == "stop_at_first_objective":
                    break

        # Allow the algorithm to store trained models or other generated data.
        self.algorithm.finalize()

        # Report results.
        self.log("Step minimum objective component: {}".format(self.test_repository.minimum_objective))

        result = self._generate_step_result(test_idx, model_skeletons)

        return result

