import numpy as np

from stgem.objective import Objective

from ambiegen.problem import Problem

class AmbieGenObjective(Objective):

    def __call__(self, t, r):
        # Distance 4.55 allows uniform random search to falsify 10% of the time
        # with 500 executions (measured over 50 replicas).
        distance = 1 - np.clip(max(r.outputs[0]), 0, 4.55)/4.55
        return distance

class AmbieGenProblem(Problem):

    def get_objectives(self):
        return [AmbieGenObjective()]

