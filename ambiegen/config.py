from ambiegen.falsification_task import *
from ambiegen.falsification_task.ambiegen import *
from ambiegen.problem.ambiegen import *
from ambiegen.variant.ambiegen import *

benchmarks = {
    # benchmark               Problem           FalsificationTask            Mutant scoring                 Variant        
    "AMBIEGEN_WOGAN":         [AmbieGenProblem, WOGAN_FalsificationTask,     WOGAN_FalsificationTask,       AmbieGenVariant],
    "AMBIEGEN_RANDOM":        [AmbieGenProblem, Random_FalsificationTask,    WOGAN_FalsificationTask,       AmbieGenVariant],
    "AMBIEGEN_OGAN":          [AmbieGenProblem, OGAN_FalsificationTask,      OGAN_FalsificationTask, 	      AmbieGenVariant],
    "AMBIEGEN_EXAMNET":       [AmbieGenProblem, Examnet_FalsificationTask,   Examnet_FalsificationTask,     AmbieGenVariant],
}


config = {
    "INITIAL_MUTATION_SEED": 1024,
    "MAX_EXECUTIONS": 500,
    "MIN_EXECUTIONS": 5,
    "MUTATE_MAX_ATTEMPTS": -1,
    "MUTATE_ACCEPT_THR": 0.6,
    "RANDOM_SEARCH_BUDGET": 0.02,
    "SCORE_SEED": 25321,
    "SCORE_REPLICAS": 5
}


