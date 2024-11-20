import sys
import click
import os.path
from ambiegen.config import benchmarks, config

def evaluate_variant(benchmark, variant_identifier, seed=0):
    problem_class = benchmarks[benchmark][0]
    falsification_task_class = benchmarks[benchmark][1]
    score_falsification_task_class = benchmarks[benchmark][2]
    variant_class = benchmarks[benchmark][3]
    variant = variant_class.load(variant_identifier)

    score = 500
    
    revision = variant.get_revision(0)
    problem = problem_class.from_revision(revision)
    falsification_task = falsification_task_class(problem)
    result = falsification_task.run_replica(seed=seed, models=None, silent=False, prev_tests=None)
    if result is None:
        score = 500
    else:
        # Check if the final execution had an error (the run is stopped
        # in this case).
        if result.test_repository.get(-1, include_all=True)[1].error is not None:
            score = 500
        else:
            score = result.test_repository.tests

    print("Score: {}".format(score))
    return score

def get_finished_count(filename):
    if not os.path.isfile(filename):
        return 0
    with open(filename) as f:
        return sum(1 for line in f)
        
@click.command()
@click.argument("bench", type=str, default="AMBIEGEN")
@click.argument("variant", type=str, default="WSV")
@click.argument("seed", type=int, default=1)
@click.argument("repeat", type=int, default=1)
@click.argument("name", type=str, default="")

def main(bench, variant, seed, repeat, name):    
    filename = "./output/{}_{}.csv".format(bench, name)    
    fc = get_finished_count(filename)

    desc = bench
    for i in range(repeat):
        seed += 1
        if i < fc:
            continue
            
        score = evaluate_variant(bench, variant, seed=seed)
            
        text = "{}".format(score)
        f = open(filename, "a+")
        f.write(text+'\n')
        f.close()
    

if __name__ == "__main__":
      main()
