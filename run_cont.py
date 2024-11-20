import importlib, os, sys
from time import sleep
import click
import csv

OUT_NAME = "examnet"

# Some imports need to be done inside functions for the environment variable
# setup to take effect.
from   stgem.objective import FalsifySTL

def get_generator_factory(description, sut_factory, objective_factory, step_factory, step_setup="default"):
    from stgem.generator import STGEM

    def generator_factory():
        objectives = objective_factory()
        return STGEM(description=description,
                     sut=sut_factory(),
                     objectives=objectives,
                     steps=step_factory(N_objectives=len(objectives), setup=step_setup))

    return generator_factory

def get_seed_factory(init_seed=0):
    def seed_generator(init_seed):
        c = init_seed
        while True:
            yield c
            c += 1

    g = seed_generator(init_seed)
    return lambda: next(g)

def get_sut_objective_factory(benchmark_module, selected_specification, mode):
    sut = benchmark_module.get_sut(mode)
    specifications, strict_horizon_check = benchmark_module.build_specification(selected_specification)

    ranges = {}
    for n in range(len(sut.input_range)):
        ranges[sut.inputs[n]] = sut.input_range[n]
    for n in range(len(sut.output_range)):
        ranges[sut.outputs[n]] = sut.output_range[n]

    def sut_factory():
        # Return the already instantiated SUT many times as Matlab uses a lot
        # of memory.
        return sut

    def objective_factory():
        nu = None
        #nu = 1
        return [FalsifySTL(specification=specification, ranges=ranges, scale=True, strict_horizon_check=strict_horizon_check, nu=nu) for specification in specifications]

    return sut_factory, objective_factory

def get_experiment_factory(N, benchmark_module, selected_specification, mode, init_seed, step_setup="default", callback=None):
    sut_factory, objective_factory = get_sut_objective_factory(benchmark_module, selected_specification, mode)

    from stgem.experiment import Experiment

    def experiment_factory():
        return Experiment(N=N,
                          stgem_factory=get_generator_factory("", sut_factory, objective_factory, benchmark_module.get_step_factory(), step_setup=step_setup),
                          seed_factory=get_seed_factory(init_seed),
                          result_callback=callback)

    return experiment_factory


benchmarks = ["AFC", "AT", "CC", "F16", "F16N", "NN", "PD", "PM", "SC", "AG"]
descriptions = {
        "AFC":  "Fuel Control of an Automotive Powertrain",
        "AT":   "Automatic Transmission",
        "CC":   "Chasing Cars",
        "F16":  "Aircraft Ground Collision Avoidance System",
        "F16N": "Aircraft Ground Collision Avoidance System (newer version)",
        "NN":   "Neural-Network Controller",
        "PD":  "Pendulum Model",
        "PM":   "Pacemaker",
        "SC":   "Steam Condenser with Recurrent Neural Network Controller",
        "AG":   "AMBIEGEN"
}
specifications = {
        "AFC":  ["AFC27", "AFC29"],
        "AT":   ["AT1", "AT2", "AT51", "AT52", "AT53", "AT54", "AT6A", "AT6B", "AT6C", "AT6ABC", "ATX13", "ATX14", "ATX1", "ATX2", "ATX61", "ATX62"],
        "CC":   ["CC1", "CC2", "CC3", "CC4", "CC5", "CCX"],
        "F16":  ["F16"],
        "F16N": ["F16"],
        "NN":   ["NN", "NNB", "NNX"],
        "PD":   ["PD"],
        "PM":   ["PM"],
        "SC":   ["SC"],
        "AG":   ["AG"]
}
N_workers = {
        "AFC": 3,
        "AT": 3,
        "CC": 4,
        "F16": 2,
        "F16N": 2,
        "NN": 3,
        "PD": 1,
        "PM": 2,
        "SC": 2,
        "AG": 2
}
def get_finished_count(selected_benchmark, selected_specification):
    filename = "{}_{}_{}.csv".format(selected_benchmark, selected_specification, OUT_NAME)
    fileObject = csv.reader(filename)
    with open(filename) as f:
        return sum(1 for line in f)


@click.command()
@click.argument("selected_benchmark", type=click.Choice(benchmarks, case_sensitive=False))
@click.argument("selected_specification", type=str)
@click.argument("mode", type=str, default="")
@click.argument("step_setup", type=str, default="default")
@click.argument("n", type=int)
@click.argument("init_seed", type=int)
@click.argument("identifier", type=str, default="")
@click.argument("replicas", type=int, default=1)
def main(selected_benchmark, selected_specification, mode, step_setup, n, init_seed, identifier, replicas=1):
    try:
        c = get_finished_count(selected_benchmark, selected_specification)
        if selected_specification == "AFC27" and mode == "power":
            c = get_finished_count(selected_benchmark, "AFC33")
    except:
        c = 0
    replicas = replicas - c
    init_seed += c
    if replicas <= 0:
        return
    N = n

    # Disable CUDA if multiprocessing is used.
    if N > 1 and N_workers[selected_benchmark] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not selected_specification in specifications[selected_benchmark]:
        raise Exception("No specification '{}' for benchmark {}.".format(selected_specification, selected_benchmark))

    def callback(idx, result, N, done):
        path = os.path.join("stgem_outputs", selected_specification)
        time = str(result.timestamp).replace(" ", "_").replace(":", "")
        file_name = "{}{}_{}.pickle.gz".format(selected_specification, "_" + identifier if identifier is not None else "", time)
        os.makedirs(path, exist_ok=True)
        result.dump_to_file(os.path.join(path, file_name))

        filename = "./output/{}_{}_{}.csv".format(selected_benchmark, selected_specification, OUT_NAME)
        if selected_specification == "AFC27" and mode == "power":
            filename =  "./output/AFC_AFC33_{}.csv".format(OUT_NAME)

        score = result.test_repository.tests
        text = "{},{}".format(init_seed, score)
        f = open(filename, "a+")
        f.write(text+'\n')
        f.close()

    for i in range(replicas):
        benchmark_module = importlib.import_module("{}.benchmark".format(selected_benchmark.lower()))
        
        experiment = get_experiment_factory(N, benchmark_module, selected_specification, mode, init_seed, step_setup, callback=callback)()

        use_gpu = N == 1 or N_workers[selected_benchmark] == 1
        experiment.run(N_workers=min(N, N_workers[selected_benchmark]), silent=False, use_gpu=use_gpu)
        init_seed += 1
            

if __name__ == "__main__":
    main()

