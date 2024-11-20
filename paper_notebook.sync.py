# %% [markdown]
# # Finding faults in cyber-physical systems using generative models

Jarkko Peltomäki (jarkko.peltomaki@abo.fi)<br> 
Ivan Porres (ivan.porres@abo.fi)<br>
Information Technology, Faculty of Science and Technology<br>
Åbo Akademi University<br>
Turku, Finland

# %% [markdown]
In order to replicate the experiments, please run the below code cells in
order.

# %% [markdown]
# ## Setup

# %%
%matplotlib widget
import copy, importlib, os, sys
import numpy as np
import pandas
from scipy import stats
import matplotlib.pyplot as plt
import lifelines

from stgem.generator import STGEM, STGEMResult

from run import get_generator_factory, get_sut_objective_factory

# %%
def collect_replica_files(path, prefix):
    if not os.path.exists(path):
        raise Exception("No path '{}'.".format(path))

    results = []
    for dir, subdirs, files in os.walk(path):
        for file in files:
            if file.startswith(prefix):
                results.append(os.path.join(dir, file))

    return results

def load_results(files, load_sut_output=True):
    results = []
    for file in files:
        results.append(STGEMResult.restore_from_file(file))

    # This reduces memory usage if these values are not needed.
    if not load_sut_output:
        for result in results:
            result.test_repository._outputs = None

    return results

def loadExperiments(path, benchmarks, experiments_description):
    experiments = {}
    for benchmark in benchmarks:
        experiments[benchmark] = {}
        for experiment in experiments_description[benchmark]:
            prefix = experiments_description[benchmark][experiment]
            files = collect_replica_files(os.path.join(path, benchmark), prefix)
            if len(files) == 0:
                raise Exception("Empty experiment for prefix '{}' for benchmark '{}'.".format(prefix, benchmark))
            experiments[benchmark][experiment] = load_results(files)

    return experiments

def falsified(result):
    return any(step.success for step in result.step_results)

def falsification_rate(experiment):
    if len(experiment) == 0:
        return None

    return sum(1 if falsified(result) else 0 for result in experiment)

def first_falsification(replica):
    _, _, Y = replica.test_repository.get()
    Y = np.array(Y).reshape(-1, 1)
    for i in range(len(Y)):
        if min(Y[i]) <= 0.0:
            return i + 1

    return None

def total_times(replica):
    t = 0
    for i in range(replica.test_repository.tests):
        performance = replica.test_repository.performance(i)
        try:
            t += performance.obtain("execution_time")
        except:
            pass
        try:
            t += performance.obtain("generation_time")
        except:
            pass
        try:
            t += performance.obtain("training_time")
        except:
            pass

    return t

def nonexecution_times(replica):
    result = []
    for i in range(replica.test_repository.tests):
        performance = replica.test_repository.performance(i)
        t = 0.0
        try:
            t += performance.obtain("generation_time")
        except:
            pass
        try:
            t += performance.obtain("training_time")
        except:
            pass
        result.append(t)

    return result

def time_ratio(replicas):
    execution = 0.0
    other = 0.0
    for replica in replicas:
        for i in range(replica.test_repository.tests):
            performance = replica.test_repository.performance(i)
            try:
                execution += performance.obtain("execution_time")
            except:
                pass
            try:
                other += performance.obtain("generation_time")
            except:
                pass
            try:
                other += performance.obtain("training_time")
            except:
                pass

    return execution / (execution + other)

def survival(data, label="", censoring_threshold=300):
    executions = [result.test_repository.tests if falsified(result) else float("inf") for result in data]
    censored = [True if x != float("inf") else False for x in executions]
    executions = [x if x != float("inf") else censoring_threshold for x in executions]

    kmf = lifelines.KaplanMeierFitter()
    kmf.fit(executions, event_observed=censored, label=label)
    FR = 1 - kmf.survival_function_at_times(censoring_threshold).iloc[0]
    FR_UB, FR_LB = 1 - kmf.confidence_interval_.iloc[-1,0], 1 - kmf.confidence_interval_.iloc[-1,1]

    return FR, FR_LB, FR_UB, kmf

def survival2(data, label="", censoring_threshold=300):
    executions = data
    censored = [True if x != float("inf") else False for x in executions]
    executions = [x if x != float("inf") else censoring_threshold for x in executions]

    kmf = lifelines.KaplanMeierFitter()
    kmf.fit(executions, event_observed=censored, label=label)
    FR = 1 - kmf.survival_function_at_times(censoring_threshold).iloc[0]
    FR_UB, FR_LB = 1 - kmf.confidence_interval_.iloc[-1,0], 1 - kmf.confidence_interval_.iloc[-1,1]

    return FR, FR_LB, FR_UB, kmf

def logrank(data_A, data_B, censoring_threshold=300):
    executions_A = [result.test_repository.tests if falsified(result) else float("inf") for result in data_A]
    censored_A = [True if x != float("inf") else False for x in executions_A]
    executions_A = [x if x != float("inf") else censoring_threshold for x in executions_A]

    executions_B = [result.test_repository.tests if falsified(result) else float("inf") for result in data_B]
    censored_B = [True if x != float("inf") else False for x in executions_B]
    executions_B = [x if x != float("inf") else censoring_threshold for x in executions_B]

    T = lifelines.statistics.logrank_test(executions_A, executions_B, event_observed_A=censored_A, event_observed_B=censored_B)

    return T.p_value

def set_boxplot_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)

def own_boxplot(data, x_labels, title="", ylabel="", line=None):
    fig = plt.figure(figsize=(2*len(data), 5))
    plt.title(title)
    plt.ylabel(ylabel)
    bp = bp = plt.boxplot(data, labels=x_labels)
    set_boxplot_color(bp, "black")

    if line is not None:
        plt.axhline(line, c="r")

    plt.tight_layout()
    return plt

# %% [markdown]
# ## Load Experiments

# %% [markdown]
An experiment consists of several replicas of STGEM runs on a given benchmark.
The replicas of an experiment correspond to files on a certain path with file
names having a common prefix. Currently the file organization is as follows.
The subdirectories of the base directory (by default `output`) correspond to
the benchmarks. Whenever a replica prefix is specified, all files (including
subdirectories) under `output/benchmark` that have the matching prefix
are collected into one experiment.

# %%
# Default path containing subdirectory for each benchmark.
output_path_base = os.path.join("..", "..", "stgem", "output")

# Which benchmarks are to be included.
benchmarks = ["AFC27", "AT1", "AT6A", "AT6B", "AT6C", "ATX2", "ATX61", "ATX62", "CC3", "CC4", "F16", "NN", "NNB", "PM"]

# Replica prefixes for collecting the experiments.
# Notice that not all results necessarilyy fit into memory!
experiments_description = {benchmark: {"OGAN": "{}_ARCH_OGAN_2".format(benchmark), "OGAN_LHS": "{}_ARCH_OGAN_LHS".format(benchmark), "RANDOM": "{}_ARCH_RANDOM".format(benchmark)} for benchmark in benchmarks}
for benchmark in benchmarks:
    if not benchmark in ["ATX2", "ATX61", "ATX62"]: continue
    experiments_description[benchmark]["ARCH23"] = "{}_ARCH23_OGAN".format(benchmark)
experiments = loadExperiments(output_path_base, benchmarks, experiments_description)
# This is for formatting.
L = max(len(benchmark) + len(experiment) + 1 for benchmark in benchmarks for experiment in experiments[benchmark] for benchmark in benchmarks)
latex = {
    "AFC27": r"$\mathrm{AFC27}$",
    "AT1": r"$\mathrm{AT1}_{20}$",
    "AT6A": r"$\mathrm{AT6}_{4,35,3000}$",
    "AT6B": r"$\mathrm{AT6}_{8,50,3000}$",
    "AT6C": r"$\mathrm{AT6}_{20,65,3000}$",
    "ATX61": r"$\mathrm{AT6}_{30,80,4500}$",
    "ATX62": r"$\mathrm{AT6}_{30,50,2700}$",
    "ATX2": r"$\mathrm{ATX2}$",
    "CC3": r"$\mathrm{CC3}$",
    "CC4": r"$\mathrm{CC4}$",
    "CCX": r"$\mathrm{CCX}$",
    "F16": r"$\mathrm{F16}$",
    "NN": r"$\mathrm{NN}_{0.03}$",
    "NNB": r"$\mathrm{NN}_{0.04}$",
    "NNX": r"$\mathrm{NNX}$",
    "PM": r"$\mathrm{PM}$"
}

# %% [markdown]
# ## Falsification Rate and First Falsifications

# %%
print("Experiment:" + " "*(L + 2 - 11) + "Falsification rate:")
print("-"*(L + 1 + 20))
for benchmark in benchmarks:
    for experiment in experiments[benchmark]:
        l = len(benchmark) + len(experiment) + 1
        FR = falsification_rate(experiments[benchmark][experiment])
        print("{}/{}".format(benchmark, experiment) + " "*(L - l + 2) +  "{}/{}".format(FR, len(experiments[benchmark][experiment])))

# %%
B = 300
label = {
    "OGAN": "OGAN US",
    "OGAN_LHS": "OGAN LHS",
    "RANDOM": "RANDOM"
}

for benchmark in benchmarks:
    FR = {}; LB = {}; UB = {}; KMF = {}
    l = max(len(experiment) for experiment in experiments[benchmark])
    E = copy.copy(experiments[benchmark])
    if "ARCH23" in E:
        del E["ARCH23"]
    for experiment in E:
        A, B, C, D = survival(experiments[benchmark][experiment], label=label[experiment])
        FR[experiment] = A
        LB[experiment] = B
        UB[experiment] = C
        KMF[experiment] = D

    # Print falsification rates and confidence intervals.
    print("{}:".format(benchmark) + " "*(l - len(benchmark) + 1) + "FR:  CI:")
    print("–"*80)
    for experiment in E:
        print("{}".format(experiment) + " "*(l - len(experiment) + 2) + "{}  [{}, {}]".format(round(FR[experiment], 2), round(LB[experiment], 2), round(UB[experiment], 2)))

    # Logrank p-values.
    print()
    for experiment_A in E:
        for experiment_B in E:
            if experiment_A == experiment_B: continue
            p_value = logrank(experiments[benchmark][experiment_A], experiments[benchmark][experiment_B])
            print("{} VS. {}, logrank p-value: {}".format(experiment_A, experiment_B, round(p_value, 3)))
    print()

    # Plot the survival functions.
    fig = plt.figure()
    fig.suptitle(latex[benchmark] if benchmark in latex else "")

    ax = plt.subplot(111)
    ax.set_xlim([0,300])
    ax.set_ylim([0,1.03])

    colors = {
        "RANDOM":   "#4477AA",
        "OGAN":     "#228833",
        "OGAN_LHS": "#CCBB44"
    }
    for experiment in E:
        KMF[experiment].plot_survival_function(ax=ax, c=colors[experiment])

    plt.xlabel("")

    file_name = "{}_base.pdf".format(benchmark)
    plt.savefig(file_name, pad_inches=0.1, dpi=150)
    plt.show()

# %%
print("Experiment:" + " "*(L + 2 - 11) + "Mean:  SD:")
print("-"*(L + 1 + 14))
for benchmark in benchmarks:
    data = []
    labels = experiments_description[benchmark]
    for experiment in experiments[benchmark]:
        l = len(benchmark) + len(experiment) + 1
        FF = np.array([first_falsification(replica) for replica in experiments[benchmark][experiment]])
        data.append(FF[FF != None])
        if len(data[-1]) > 0:
            mean = round(np.mean(data[-1]), 2)
            sd = round(np.std(data[-1]), 2)
        else:
            mean = 0
            sd = 0
        print("{}/{}".format(benchmark, experiment) + " "*(L - l + 2) + str(mean) + " "*(7 - len(str(mean))) + str(sd))

    own_boxplot(data, labels, title="First Falsifications {}".format(benchmark), ylabel="First falsification", line=75)

# %% [markdown]
# ## Times

# %%
print("Experiment:" + " "*(L + 2 - 11) + "Mean Total Time:  Mean Generation Time:  Execution Ratio:")
print("-"*(L + 1 + 58))
for benchmark in benchmarks:
    for experiment in experiments[benchmark]:
        l = len(benchmark) + len(experiment) + 1
        T = np.array([total_times(replica) for replica in experiments[benchmark][experiment]])
        GT = np.array(sum((nonexecution_times(replica) for replica in experiments[benchmark][experiment]), start=[]))
        R = round(time_ratio(experiments[benchmark][experiment]), 2)
        V1 = round(np.mean(T), 1)
        #V2 = round(np.mean(GT), 4)
        V2 = np.mean(GT)
        print("{}/{}".format(benchmark, experiment) + " "*(L - l + 2) + "{}             {}                 {}".format(V1, V2, R))

# %% [markdown]
# ## OGAN Discriminator Accuracy

# %%
def set_seed(seed):
    import random, torch
    torch.use_deterministic_algorithms(mode=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def score_spearman(model, tests, test_repository):
    """For the given test indices from the test repository, computes the ground
    truth objective values (ordered from min to max) and the corresponding
    model objective estimates. Returns the Spearman correlation between these
    two sequences and the 95% confidence interval for it obtained using
    bootstrapping."""

    # Order the tests by robustness and get estimates from the model.
    tests.sort(key=lambda i: test_repository.get(i)[2][0])
    A = []
    B = []
    for i in tests:
        X, _, Y = test_repository.get(i)
        test = np.array(X.inputs).reshape(1, -1)
        estimate = model.predict_objective(test)[0][0]
        A.append(Y[0])
        B.append(estimate)

    statistic = stats.spearmanr(A, B)

    # Use bootstrapping to find 95% confidence interval.
    def tmp(X, Y):
        return stats.spearmanr(X, Y).correlation

    ci = stats.bootstrap(data=(A, B),
                         statistic=tmp,
                         vectorized=False,
                         paired=True,
                         n_resamples=9999,
                         method="percentile")

    low = ci.confidence_interval.low
    high = ci.confidence_interval.high

    return statistic.correlation, low, high

def weighted_correlation(X, Y, weights):
    """Computes a weighted correlation between X and Y according to the given
    weights."""

    W = np.sum(weights)

    meanX = np.sum(weights*X) / W
    meanY = np.sum(weights*Y) / W

    varianceX = np.sum(weights * (X - meanX)**2) / W
    varianceY = np.sum(weights * (Y - meanY)**2) / W

    covariance = np.sum(weights * (X - meanX) * (Y - meanY)) / W
    
    correlation = covariance / (np.sqrt(varianceX) * np.sqrt(varianceY))

    return correlation

def weighted_spearman_correlation(X, Y, weights):
    """Spearman correlation for weighted correlation."""

    RX = stats.rankdata(X)
    RY = stats.rankdata(Y)

    return weighted_correlation(RX, RY, weights)

def score_biased_spearman(model, tests, test_repository):
    """Same as score_spearman but for weighted correlation."""

    # TODO: The code is a bit messy and assumes that the tests have been
    # selected in such a way that the first K tests belong to robustness bin
    # [0, 1/B] etc.

    # Order the tests by robustness and get estimates from the model.
    tests.sort(key=lambda i: test_repository.get(i)[2][0])
    A = []
    B = []
    for i in tests:
        X, _, Y = test_repository.get(i)
        test = np.array(X.inputs).reshape(1, -1)
        estimate = model.predict_objective(test)[0][0]
        A.append(Y[0])
        B.append(estimate)

    # Prepare the weights.
    bins = 10
    bin_weights = [[1 - ((1/bins)*i)**(1/4)] for i in range(bins)]
    # Here K = 50.
    weights = [50*W for W in bin_weights]
    weights = sum(weights, start=[])

    statistic = weighted_spearman_correlation(A, B, weights)

    # Use bootstrapping to find 95% confidence interval.
    ci = stats.bootstrap(data=(A, B, weights),
                         statistic=weighted_spearman_correlation,
                         vectorized=False,
                         paired=True,
                         n_resamples=9999,
                         method="percentile")
    low = ci.confidence_interval.low
    high = ci.confidence_interval.high

    return statistic, low, high

def score_model(model, tests, test_repository):
    score, ci_l, ci_h = score_spearman(model, tests, test_repository)
    return score, ci_l, ci_h

def score_model_biased(model, tests, test_repository):
    score, ci_l, ci_h = score_biased_spearman(model, tests, test_repository)
    return score, ci_l, ci_h

def score_model(model, test_repository):
    X, _, Y = test_repository.get()
    X = np.array([x.inputs for x in X])
    Y = np.array(Y)
    estimates = model.predict_objective(X)
    score = np.sum(np.abs(estimates - Y)) / len(Y)
    return score

def get_benchmark_id(benchmark):
    prefixes = ["AFC", "AT", "CC", "F16", "NN", "PM"]
    for prefix in prefixes:
        if benchmark.startswith(prefix):
            return prefix
    return None

def get_generator(benchmark, experiment):
    """Gets the STGEM generator for the given benchmark and experiment."""

    benchmark_id = get_benchmark_id(benchmark)

    benchmark_module = importlib.import_module("{}.benchmark".format(benchmark_id.lower()))

    if experiment == "OGAN":
        setup = "default"
    else:
        setup = "default_lhs"

    mode = "normal"
    if benchmark_id in ["CC", "NN"]:
        mode = benchmark
    if benchmark == "NNB":
        mode = "NN"

    sut_factory, objective_factory = get_sut_objective_factory(benchmark_module=benchmark_module, selected_specification=benchmark, mode=mode)
    step_factory = lambda N_objectives=1, setup="default": benchmark_module.step_factory(N_objectives=N_objectives, setup=setup)
    generator_factory = get_generator_factory(description="", sut_factory=sut_factory, objective_factory=objective_factory, step_factory=step_factory)
    return generator_factory()

def get_balanced_validation_set(benchmark, experiment, maximum, nonempty, seed, bins=10, tests_per_bin=25):
    """For the given benchmark and experiment, find a balanced validation set
    of specified amount of bins with the specified amount of tests per bin."""

    from stgem.algorithm.ogan.algorithm import OGAN
    from stgem.algorithm.ogan.model import OGAN_Model
    from stgem.algorithm.random.algorithm import Random
    from stgem.algorithm.random.model import Uniform
    from stgem.generator import Search, Load
    from stgem.test_repository import TestRepository
    from stgem.sut import SearchSpace, SUTInput

    file_name = os.path.join(output_path_base, benchmark, "{}_BALANCED_VALIDATION.pickle.gz".format(benchmark))

    if os.path.exists(file_name):
        result = STGEMResult.restore_from_file(file_name)
    else:
        get_bin = lambda x: int(x/(maximum/bins)) if x < maximum else bins - 1
        generator = get_generator(benchmark, experiment)

        algorithm = generator.steps[0].algorithm
        sut = generator.sut
        sut_parameters = generator.sut.parameters
        sut.setup()
        objectives = generator.objectives

        search_space = SearchSpace()
        search_space_rng = np.random.RandomState(seed=seed)
        search_space.setup(sut=sut, objectives=objectives, rng=search_space_rng)
        algorithm.setup(
            search_space=search_space,
            device=None,
            logger=lambda x, y: None)
        for objective in objectives:
            objective.setup(sut)

        counts = {i:[] for i in range(bins)}
        test_repository = TestRepository()
        c = 0
        M = 0
        while np.min([len(counts[i]) for i in nonempty]) < tests_per_bin:
            test_repository.new_record()
            next_test = algorithm.generate_next_test(test_repository, 1)
            sut_input = SUTInput(next_test, None, None)
            sut_output = sut.execute_test(sut_input)
            Y = [objective(sut_input, sut_output) for objective in objectives]

            if Y[0] > M:
                M = Y[0]
                print("Max {} at {}".format(M, c))

            idx = get_bin(Y[0])
            if len(counts[idx]) < tests_per_bin:
                test_repository.record_input(sut_input)
                test_repository.record_output(sut_output)
                test_repository.record_objectives(Y)
                j = test_repository.finalize_record()
                counts[idx].append(j)

            if c % 150 == 3:
                print(c, [len(counts[i]) for i in nonempty])

            c += 1

        result = STGEMResult(description="",
                             sut_name=benchmark,
                             sut_parameters=sut_parameters,
                             seed=seed,
                             test_repository=test_repository,
                             step_results=[]
        )
        result.dump_to_file(file_name)

    return result

# %%
# The maxima ond offsets have been found by running
# get_balanced_validation_set for a while.
maxima = {
    "AFC27": 0.9596259492703892,
    "AT1": 0.7586621608253636,
    "AT6A": 0.9863087669968283,
    "AT6B": 0.9856003417828454,
    "AT6C": 0.9854430785628862,
    "ATX61": 0.8831127486178746,
    "ATX62": 0.9871210791322222,
    "ATX2": 0.9388484046189519,
    "CC3": 0.018518518518408463,
    "CC4": 0.00815271098199732,
    "F16": 0.16715235898915404,
    "NN": 0.912144555743891,
    "NNB": 0.9004185917999927,
    "PM": 0.375
}
nonempty = {
    "AFC27": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "AT1": [2, 3, 4, 5, 6, 7],
    "AT6A": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "AT6B": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "AT6C": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "ATX61": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "ATX62": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "ATX2": [0, 1, 2, 3, 4, 5, 6, 7],
    "CC3": [2, 3, 4, 5, 6, 7, 8, 9],
    "CC4": [8, 9],
    "F16": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "NN": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "NNB": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "PM": [0, 1, 3, 5, 6, 9]
}

for benchmark in benchmarks:
    for experiment in experiments[benchmark]:
        if experiment in ["RANDOM", "ARCH23"]: continue
        seed = 982734987
        E = experiment.split("_")[0] if "_" in experiment else experiment
        validation = get_balanced_validation_set(benchmark, experiment, maxima[benchmark], nonempty[benchmark], seed=seed)

        print("{}/{}:".format(benchmark, experiment))
        print("Got balanced validation set for {}/{}.".format(benchmark, E))
        #print()

        # Score the OGAN models.
        scores = []
        for result in experiments[benchmark][experiment]:
            # Obtain the trained model.
            model = result.step_results[1].parameters["model"]

            score = score_model(model, validation.test_repository)
            scores.append(score)

        #print(scores)
        print()
        print("Mean: {}, SD: {}".format(np.mean(scores), np.std(scores)))
        print()

# %% [markdown]
# ## Nonadaptive OGAN

# %%
B = 300
for benchmark in benchmarks:
    for experiment in ["OGAN", "OGAN_LHS"]:
        # Get nonadaptive OGAN results for the current requirement.
        prefix = "{}_ARCH_OGAN_SCORE".format(benchmark) + ("_LHS" if experiment == "OGAN_LHS" else "_2")
        random_results = loadExperiments(output_path_base, [benchmark], {benchmark: {"OGAN_RANDOM": prefix}})[benchmark]["OGAN_RANDOM"]

        # Find the number of tests needed for falsification for nonadaptive
        # OGAN and the current experiment.
        executions_random = [result.test_repository.tests if falsified(result) else float("inf") for result in random_results]
        executions_current = [result.test_repository.tests if falsified(result) else float("inf") for result in experiments[benchmark][experiment]]

        censored_random = [True if x != float("inf") else False for x in executions_random]
        censored_current = [True if x != float("inf") else False for x in executions_current]
        executions_random = [x if x != float("inf") else B for x in executions_random]
        executions_current = [x if x != float("inf") else B for x in executions_current]

        survival_random = survival2(executions_random, censoring_threshold=B)
        survival_current = survival2(executions_current, censoring_threshold=B)

        X = executions_random
        Y = executions_current

        """
        # Find the p-value and effect size. When LHS is used, then the samples
        # are independent, so Wilcoxon–Mann–Whitney test is used with XXX
        # effect size. When LHS is not used, then the samples are dependent and
        # Wilcoxon signed rank test is used with matched-pairs rank biserial
        # correlation coefficient for effect size.
        if experiment == "OGAN_LHS":
            label = "RANDOM_OGAN_LHS"
            # Compute the p-values. The alternative hypothesis is that the
            # current experiment has smaller number of executions than the
            # nonadaptive OGAN.
            p = stats.mannwhitneyu(Y, X, alternative="less")

            # TODO: Figure out the appropriate effect size.
            r = 0

            p = p.pvalue
        else:
            label = "NONADAPTIVE_OGAN"
            # Total rank sum.
            S = len(Y) * (len(Y) + 1) // 2

            # Compute the p-values. The alternative hypothesis is that the
            # current experiment has smaller number of executions than the
            # nonadaptive OGAN.
            p = stats.wilcoxon(Y, X, alternative="less")

            # Effect size as matched-pairs rank-biserial correlation. -1 means
            # that OGAN is perfectly smaller than random surrogate, 0 no
            # effect, etc.
            r = 2*p.statistic / S - 1

            p = p.pvalue
        """

        FR_random = falsification_rate(random_results) / len(random_results)
        kmf_random = lifelines.KaplanMeierFitter()
        kmf_random.fit(executions_random, event_observed=censored_random, label="Nonadaptive OGAN")
        UB_random, LB_random = 1 - kmf_random.confidence_interval_.iloc[-1,0], 1 - kmf_random.confidence_interval_.iloc[-1,1]

        FR_current = falsification_rate(experiments[benchmark][experiment]) / len(experiments[benchmark][experiment])
        kmf_current = lifelines.KaplanMeierFitter()
        kmf_current.fit(executions_current, event_observed=censored_current, label="Adaptive OGAN")
        UB_current, LB_current = 1 - kmf_current.confidence_interval_.iloc[-1,0], 1 - kmf_current.confidence_interval_.iloc[-1,1]

        effect_size = abs(FR_current - FR_random)

        print("{}/{}:".format(benchmark, experiment))
        print("–"*80)
        print("Nonadaptive OGAN data:")
        print(X)
        print("Nonadaptive OGAN FR (95% CI):")
        print("{} [{}, {}]".format(FR_random, LB_random, UB_random))
        print()
        print("OGAN data:")
        print(Y)
        print("Nonadaptive OGAN FR (95% CI):")
        print("{} [{}, {}]".format(FR_current, LB_current, UB_current))
        print()
        print("Effect size: {}".format(effect_size))
        print()

        T = lifelines.statistics.logrank_test(executions_random, executions_current, event_observed_A=censored_random, event_observed_B=censored_current)
        p_value = T.p_value

        print("Logrank p-value: {}".format(p_value))
        print()

        """
        R_current = survival_current[-1] + (survival_current[0] - survival_current[-1])/2
        print(R_current)
        for i in range(len(survival_current)):
            if survival_current[i] <= R_current:
                print("Current 'median' = {}".format(i))
                break
        R_random = survival_random[-1] + (survival_random[0] - survival_random[-1])/2
        print(R_random)
        for i in range(len(survival_random)):
            if survival_random[i] <= R_random:
                print("Random 'median' = {}".format(i))
                break
        print()
        """
        #X = [x for x in X if x < 300]
        #Y = [y for y in Y if y < 300]
        # Plot survival curves.
        """
        plt.figure()
        plt.plot(np.arange(300), survival_random, label="OGAN RANDOM")
        plt.plot(np.arange(300), survival_current, label="OGAN")
        plt.legend()
        """
        fig = plt.figure()
        fig.suptitle("{} / OGAN {}".format(latex[benchmark], "US" if experiment == "OGAN" else "LHS"))

        ax = plt.subplot(111)
        ax.set_xlim([0,300])
        ax.set_ylim([0,1.03])

        kmf_random.plot(ax=ax, c="#CCBB44")
        kmf_current.plot(ax=ax, c="#228833")

        plt.xlabel("")

        file_name = "{}_{}.pdf".format(benchmark, experiment)
        plt.savefig(file_name, pad_inches=0.1, dpi=150)
        plt.show()

        # Plot first falsifications.
        #own_boxplot([X, Y], [label, experiment], title="First Falsifications {}".format(benchmark), ylabel="First falsification", line=75)

# %% [markdown]
# ## ARCH23 Data Analysis

# %%
# We assume that the zip file from https://zenodo.org/record/8024426 has been
# extracted to this path.
arch23_path = "2023-ARCH-COMP-Falsification"
arch23_benchmarks = ["AFC27", "AT1", "AT6A", "AT6B", "AT6C", "CC3", "CC4", "F16", "NN", "NNB", "PM"]
tools = {
    "ARISTEO":      ["Aristeo_ARCH2023_Instance2", "Aristeo_ARCH2023_Instance2.csv"],
    "ATHENA":       ["Athena_ARCH2023_Instance2", "Athena_ARCH2023_Instance2.csv"],
    "ATHENA_B":     ["Athena_ARCH2023_Instance1", "Athena_ARCH2023_Instance1.csv"],
    "FALCAUN":      ["FalCAuN_ARCH2023_Final", "FalCAuN_ARCH2023_Final.csv"],
    "FORESEE":      ["ForeSee_ARCH2023_Final", "ForeSee_ARCH2023_Final.csv"],
    "NNFAL":        ["NNFal_ARCH-COMP_2023_v2", "NNFal_ARCH-COMP_2023_v2.csv"],
    "PSY-TALIRO":   ["BO_ARCH2023_06052023_inst2", "BO_ARCH2023_06052023_inst2.csv"],
    "PSY-TALIRO_B": ["LSemiBO_ARCH2023_06052023_inst2", "LSemiBO_ARCH2023_06052023_inst2.csv"],
    "STGEM":        ["STGEM_ARCH-COMP2023_Instance2", "STGEM_ARCH-COMP2023_Instance2.csv"]
}
map_benchmark = {
    "ARISTEO": {
        "AFC27": "AFC27",
        "AT1":   "AT1",
        "AT6A":  "AT6a",
        "AT6B":  "AT6b",
        "AT6C":  "AT6c",
        "CC3":   "CC3",
        "CC4":   "CC4",
        "CCX":   "CCx",
        "NN":    "NN",
        "NNB":   "(NN",
        "NNX":   "NNx",
        "PM":    "PM"
    },
    "ATHENA": {
        "AFC27": "AFC27",
        "AT1":   "AT1",
        "AT6A":  "AT6a",
        "AT6B":  "AT6b",
        "AT6C":  "AT6c",
        "CC3":   "CC3",
        "CC4":   "CC4",
        "CCX":   "CCx",
        "NN":    "NN",
        "NNB":   "(NN",
        "NNX":   "NNx",
        "PM":    "PM"
    },
    "ATHENA_B": {
        "F16": "F16"
    },
    "FALCAUN": {
        "AT1":   "AT1",
        "AT6C":  "AT6c"
    },
    "FORESEE": {
        "AFC27": "(AFC27",
        "AT1":   "AT1",
        "AT6A":  "AT6a",
        "AT6B":  "AT6b",
        "AT6C":  "AT6c",
        "CC3":   "car3",
        "CC4":   "car4",
        "CCX":   "ccx",
        "NN":    "(NN 0.005 0.03)",
        "PM":    "PM"
    },
    "NNFAL": {
        "AT1":   "AT1",
        "AT6C":  "AT6c"
    },
    "PSY-TALIRO": {
        "AFC27": "AFC27",
        "AT1":   "AT1",
        "AT6A":  "AT6a",
        "AT6B":  "AT6b",
        "AT6C":  "AT6c",
        "CC3":   "CC3",
        "CC4":   "CC4",
        "NN":    "NN",
        "NNX":   "NNx",
        "PM":    "PM"
    },
    "PSY-TALIRO_B": {
        "CCX":   "CCx",
        "NNX":   "NNx"
    },

    "STGEM": {
        "AFC27": "AFC27",
        "AT1":   "AT1",
        "AT6A":  "AT6A",
        "AT6B":  "AT6B",
        "AT6C":  "AT6C",
        "CC3":   "CC3",
        "CC4":   "CC4",
        "CCX":   "CCX",
        "F16":   "F16",
        "NN":    "NN",
        "NNB":   "NNB",
        "NNX":   "NNX",
        "PM":    "PM"
    }
}
labels = {
    "ARISTEO":      r"ARIsTEO",
    "ATHENA":       r"ATheNA",
    "ATHENA_B":     r"ATheNA",
    "FALCAUN":      r"FalCAuN",
    "FORESEE":      r"FORESEE",
    "NNFAL":        r"NNFal",
    "PSY-TALIRO":   r"$\Psi$-TaLiRo",
    "PSY-TALIRO_B": r"$\Psi$-TaLiRo",
    "STGEM":        r"STGEM"
}

KMF = {benchmark:[] for benchmark in arch23_benchmarks}
for tool, (folder, csv_file) in tools.items():
    full_data = pandas.read_csv(os.path.join(arch23_path, folder, csv_file))

    print("{}:".format(tool))
    print("-"*79)

    for benchmark in arch23_benchmarks:
        if not benchmark in map_benchmark[tool].keys(): continue
        if benchmark not in ["AT6A", "NN"]:
            data = full_data[full_data["property"].str.startswith(map_benchmark[tool][benchmark])]
            if tool == "FALCAUN":
                # This apparently is the way to obtain the report data.
                data = data[data["simulations"] < 400]
        else:
            data = full_data[full_data["property"] == map_benchmark[tool][benchmark]]
            if benchmark == "NN" and tool == "FORESEE":
                # This apparently is the way to obtain the report data.
                data = data[10:]
        # We correct the STGEM PM benchmark data which incorrectly states that
        # non of the replicas managed to falsify the requirement.
        if tool == "STGEM" and benchmark == "PM":
            data["falsified"] = "yes"
        if len(data) == 0: continue
        executions = [int(x) for x in data[data["falsified"] == "yes"]["simulations"]] + [float("inf")]*len(data[data["falsified"] == "no"])
        executions += [float("inf")] * (10 - len(executions))
        mean = np.mean([x for x in executions if x != float("inf")])

        FR, FR_LB, FR_UB, kmf = survival2(executions, label=labels[tool], censoring_threshold=1500)
        KMF[benchmark].append((tool, kmf))

        print("{}: {}, [{}, {}], {}, {}".format(benchmark, FR, round(FR_LB, 2), round(FR_UB, 2), round(mean, 1), executions))

    print()

for benchmark in arch23_benchmarks:
    fig = plt.figure()
    fig.suptitle(latex[benchmark])

    ax = plt.subplot(111)
    ax.set_xlim([0,1500])
    ax.set_ylim([0,1.03])

    # Paul Tol's bright.
    colors = {
        "ARISTEO":      "#4477AA",
        "ATHENA":       "#EE6677",
        "ATHENA_B":     "#EE6677",
        "FALCAUN":      "#228833",
        "FORESEE":      "#CCBB44",
        "NNFAL":        "#66CCEE",
        "PSY-TALIRO":   "#AA3377",
        "PSY-TALIRO_B": "#AA3377",
        "STGEM":        "#BBBBBB"
    }
    for tool, kmf in KMF[benchmark]:
        kmf.plot_survival_function(ax=ax, ci_show=False, c=colors[tool])

    plt.xlabel("")

    file_name = "{}_arch23.pdf".format(benchmark)
    plt.savefig(file_name, pad_inches=0.1, dpi=150)
    plt.show()

# %%
for benchmark in benchmarks:
    if not "ARCH23" in experiments[benchmark]: continue
    results = experiments[benchmark]["ARCH23"]
    _, _, _, KMF = survival(results, label="STGEM", censoring_threshold=1500)

    # Plot the survival functions.
    fig = plt.figure()
    fig.suptitle(latex[benchmark])

    ax = plt.subplot(111)
    ax.set_xlim([0,1500])
    ax.set_ylim([0,1.03])

    colors = {
        "STGEM":      "#BBBBBB"
    }
    KMF.plot_survival_function(ax=ax, ci_show=False, c=colors["STGEM"])

    plt.xlabel("")

    file_name = "{}_arch23.pdf".format(benchmark)
    plt.savefig(file_name, pad_inches=0.1, dpi=150)
    plt.show()

# %% [markdown]
# ## Zenodo CSV

# %%
append = True

f = open("zenodo.csv", mode="a" if append else "w")
# Header.
if not append:
    f.write('"benchmark","algorithm","replica","iteration","input","robustness","scaled robustness","generation time",training time","execution_time"\n')
# The data.
for benchmark in benchmarks:
    # Get the unscaled objective.
    benchmark_id = get_benchmark_id(benchmark)
    mode = "normal"
    if benchmark_id in ["CC", "NN"]:
        mode = benchmark
    if benchmark == "NNB":
        mode = "NN"

    benchmark_module = importlib.import_module("{}.benchmark".format(benchmark_id.lower()))
    sut_factory, objective_factory = get_sut_objective_factory(benchmark_module, benchmark, mode)
    sut = sut_factory()
    objectives = objective_factory()
    for objective in objectives:
        objective.setup(sut)
        objective.scale = False

    for experiment in experiments[benchmark]:
        results = experiments[benchmark][experiment]
        for n, replica in enumerate(results):
            tr = replica.test_repository
            for k in range(tr.tests):
                X, Z, Y = tr.get(k)
                perf = tr.performance(k)

                input = list(X.inputs)
                robustness = min([objective(X, Z) for objective in objectives])
                scaled_robustness = Y[0]
                generation_time = perf.obtain("generation_time")
                training_time = perf.obtain("training_time")
                execution_time = perf.obtain("execution_time")

                s = '"{}","{}","{}","{}","{}","{}","{}","{}","{}","{}"\n'.format(benchmark, experiment,n + 1, k + 1, input, robustness, scaled_robustness, generation_time, training_time, execution_time)
                f.write(s)
f.close()

# %% [markdown]
# ## Visualizations

# %% [markdown]
# ### Visualize Test Inputs and Outputs

# %% [markdown]
Visualize tests (indices given in `idx`) from a replica. For signal inputs or
outputs, we draw the plots representing the signals. For vector inputs or
outputs, we simply print the vector components. The inputs are always
denormalized, that is, they are given in the format actually given to the SUT.
Outputs are always the outputs of the SUT unmodified.

# %%
benchmark = "CC"
experiment = "CC3"
replica_idx = [0]
test_idx = [0]

for i in replica_idx:
    for j in test_idx:
        plotTest(experiments[benchmark][experiment][i], j)

# %% [markdown]
# ### Visualization of 1-3D Vector Input Test Suites.

# %% [markdown]
This visualizes the test suites for SUTs which have vector input of dimension
$d$ with $d \leq 3$. The input space is represented as $[-1, 1]^d$ meaning that
inputs are not presented as denormalized to their actual ranges.

# %%
benchmark = "F16"
experiment = "F16"
idx = [0]

for i in idx:
    visualize3DTestSuite(experiments[benchmark][experiment], i)

# %% [markdown]
# ### Animate Signal Input/Output Test Suite

# %%
benchmark = "AT"
experiment = "ATX1_ATX1_10000"
replica_idx = 0

anim = animateResult(experiments[benchmark][experiment][replica_idx])
HTML(anim.to_jshtml())

# %%
benchmark = "AT"
experiment = "ATX62_ARCH_OGAN"
for result in experiments[benchmark][experiment]:
    D = [result.test_repository.performance(i)._record["diversifying"] for i in range(75, result.test_repository.tests)]
    print(D)

# %%
benchmark = "F16"
experiment = "F16_ARCH_OGAN_NOD"
print(len(experiments[benchmark][experiment]))

