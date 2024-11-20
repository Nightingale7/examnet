#sys.path.append("./ambiegen_tools")
from ambiegen_tools.code_pipeline.dave2_executor import Dave2Executor
from code_pipeline.tests_generation import RoadTestFactory

def execute_dave(tests,seed):
    result_folder = "./BeamngResults/{}/".format(seed)
    map_size = 200
    time_budget = 200
    the_executor = Dave2Executor(result_folder, map_size, "./ambiegen_tools/dave2/dave2-2022.h5",
                                     time_budget=time_budget, oob_tolerance=0.95, max_speed=70,
                                     beamng_home=None, beamng_user=None, road_visualizer=None)
   
    for test in tests:
        rtest = RoadTestFactory.create_road_test(test)
        print("rtest val: {} - {}".format(type(rtest), type(rtest) is RoadTestFactory.RoadTest))
        test_outcome, description, execution_data = the_executor.execute_test(rtest)
        print("outcome: {}".format(test_outcome))


