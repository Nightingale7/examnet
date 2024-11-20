from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS
import heapq
import random
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
   
