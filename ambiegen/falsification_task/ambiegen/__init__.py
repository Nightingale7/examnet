from stgem.algorithm.model import Model

class UniformDependent(Model):
    """Model for uniformly random search which does not select components
    independently."""

    default_parameters = {"rotate": False}

    def generate_test(self):
        # The components of the actual test are curvature values in the input
        # range (default [-0.07, 0.07]). Here we do not choose the components
        # of a test independently in [-1, 1] but we do as in the Frenetic
        # algorithm where the next component is in the range of the previous
        # value +- 0.05 (in the scale [-0.07, 0.07]).

        test = np.zeros(self.search_space.input_dimension)
        K = 1 if self.rotate else 0
        for i in range(K + 1):
            test[i] = np.random.uniform(-1, 1)
        for i in range(K + 1, len(test)):
            test[i] = max(-1, min(1, test[i - 1] + (0.05/0.07) * np.random.uniform(-1, 1)))

        return test

