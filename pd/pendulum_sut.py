import math

import numpy as np

from stgem.sut import SUT, SUTOutput

from pendulum_benchmark import pendulum_model

class Pendulum(SUT):
    """SUT for the Python version of the F16 problem. Notice that running this
    requires Python 2 with numpy to be installed. The parameters set in the
    script seem to be the same as in the Matlab m files. We assume that input
    and output ranges are set externally."""

    default_parameters = {
        "simulation_time": 15,
        "time_slices": [5, 5, 5],
        "sampling_step": 0.01
    }

    def __init__(self, parameters):
        SUT.__init__(self, parameters)

        self.input_type = "vector"
        self.output_type = "signal"

        mandatory_parameters = ["time_slices", "simulation_time", "sampling_step"]
        for p in mandatory_parameters:
            if not p in self.parameters:
                raise Exception("Parameter '{}' must be defined for piecewise constant signal inputs.".format(p))

        # How often input signals are sampled for execution (in time units).
        self.steps = int(self.simulation_time / self.sampling_step)
        # How many inputs we have for each input signal.
        self.pieces = [math.ceil(self.simulation_time / time_slice) for time_slice in self.time_slices]

        self.params = {
            "model": "pendulum",
            "eta": 0.223,
            "alpha": 0.028
        }

        self.has_been_setup = False

    def setup(self):
        super().setup()

        if self.has_been_setup: return

        if not len(self.time_slices) == self.idim:
            raise Exception("Expected {} time slices, found {}.".format(self.idim, len(self.time_slices)))

        self.N_signals = self.idim
        self.idim = sum(self.pieces)

        self.descaling_intervals = []
        for i in range(len(self.input_range)):
            for _ in range(self.pieces[i]):
                self.descaling_intervals.append(self.input_range[i])

        self.has_been_setup = True

    def _convert(self, test):
        # TODO: This should be generic and not really part of this class.
        denormalized = self.descale(test.inputs.reshape(1, -1), self.descaling_intervals).reshape(-1)

        # Common timestamps to all input signals.
        timestamps = np.array([i*self.sampling_step for i in range(self.steps + 1)])
        # Signals.
        signals = np.zeros(shape=(self.N_signals, len(timestamps)))
        offset = 0
        for i in range(self.N_signals):
            idx = lambda t: int(t // self.time_slices[i]) if t < self.simulation_time else self.pieces[i] - 1
            signal_f = lambda t: denormalized[offset + idx(t)]
            signals[i] = np.asarray([signal_f(t) for t in timestamps])
            offset += self.pieces[i]

        test.input_timestamps = timestamps
        test.input_denormalized = signals

    def _execute_test(self, test):
        # Convert the input (piecewise constant signal) into a signal.
        self._convert(test)

        timestamps, trajectories = pendulum_model(
            static=[],
            times=test.input_timestamps,
            signals=test.input_denormalized,
            params=self.params
        )

        print()
        print(list(trajectories))
        print()

        return SUTOutput(trajectories.T, timestamps, None, None)

