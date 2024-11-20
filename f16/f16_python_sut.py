#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib, os, subprocess, sys

import numpy as np

from stgem.sut import SUT, SUTOutput

class F16GCAS_PYTHON2(SUT):
    """SUT for the Python version of the F16 problem. Notice that running this
    requires Python 2 with numpy to be installed. The parameters set in the
    script seem to be the same as in the Matlab m files. We assume that input
    and output ranges are set externally."""

    def __init__(self, parameters):
        SUT.__init__(self, parameters)

        if not "initial_altitude" in self.parameters:
            raise Exception("Initial altitude not defined as a SUT parameter.")

        self.input_type = "vector"
        self.output_type = "vector"

    def _execute_test(self, test):
        denormalized = self.descale(test.inputs.reshape(1, -1), self.input_range).reshape(-1)

        output = subprocess.run(["f16/AeroBenchVVPython/check_gcas_v1.sh", str(self.initial_altitude), str(denormalized[0]) , str(denormalized[1]), str(denormalized[2]) ], capture_output=True)

        # Altitude on the last line.
        # TODO: Better error handling.
        try:
            v = float(str(output.stdout).split("\\n")[-2])
        except:
            v = self.initial_altitude

        test.input_denormalized = denormalized
        return SUTOutput(np.asarray([v]), None, None, None)

class F16GCAS_PYTHON3(SUT):
    """SUT for the Python 3 version of the F16 problem. The parameters set in
    the script seem to be the same as in the Matlab m files. We assume that
    input and output ranges are set externally."""

    default_parameters = {
        "initial_altitude": 4040,
        "input_range_preset": "full"
    }

    def __init__(self, parameters):
        SUT.__init__(self, parameters)

        self.inputs = ["phi", "theta", "psi"]
        self.outputs = ["altitude"]

        from math import pi
        if self.input_range_preset == "arch-comp-2021":
            roll_range = [0.2*pi, 0.2833*pi]
            pitch_range = [-0.4*pi, -0.35*pi]
            yaw_range = [-0.375*pi, -0.125*pi]
        elif self.input_range_preset == "part-x":
            roll_range = [0.2*pi, 0.2833*pi]
            pitch_range = [-0.5*pi, -0.54*pi]
            yaw_range = [0.25*pi, 0.375*pi]
        elif self.input_range_preset == "full":
            # It seems that this can occasionally break the SUT with failed
            # integrator status.
            roll_range = [-pi, pi]
            pitch_range = [-pi, pi]
            yaw_range = [-pi, pi]
        else:
            raise ValueError("Unknown input range preset '{}'.".format(self.input_range_preset))

        self.input_range = [roll_range, pitch_range, yaw_range]
        self.output_range = [[0, 4040]]

        try:
            # sys.path.append(os.path.join("f16", "AeroBenchVVPython", "v2", "code"))
            sys.path.append(os.path.join(os.path.dirname(__file__), "AeroBenchVVPython", "v2", "code"))
            self.f16 = importlib.import_module("aerobench.run_f16_sim")
        except ModuleNotFoundError:
            import traceback
            traceback.print_exc()
            print("Could not load run_f16_sim module for F16GCAS_PYTHON3 SUT.")
            raise SystemExit

        try:
            self.gcas = importlib.import_module("aerobench.examples.gcas.gcas_autopilot")
        except ModuleNotFoundError:
            import traceback
            traceback.print_exc()
            print("Could not load gcas_autopilot module for F16GCAS_PYTHON3 SUT.")
            raise SystemExit

        self.input_type = "vector"
        self.output_type = "signal"

    def _execute_test(self, test):
        denormalized = self.descale(test.inputs.reshape(1, -1), self.input_range).reshape(-1)

        # The code below is adapted from AeroBenchVVPython/v2/code/aerobench/examples/gcas/run_GCAS.py

        ### Initial Conditions ###
        power = 9 # engine power level (0-10)

        # Default alpha & beta
        alpha = np.deg2rad(2.1215) # Trim Angle of Attack (rad)
        beta = 0                # Side slip angle (rad)

        # Initial Attitude
        alt = self.initial_altitude        # altitude (ft)
        vt = 540          # initial velocity (ft/sec)
        phi = denormalized[0]           # Roll angle from wings level (rad)
        theta = denormalized[1]         # Pitch angle from nose level (rad)
        psi = denormalized[2]   # Yaw angle from North (rad)

        # Build Initial Condition Vectors
        # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
        init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
        tmax = 15 # simulation time

        ap = self.gcas.GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')

        step = 1/30
        res = self.f16.run_f16_sim(init, tmax, ap, step=step, extended_states=True)

        t = res["times"]
        altitude = res["states"][:, 11]
        test.input_denormalized = denormalized
        return SUTOutput(np.array([altitude]), np.array(t), None, None)

