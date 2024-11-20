'''
Stanley Bak
Python version of GCAS maneuver benchmark
'''

from math import pi
from numpy import deg2rad

from RunF16Sim import RunF16Sim
from PassFailAutomaton import FlightLimitsPFA, FlightLimits
from CtrlLimits import CtrlLimits
from LowLevelController import LowLevelController
from Autopilot import GcasAutopilot
from controlledF16 import controlledF16
import sys

def main():
    'main function'

    flightLimits = FlightLimits()
    ctrlLimits = CtrlLimits()
    llc = LowLevelController(ctrlLimits)
    ap = GcasAutopilot(llc.xequil, llc.uequil, flightLimits, ctrlLimits)
    pass_fail = FlightLimitsPFA(flightLimits)
    pass_fail.break_on_error = False

    ### Initial Conditions ###
    power = 9 # Power

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)


    alt = float(sys.argv[1])     # Initial Attitude
    phi=  float(sys.argv[2])     # Roll angle from wings level (rad)
    theta=  float(sys.argv[3])   # Pitch angle from nose level (rad)
    psi= float(sys.argv[4])      # Yaw angle from North (rad)
    print(alt,phi,theta,psi)
    Vt = 540                     # Pass at Vtg = 540;    Fail at Vtg = 550;

    # phi = (pi/2)*0.5           # Roll angle from wings level (rad)
    # theta = (-pi/2)*0.8        # Pitch angle from nose level (rad)
    # psi = -pi/4                # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initialState = [Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    # if not None will do animation. Try a filename ending in .gif or .mp4 (slow). Using '' will plot to the screen.

    # Select Desired F-16 Plant
    f16_plant = 'stevens' # 'stevens' or 'morelli'

    tMax = 15 # simulation time

    xcg_mult = 1.0 # center of gravity multiplier

    val = 1.0      # other aerodynmic coefficient multipliers
    cxt_mult = val
    cyt_mult = val
    czt_mult = val
    clt_mult = val
    cmt_mult = val
    cnt_mult = val

    multipliers = (xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult)

    der_func = lambda t, y: controlledF16(t, y, f16_plant, ap, llc, multipliers=multipliers)[0]

    try:
        passed, times, states, modes, ps_list, Nz_list, u_list,minAlt = RunF16Sim(\
            initialState, tMax, der_func, f16_plant, ap, llc, pass_fail, multipliers=multipliers)
    except:
        pass

    print "Simulation Conditions Passed: {}".format(passed)
    print "minimum altitude = {}".format(minAlt)
    print minAlt

if __name__ == '__main__':
    main()
