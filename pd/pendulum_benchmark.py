import numpy as np
from numpy.typing import NDArray

from pendulum_v1 import run_pendulum as pendulum_model_seeded

def get_gt_over_eta_and_error_over_alpha(gt, ERR, eta, alpha):
    prob_gt_over_eta = []
    prob_err_over_alpha = []
    for i in range(gt.shape[0]):
        gt_at_t = gt[i, :]
        gt_at_t_over_eta = gt_at_t > eta
        prob = np.count_nonzero(gt_at_t_over_eta) / 100
        prob_gt_over_eta.append(prob)
        
        err_at_t = ERR[i, :]
        err_at_t_over_alpha = err_at_t > alpha
        prob = np.count_nonzero(err_at_t_over_alpha) / 100
        prob_err_over_alpha.append(prob)
        
    return prob_gt_over_eta, prob_err_over_alpha

def pendulum_model(static, times, signals, params):

    sims = 100

    if static:
        param1 = static[0]
    else:
        param1 = 1.0
        
    if params['model'] == 'pendulum':
        result, gt, err = pendulum_model_seeded(
            times, 
            signals, 
            max_time = max(times), 
            sims = sims, 
            etaMax = params["eta"], 
        )
    else:
        raise Exception("Unknown model")
    
    prob_gt_over_eta, prob_err_over_alpha = get_gt_over_eta_and_error_over_alpha(gt, err, params["eta"], params['alpha'])
    prob_err_over_alpha = np.array(prob_err_over_alpha)
    prob_gt_over_eta = np.array(prob_gt_over_eta)
    
    trajectories = np.concatenate((prob_gt_over_eta.reshape(-1, 1), prob_err_over_alpha.reshape(-1, 1)), axis=1)
    
    timestamps: NDArray[np.float_] = result[:,0]

    return timestamps, trajectories#, gt, err, result


