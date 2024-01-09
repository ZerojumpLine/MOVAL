import scipy
import numpy as np
from itertools import repeat
import multiprocessing as mp
from moval.models.utils import cal_softmax

def solve_T(inp: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Derive the pseudo-temperature that match the process from inp to score.

    Note:
        I try to accelerate this process with multi-process.

    Args:
        inp: The network output (logits) of shape ``(n, d)``
        score: The calibrated scores of shape ``(n, )``    
    
    Returns:
        pseudo_t: The solved temperature of shape ``(n, )``

    """

    ksample = list(range(len(score)))
    mp_pool = mp.Pool(processes=np.minimum(8, len(score)))
    
    try:
        with mp_pool as pool:
            pseudo_t = pool.starmap(optimize_T, zip(ksample, repeat(inp), repeat(score)))
    except mp.TimeoutError:
        print("time out?")
    except:  # Catches everything, even a sys.exit(1) exception.
        mp_pool.terminate()
        mp_pool.join()
        raise Exception("Unexpected error.")
    else:  # Nothing went wrong
        # Needed in case any processes are hanging. mp_pool.close() does not solve this.
        mp_pool.terminate()
        mp_pool.join()

    return np.array(pseudo_t)

def optimize_T(ksample, inp, score):

    inp_k = inp[ksample: ksample + 1, :]
    score_k = score[ksample: ksample + 1]

    optimization_result = scipy.optimize.minimize(
        fun = lambda x: np.abs(np.max(cal_softmax(inp_k, x), axis = 1) - score_k),
        x0 = np.array([1.0]),
        method = 'Nelder-Mead',
        bounds = [(1e-06,None)],
        tol = 1e-07)
    
    return optimization_result.x[0]