from scipy.stats import entropy
import numpy as np

def adjusted_log(x,log_base = 2):
    """
    Compute the logarithm of x with an arbitrary base.

    Parameters
    ----------
    x : float or array-like
        Input value(s) for which the logarithm will be computed.
    log_base : float, optional
        The base of the logarithm. Default is 2.

    Returns
    -------
    float or ndarray
        The logarithm of x with respect to the specified base.
    """
    return np.log(x)/np.log(log_base)



def renyi_entropy(probabilities, alpha, log_base = 2):
    """
    Compute the Rényi entropy of a probability distribution. Original paper: Rényi, Alfréd. "On measures of entropy and information.", 1961.

    Parameters
    ----------
    probabilities : array-like, shape (n,)
        A valid probability distribution (non-negative entries summing to 1).
    alpha : float (positive or null)
        The order of the Rényi entropy.
        - alpha = 1: Shannon entropy
        - alpha = 0: Hartley entropy
        - alpha = ∞: Min-entropy
        - otherwise: general Rényi entropy
    log_base : float, optional
        The base of the logarithm used in the entropy calculation, defines the units of the output. Default is 2 (bits).

    Returns
    -------
    float
        The Rényi entropy of order alpha for the given probability distribution.
    """
    probabilities = np.asarray(probabilities)
    assert np.min(probabilities) >= 0, "The input is not a probability distribution, check positivity"
    assert len(probabilities.shape) == 1, "The input needs to be a 1D array"
    assert np.sum(probabilities) == 1, "The input is not a probability distribution, check normalization"
    assert alpha >= 0, "alpha needs to be positive or null"

    if alpha == 1:
        # Special case: Shannon entropy
        return entropy(probabilities, base = log_base)
    elif alpha == 0:
        # Special case: Hartley entropy
        return adjusted_log(np.sum(probabilities > 0), base = log_base)
    elif alpha == np.inf:
        # Special case: Min-entropy
        return -adjusted_log(probabilities.max(), log_base)
    else:
        # General case: Rényi entropy
        return 1 / (1 - alpha) * adjusted_log(np.sum(probabilities ** alpha), log_base)