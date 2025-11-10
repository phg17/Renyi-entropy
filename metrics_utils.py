import numpy as np

def compute_AIC(V, W, H):
    '''
    Given a clustering with V estimated by the product of W and H (generative matrices, see NMF notation), return the Akaike Information Criterion
    '''
    V_est = W@H
    SSE_res = np.linalg.norm((V - V_est)**2)
    n = np.prod(V.shape)  # Total number of elements in V
    num_params = (W.shape[0] + H.shape[1]) * W.shape[1]  # (m + n) * r
    log_likelihood = -n / 2 * np.log(SSE_res)
    aic = 2 * num_params - 2 * log_likelihood

    return aic

def compute_R2(V, V_est):
    '''
    Given a matrix a its estimation, return the explained variance
    '''
    SSE_tot = np.linalg.norm((V - np.mean(V))**2)
    SSE_res = np.linalg.norm((V - V_est)**2)
    return 1 - SSE_res/SSE_tot