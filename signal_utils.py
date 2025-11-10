import scipy.signal as signal
import numpy as np
from numpy.random import rand
from numpy import nan_to_num

def sparse_resample(sparse_signal, new_fs, current_fs):
    ratio = new_fs / current_fs
    if len(sparse_signal.shape) == 2:
      new_signal = np.zeros([sparse_signal.shape[0],int(sparse_signal.shape[1] * ratio + 1)])
      for i in range(sparse_signal.shape[0]):
        for j in range(sparse_signal.shape[1]):
          if sparse_signal[i,j] > 0:
            new_signal[i,int(j*ratio)] = sparse_signal[i,j]
      return new_signal
    
    elif len(sparse_signal.shape) == 1:
      new_signal = np.zeros(int(sparse_signal.shape[0] * ratio + 1))
      for i in range(len(sparse_signal)):
          if sparse_signal[i] > 0:
              new_signal[round(i*ratio)] = sparse_signal[i]
      return new_signal
    else:
      print("Sparse resample only handle signals of 1 or 2 dimensions, with time as the last dimension.")
      

def moving_average_3d(data, window_size=5):
    """
    Computes the moving average over the first dimension of a 3D array.
    
    Parameters:
        data (np.ndarray): A 3D numpy array where the first dimension is time.
        window_size (int): The number of steps to include in the average. Default is 5.
    
    Returns:
        np.ndarray: A 3D numpy array containing the moving averages.
    """
    # Validate the window size
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    
    # Check if window size is greater than the time dimension
    if window_size > data.shape[0]:
        raise ValueError("Window size cannot be greater than the number of time steps")
    
    # Initialize an array to store the moving averages
    result_shape = (data.shape[0] - window_size + 1, data.shape[1], data.shape[2])
    moving_averages = np.empty(result_shape)
    
    # Compute the moving average using efficient windowed sum
    for i in range(result_shape[0]):
        moving_averages[i] = np.mean(data[i:i+window_size], axis=0)
    
    return moving_averages

def lag_finder(y1, y2, sr):
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
    delay = delay_arr[np.argmax(corr)]

    return delay, corr



def onmf(X, rank, alpha=1.0, max_iter=100, H_init=None, W_init=None):
        """
        Orthogonal non-negative matrix factorization. Originally from https://github.com/mstrazar

        Parameters
        ----------
        X: array [m x n]
            Data matrix.
        rank: int
            Maximum rank of the factor model.
        alpha: int
            Orthogonality regularization parameter.
        max_iter: int
            Maximum number of iterations.
        H_init: array [rank x n]
            Fixed initial basis matrix.
        W_init: array [m x rank]
            Fixed initial coefficient matrix.

        Returns
        W: array [m x rank]
            Coefficient matrix (row clustering).
        H: array [rank x n]
            Basis matrix (column clustering / patterns).
        """

        m, n = X.shape
        W = rand(m, rank) if isinstance(W_init, type(None)) else W_init
        H = rand(rank, n) if isinstance(H_init, type(None)) else H_init

        for itr in range(max_iter):
            if isinstance(W_init, type(None)):
                enum = X.dot(H.T)
                denom = W.dot(H.dot(H.T))
                W = nan_to_num(W * enum/denom)

            if isinstance(H_init, type(None)):
                HHTH = H.dot(H.T).dot(H)
                enum = W.T.dot(X) + alpha * H
                denom = W.T.dot(W).dot(H) + 2.0 * alpha * HHTH
                H = nan_to_num(H * enum / denom)


        return W, H

def surprisal_variance(p):
    surprisals = -np.log2(p)
    expected_surprisal = np.sum(p * surprisals)
    expected_surprisal_squared = np.sum(p * surprisals**2)
    variance = expected_surprisal_squared - expected_surprisal**2
    return variance

def sparse_realign(x,y):
    '''
    Re-align x and y. Only works if x and y are sparse signals and have perfect alignment. Otherwise, use smith-waterman to find optimal alignment. 
    The signal will align according to x
    '''
    x1,y1 = x.copy(), y.copy()
    x1[x1 > 0] = 1
    y1[y1 > 0] = 1
    a,_ = signal.find_peaks(x)
    b,_ = signal.find_peaks(y)
    z = np.zeros(y.shape[0])
    for onset_index in range(len(b)):
        z[a[onset_index]] = y[b[onset_index]]
    return x, z
