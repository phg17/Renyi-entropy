import numpy as np
from scipy.ndimage import label, find_objects

def cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    larger = sum([1 for x_i in x for y_i in y if x_i > y_i])
    smaller = sum([1 for x_i in x for y_i in y if x_i < y_i])
    return (larger - smaller) / (n_x * n_y)

def cohen_d(x, y):
    x,y = np.asarray(x), np.asarray(y)
    # Calculate differences
    diffs = x - y
    # Mean of differences
    mean_diff = np.mean(diffs)
    # Standard deviation of differences
    std_dist = np.std(np.concatenate([x,y]), ddof=1)  # ddof=1 for sample standard deviation
    # Calculate Cohen's d
    d = mean_diff / std_dist
    return d

def max_cluster(data, thres = 0.001):
    """
    Find the biggest cluster along the first axis

    Parameters
    ----------
    data : ndarray
        data of shape (n_roi, n_times)
    thres : float
        threshold

    Returns
    ----------
    max_cluster : int
        Largest cluster found
    """
    max_cluster = 0
    for roi_i, roi_data in enumerate(data):
        clusters = (roi_data > thres).astype(int)
        segments = ''.join(map(str, clusters)).split('0')
        max_cluster = max(max_cluster,max(len(s) for s in segments))
    return max_cluster

def select_clusters(data, data_p, thres = 0.001, pval = 0.05):
    """
    Select channels containing channels with relevant clusters

    Parameters
    ----------
    data : ndarray
        Non permuted data, of shape (n_roi, n_times)
    data_p : ndarray
        Permuted data, of shape (n_roi, n_perm, n_times)
    thres : float
        Threshold of clusters
    pval : float
        alpha risk error

    Returns
    ----------
    stat_mask : ndarray
        Mask (True if relevant, False if rejected), of shape (n_roi,)
    perm_dist : ndarray
        Distribution of permuted cluster size, of shape (n_perm,)
    no_perm_dist : ndarray
        Distribution of actual data cluster size, of shape (n_roi)
    """
    data_perm = np.asarray(data_p)
    data_no_perm = np.asarray(data)
    perm_dist = []
    no_perm_dist = []
    for i_perm in range(data_perm.shape[1]):
        perm = data_perm[:,i_perm,:]
        perm_size = max_cluster(perm, thres = thres)
        perm_dist.append(perm_size)

    for i_chan in range(data_perm.shape[0]):
        no_perm = data_no_perm[i_chan:i_chan+1,:]
        no_perm_size = max_cluster(no_perm, thres = thres)
        no_perm_dist.append(no_perm_size)
    no_perm_dist = np.asarray(no_perm_dist)
    perm_dist = np.asarray(perm_dist)
    stat_mask = no_perm_dist > np.percentile(no_perm_dist, 100 - pval*100)
    return stat_mask, perm_dist, no_perm_dist



def filter_binary(binary_array, min_cluster_size=4):
    """
    Finds clusters of True values in a binary array and returns a mask array
    where only clusters of size >= min_cluster_size are True. Useful to 
    filter out isolated points.
    
    Parameters
    ----------
    binary_array : ndarray
        1D array of booleans to filter
    min_cluster_size : int
        How many successive True values are necessary to define a cluster

    Returns
    -------
    cluster_mask : ndarray
        masking array corresponding to the filtering.
    """
    

    # Label all connected components
    labeled_array, num_features = label(binary_array)
    slices = find_objects(labeled_array)

    # Filter clusters by size
    cluster_mask = np.zeros_like(binary_array, dtype=bool)
    for s in slices:
        cluster_size = s[0].stop - s[0].start
        if cluster_size >= min_cluster_size:
            cluster_mask[s] = True

    return cluster_mask