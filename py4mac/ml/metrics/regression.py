from sklearn import metrics
import numpy as np


def mean_absolute_error(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)

def mean_squared_error(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)

def mean_squared_log_error(y_true, y_pred):
    return metrics.mean_squared_log_error(y_true, y_pred)

def mean_percentage_error(y_true, y_pred):
    error = 0

    for y_t, y_pr in zip(y_true, y_pred):
        error += (y_t - y_p) / y_t
    
    return error / len(y_true)

def mean_abs_percentage_error(y_true, y_pred):
    error = 0

    for y_t, y_pr in zip(y_true, y_pred):
        error += np.abs(y_t - y_p) / y_t
    
    return error / len(y_true)

def r2(y_true, y_pred):
    return metrics.r2_score(y_true, y_pred)

def mae_np(y_true, y_pred):
    """ y_true and y_pred must be passed as numpy array
    You can convert standard array to numpy array using
    y_true = [1,2,3]
    y_true = np.array(y_true)
    """
    return np.mean(np.abs(y_true-y_pred))