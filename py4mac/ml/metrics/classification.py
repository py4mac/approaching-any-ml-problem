import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Used for binary and multi class prediction

def true_positive(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp

def false_positive(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp

def true_negative(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn

def false_negative(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn

def accuracy(y_true, y_pred):
    # correct_counter / len(y_true)
    # or
    # (tp + tn) / (tp + tn + fp + fn)
    return metrics.accuracy_score(y_true, y_pred)

def precision(y_true, y_pred):
    # (tp) / (tp + fp)
    return metrics.precision_score(y_true, y_pred)

def recall(y_true, y_pred):
    # (tp) / (tp + fn)
    return metrics.recall_score(y_true, y_pred)

def tpr(y_true, y_pred):
    return recall(y_true, y_pred)

def fpr(y_true, y_pred):
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return fp /  (fp + tn)

def f1_score(y_true, y_pred):
    # 2*t*p / (2*t*p + f*p + f*n)
    # or
    # 2*precision*recall / (precision + recall)
    return metrics.f1_score(y_true, y_false)

def roc_auc_score(y_true, y_pred_proba):
    """ Usefull for skewed binary data. """
    return metrics.roc_auc_score(y_pred, y_pred_proba)

def kappa_score_quadratic(y_true, y_pred):
    return metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")

def mcc_score(y_true, y_pred):
    return metrics.matthews_corrcoef(y_true, y_pred)

def plot_roc_curve(y_true, y_pred_proba, thresholds):
    """ Usefull for skewed binary data. """
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        temp_pred = [1 if x >= thresh else 0 for x in y_pred_proba]
        temp_pr = tpr(y_true, temp_pred)
        temp_fpr = fpr(y_true, temp_pred)
        tpr_list.append(temp_pr)
        fpr_list.append(temp_fpr)

    plt.figure(figsize=(7,7))
    plt.fill_between(fpr_list, tpr_list, alpha=0.4)
    plt.plot(fpr_list, tpr_list, lw=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    plt.show()

    df = pd.DataFrame()
    df['threshold'] = thresholds
    df['tpr'] = tpr_list
    df['fpr'] = fpr_list
    return df

def select_best_threshold(y_true, y_pred_proba, thresholds):
    """ Usefull for skewed binary data. """
    tp_list = []
    fp_list = []

    for thresh in thresholds:
        temp_pred = [1 if x >= thresh else 0 for x in y_pred_proba]
        temp_p = true_positive(y_true, temp_pred)
        temp_fp = false_positive(y_true, temp_pred)
        tp_list.append(temp_p)
        fp_list.append(temp_fp)
    
    df = pd.DataFrame()
    df['threshold'] = thresholds
    df['tp'] = tp_list
    df['fp'] = fp_list
    return df

def plot_confusion_matrix(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    cmap = sns.cubehelix_palette(
        50,
        hue=0.05,
        rot=0,
        light=0.9,
        dark=0,
        as_cmap=True
    )
    sns.set(font_scale=2.5)
    sns.heatmap(cm, annot=True, cmap=cmap, cbar=False)
    plt.ylabel("Actual Labels", fontsize=20)
    plt.xlabel("Predicted Labels", fontsize=20)

