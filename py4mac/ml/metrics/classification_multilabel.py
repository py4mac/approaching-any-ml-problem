from sklearn import metrics

def pk(y_true, y_pred, k):
    if k == 0: 
        return 0

    y_pred = y_pred[:k]
    pred_set = set(y_pred)
    true_set = set(y_true)
    common_values = pred_set.intersection(true_set)
    return len(common_values) / len(y_pred)

def apk(y_true, y_pred, k):
    pk_values = []
    for i in range(1, k+1):
        pk_values.append(pk(y_true, y_pred, i))
    
    if len(pk_values) == 0:
        return 0
    
    return sum(pk_values) / len(pk_values)

def mapk(y_true, y_pred, k):
    apk_values = []
    for i in range(len(y_true)):
        apk_values.append(apk(y_true[i], y_pred[i], k=k))

    return sum(apk_values) / len(apk_values)