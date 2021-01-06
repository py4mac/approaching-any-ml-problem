import numpy as np
import pandas as pd

from functools import partial

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

from projects.mobile_price.src.config import TRAINING_FILE,MODEL_OUTPUT


def optimize(params, x, y):

    model = ensemble.RandomForestClassifier(**params)

    kf = model_selection.StratifiedKFold(n_splits=5)

    accuracies = []

    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)

        fold_accuracy = metrics.accuracy_score(
            ytest,
            preds
        )
        accuracies.append(fold_accuracy)
    
    return -1*np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv(f"{TRAINING_FILE}")
    
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values
    
    param_space = {
        "max_depth": scope.int(
            hp.quniform("max_depth", 1, 15, 1)
        ),
        "n_estimators": scope.int(
            hp.quniform("n_estimators", 100, 1500, 1)
        ),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0.1, 1),
    }

    optimize_function = partial(
        optimize,
        x=X,
        y=y
    )

    trials = Trials()

    hopt = fmin(
        fn=optimize_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials
    )

    print(hopt)

