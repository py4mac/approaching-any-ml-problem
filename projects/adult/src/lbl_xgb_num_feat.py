import itertools
import joblib
import pandas as pd
import xgboost as xgb
from sklearn import metrics, linear_model, preprocessing
import argparse

from projects.adult.src.config import TRAINING_FILE,MODEL_OUTPUT

def feature_engineering(df, cat_cols):
    combi = list(itertools.combinations(cat_cols, 2))

    for c1, c2 in combi:
        df.loc[
            :,
            c1 + "_" + c2
        ] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df

def run(fold):

    df = pd.read_csv(f"{TRAINING_FILE}")

    num_cols = [
        'age',
        'fnlwgt',
        'education-num',
        'capital-gain',
        'capital-loss',
        'hours-per-week'
    ]

    # df = df.drop(num_cols, axis=1)

    target_mapping = {
        " <=50K": 0,
        " >50K": 1
    }

    df.loc[:, "income"] = df.income.map(target_mapping)

    cat_cols = [
        f for f in df.columns if f not in ("income", "kfold") and f not in num_cols
    ]

    print(cat_cols)
    df = feature_engineering(df, cat_cols)

    features = [
        f for f in df.columns if f not in ("income", "kfold")
    ]

    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna('NONE')
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(
        n_jobs=-1,
        # max_depth=7,
        # n_estimators=200
    )

    model.fit(x_train, df_train.income.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(auc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )

    args = parser.parse_args()
    run(fold=args.fold)
