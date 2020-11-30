import copy
import itertools
import joblib
import pandas as pd
import xgboost as xgb
from sklearn import metrics, linear_model, preprocessing
import argparse

from projects.adult.src.config import TRAINING_FILE,MODEL_OUTPUT

def mean_target_encoding(data):
    df = copy.deepcopy(data)
    
    num_cols = [
        'age',
        'fnlwgt',
        'education-num',
        'capital-gain',
        'capital-loss',
        'hours-per-week'
    ]

    target_mapping = {
        " <=50K": 0,
        " >50K": 1
    }

    df.loc[:, "income"] = df.income.map(target_mapping)

    features = [
        f for f in df.columns if f not in ("income", "kfold") and f not in num_cols
    ]

    for col in features:
        if  col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna('NONE')
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])
    
    encoded_dfs = []

    # The idea here is to encode with income mean value of the other fold to prevent from overfitting
    for fold in range(5):        
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        for col in features:
            mapping_dict = dict(
                df_train.groupby(col)['income'].mean()
            )
            df_valid.loc[
                :, col + "_enc"
            ] = df_valid[col].map(mapping_dict)
        encoded_dfs.append(df_valid)
    encoded_df = pd.concat(encoded_dfs, axis=0)

    return encoded_df


def run(df, fold):

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    features = [
        f for f in df.columns if f not in ("income", "kfold")
    ]

    print(features)
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
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

    df = pd.read_csv(f"{TRAINING_FILE}")

    print(df.shape[0])
    df = mean_target_encoding(df)

    print(df.shape[0])

    run(df=df, fold=args.fold)
