from sklearn import datasets
import pandas as pd
from sklearn import model_selection

from projects.imdb.src.config import INPUT_FILE,TRAINING_FILE

df = pd.read_csv(f"{INPUT_FILE}")

df["kfold"] = -1

df = df.sample(frac=1).reset_index(drop=True)

y = df.sentiment.values

kf = model_selection.StratifiedKFold(n_splits=5)

for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

df.to_csv(f"{TRAINING_FILE}", index=False)