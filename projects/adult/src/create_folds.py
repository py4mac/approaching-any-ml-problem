from sklearn import datasets
import pandas as pd
from sklearn import model_selection

from projects.adult.src.config import TRAINING_FILE

print("Import adult dataset")
# See http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data rename to adult.csv

df = pd.read_csv("./projects/adult/inputs/adult.csv")

df['kfold'] = 1
df = df.sample(frac=1).reset_index(drop=True)
y = df.income.values

kf = model_selection.StratifiedKFold(n_splits=5)

for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

print("Save train kfold")
df.to_csv(TRAINING_FILE, index=False)