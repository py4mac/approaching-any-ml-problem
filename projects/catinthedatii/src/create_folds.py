from sklearn import datasets
import pandas as pd
from sklearn import model_selection

from projects.catinthedatii.src.config import TRAINING_FILE

print("Import cat-in-the-dat-ii dataset")
# See https://www.kaggle.com/c/cat-in-the-dat-ii/data

df = pd.read_csv("./projects/catinthedatii/inputs/cat_train.csv")

df['kfold'] = 1
df = df.sample(frac=1).reset_index(drop=True)
y = df.target.values

kf = model_selection.StratifiedKFold(n_splits=5)

for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

print("Save train kfold")
df.to_csv(TRAINING_FILE, index=False)