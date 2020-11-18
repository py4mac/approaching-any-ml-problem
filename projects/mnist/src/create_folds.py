from sklearn import datasets
import pandas as pd
from sklearn import model_selection

from projects.mnist.src.config import TRAINING_FILE

print("Import minst")
data = datasets.load_digits(
    return_X_y=True,
    as_frame=True
)

print("Create dataframe")
df = data[0]
df['label'] = data[1]

print("Create train/test")
df = df.sample(frac=1).reset_index(drop=True)
df_train = df.head(int(df.shape[0] * 0.75))
df_test = df.tail(df.shape[0] - int(df.shape[0] * 0.75))

print("Save train/test")
df_train.to_csv('./projects/mnist/inputs/mnist_train.csv', index=False)
df_test.to_csv('./projects/mnist/inputs/mnist_test.csv', index=False)

print("Create train kfold")
df_train = df_train.copy(deep=True)
df_train['kfold'] = -1
y = df_train.label.values

kf = model_selection.StratifiedKFold(n_splits=5)

for f, (t_, v_) in enumerate(kf.split(X=df_train, y=y)):
    df_train.loc[v_, 'kfold'] = f

print("Save train kfold")
df_train.to_csv(TRAINING_FILE, index=False)