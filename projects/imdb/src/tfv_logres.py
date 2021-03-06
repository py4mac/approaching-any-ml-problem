import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from projects.imdb.src.config import INPUT_FILE,MODEL_OUTPUT

if __name__ == "__main__":
    df = pd.read_csv(f"{INPUT_FILE}")

    df["kfold"] = 1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.sentiment.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    for fold_ in range(5):
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)

        tfv = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
        tfv.fit(train_df.review)

        xtrain = tfv.transform(train_df.review)
        xtest = tfv.transform(test_df.review)

        model = linear_model.LogisticRegression()

        model.fit(xtrain, train_df.sentiment)

        preds = model.predict(xtest)

        accuracy = metrics.accuracy_score(test_df.sentiment, preds)

        print(f"Fold {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")