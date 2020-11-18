
def train_test_split(df, frac=1, train_size=0.25):
    df = df.sample(frac=frac).reset_index(drop=True)

    df_len = df.shape[0]
    train_len = round(df_len * train_size)

    df_train = df.head(train_len)
    df_test = df.tail(df_len - train_len)

    return df_train, df_test

