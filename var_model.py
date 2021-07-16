import pandas as pd
import matplotlib.pyplot as plt
import datetime

from statsmodels.tsa.api import VAR


def correct_seasonality(df, period=365):
    adj_df = df.truncate(before=df.index[0] + datetime.timedelta(days=period))
    for index, row in adj_df.iterrows():
        for s_index, value in row.items():
            try:
                prev_value = df.at[index - datetime.timedelta(days=period), s_index]
            except:
                prev_value = value

            adj_df.at[index, s_index] = value - prev_value

    return adj_df


def lag_selector(df, max_lag=10):
    model = VAR(df)
    aics = []
    for i in range(1, max_lag+1):
        result = model.fit(i)
        print(result.aic)
        aics.append(result.aic)

    return aics.index(min(aics)) + 1


if __name__ == "__main__":
    path_name = input("Path name: ")
    df = pd.read_csv(path_name)
    df.index = pd.to_datetime(df["datetime"])
    df = df.drop(columns="datetime")

    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    df = correct_seasonality(df)

    df.index = range(len(df))
    lag = lag_selector(df)

    model = VAR(df)
    results = model.fit(lag)
    results.summary()