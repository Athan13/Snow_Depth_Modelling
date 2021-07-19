import pandas as pd
import matplotlib.pyplot as plt
import datetime

from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error


def correct_seasonality(df, period=365):
    adj_df = df.truncate(before=df.index[0] + datetime.timedelta(days=period))
    for index, row in adj_df.iterrows():
        for column_name, value in row.items():
            try:
                prev_value = df.at[index - datetime.timedelta(days=period), column_name]
            except KeyError:
                prev_value = value

            adj_df.at[index, column_name] = value - prev_value

    return adj_df


def daterange(start_date, iterations):
    dates = [start_date]
    for i in range(1, iterations):
        dates.append(dates[i-1] + datetime.timedelta(hours=1))
    return dates


def sum_seasonal_data(train_data, forecast, period=365):
    min_test_date = datetime.datetime(2021, 1, 1, 0, 0, 0)
    forecast = pd.DataFrame(forecast, columns=train_data.columns, index=daterange(min_test_date, len(forecast)))
    print(forecast)
    for index, row in forecast.iterrows():
        for column_name, value in row.items():
            try:
                prev_value = train_data.at[index - datetime.timedelta(days=period), column_name]
            except KeyError:
                prev_value = train_data.at[index - datetime.timedelta(days=period+1), column_name]
            forecast.at[index, column_name] += prev_value
    return forecast


if __name__ == "__main__":
    path_name = "/Users/athan/Documents/Wegaw/Aigen Lake Test Data/combined snow data.csv"
    # path_name = input("Path name: ")
    df = pd.read_csv(path_name)
    df.index = pd.to_datetime(df["datetime"])
    df = df.drop(columns="datetime")

    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    train_data = df[df.index < datetime.datetime(2021, 1, 1, 0, 0, 0)]
    test_data = df[df.index >= datetime.datetime(2021, 1, 1, 0, 0, 0)]

    adj_train_data = correct_seasonality(train_data)

    model = VAR(adj_train_data)
    results = model.fit(maxlags=15, ic="aic")
    output = results.forecast(adj_train_data.to_numpy(), len(test_data))
    forecast = sum_seasonal_data(train_data, output, period=365)

    snow_forecast = forecast[forecast.columns[0]]
    snow_true = test_data[test_data.columns[0]]
    rmse = round(mean_squared_error(snow_forecast, snow_true, squared=False), 2)

    plt.plot(snow_forecast, color="blue")
    plt.plot(snow_true, color="green")
    plt.legend(["forecast", "true"])
    plt.ylabel(forecast.columns[0])
    plt.text(datetime.date(2021, 6, 1), 95, s=f"RMSE = {rmse}")
    plt.show()