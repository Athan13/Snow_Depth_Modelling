import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from var_model import sum_seasonal_data, correct_seasonality
from sklearn.metrics import mean_squared_error


def difference_values(values, max_d=4, time_diff=1):
    diff_values = list(values)
    diff_differences = []
    top_d = 0
    top_d_found = False

    for d in range(max_d + 1):
        diff_differences.append(diff_values)
        p_value = adfuller(diff_values)[1]
        if p_value <= 0.05 and not top_d_found:
            top_d = d
            top_d_found = True

        diff_list = [x1 - x2 for (x1, x2) in zip(diff_values[time_diff:], diff_values[:-time_diff])]
        diff_values = np.concatenate([[diff_values[0]], diff_list])

    return top_d, diff_differences


if __name__ == "__main__":
    path_name = input("Path name: ")
    df = pd.read_csv(path_name)
    data_type = df.columns[1]
    df.index = pd.to_datetime(df["datetime"])
    df = df[[data_type]]

    # Create non-seasonal train data, test data
    train_data = df[df.index < datetime.datetime(2021, 1, 1, 0, 0, 0)]
    test_data = df[df.index >= datetime.datetime(2021, 1, 1, 0, 0, 0)]

    # Difference data appropriately
    adj_train_data = correct_seasonality(train_data, 365)
    d = difference_values(adj_train_data[data_type], 2)[0]

    # ARIMA model
    model = ARIMA(list(adj_train_data[data_type]), order=(1, d, 2)).fit()
    output = model.forecast(len(test_data))
    forecast = sum_seasonal_data(train_data, output, period=365)

    # plot forecasts against actual outcomes
    plt.plot(test_data, color='green')
    plt.plot(forecast, color='blue')
    plt.legend(["true", "forecast (ARIMA)"])
    plt.ylabel(data_type)
    rmse = round(mean_squared_error(forecast, test_data, squared=False), 2)
    plt.text(datetime.date(2021, 6, 1), 95, s=f"RMSE = {rmse}")
    plt.show()

