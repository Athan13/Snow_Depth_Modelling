import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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
    path_name = input("Path name: ")  # /Users/athan/Documents/Wegaw/Aigen Lake Test Data/combined snow data.csv
    df = pd.read_csv(path_name)
    data_type = df.columns[1]
    df = df[["datetime", data_type]]
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(df[data_type])

    # Get optimal differencing parameter and differenced values
    d, diffed_values = difference_values(df[data_type], 2)

    # Create train data, test data
    train_data = list(df[df["datetime"] < datetime.datetime(2021, 1, 1, 0, 0, 0)][data_type])
    test_data = list(df[df["datetime"] >= datetime.datetime(2021, 1, 1, 0, 0, 0)][data_type])

    # Plot ACF and PACF to get p and q respectively
    # fig, ax = plt.subplots(2, len(diffed_values))
    # for i in range(len(diffed_values)):
    #     plot_acf(x=np.asarray(diffed_values[i]), lags=np.arange(8760), ax=ax[0, i], title=f"p = {i + 1}")
    #     print(f"ACF[{i}] is done")
    #     plot_pacf(x=np.asarray(diffed_values[i]), lags=np.arange(8760), ax=ax[1, i], title=f"q = {i + 1}")
    #     print(f"PACF[{i}] is done \n")
    #
    # plt.show()

    # (s)ARIMA model
    # model = ARIMA(train_data, order=(2, 1, 2))
    model = SARIMAX(train_data, order=(2, 1, 2), seasonal_order=(1, 0, 1, 8760))
    model_fit = model.fit()
    output = model_fit.forecast(len(test_data))

    # plot forecasts against actual outcomes
    plt.plot(test_data) 
    plt.plot(output, color='red')
    plt.show()

