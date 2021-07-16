import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics
import numpy as np


def get_num_data(int_series):
    int_series = list(int_series)
    for i in range(len(int_series)):
        try:
            if not math.isnan(float(int_series[i])):
                int_series[i] = float(int_series[i])
            else:
                raise ValueError
        except ValueError:
            int_series[i] = (float(int_series[i - 1]) + float(int_series[i - 2])) / 2

    return int_series


def fix_oom(int_series):
    int_series = list(int_series)
    new_int_series = [statistics.median(int_series)]

    prev_problem = False
    prev_multiplier = 1
    for i in range(len(int_series)):
        curr_value = int_series[i]
        prev_value = new_int_series[i]

        curr_value = abs(curr_value)
        if curr_value == 0:
            new_value = 0.0
            prev_problem = False
        elif 0.1 < prev_value/curr_value < 10:
            new_value = curr_value
            prev_problem = False
        else:
            if prev_problem:
                new_value = curr_value * prev_multiplier
            else:
                curr_oom = math.floor(math.log10(curr_value))

                if prev_value == 0.0:
                    true_oom = -1
                else:
                    true_oom = math.floor(math.log10(prev_value))

                prev_multiplier = 10 ** (-1*curr_oom) * 10 ** true_oom
                new_value = curr_value * prev_multiplier
            prev_problem = True

        new_int_series.append(new_value)

    return new_int_series[1:]


def final_outlier_cleanup(int_series, sensitivity):
    new_int_series = []
    int_series = list(int_series)
    mean = np.mean(int_series)
    stddev = np.std(int_series)

    outlier_count = 0
    for i in range(len(int_series)):
        curr_value = int_series[i]
        if mean + sensitivity * stddev > curr_value > mean - sensitivity * stddev:
            new_int_series.append(curr_value)
        else:
            new_int_series.append((new_int_series[i-1] + new_int_series[i-2]) / 2)
            outlier_count += 1

    print(f"No. of outliers: {outlier_count}")
    return new_int_series


def read_and_clean_data(path_name, data_type, outlier_sensitivity=4):
    data = pd.read_csv(path_name).drop(range(5))
    data.columns = ["datetime", data_type]

    # Check for NaNs
    data[data_type] = get_num_data(data[data_type])

    # Fix orders of magnitude
    data[data_type] = fix_oom(data[data_type])

    # Remove outliers
    data[data_type] = final_outlier_cleanup(data[data_type], outlier_sensitivity)

    # Datetime format
    data["datetime"] = pd.to_datetime(data["datetime"])

    return data

# /Users/athan/Documents/Wegaw/Aigen Lake Test Data/Snow-Depth-(cm).csv
# /Users/athan/Documents/Wegaw/Aigen Lake Test Data/Temperature (K).csv
# /Users/athan/Documents/Wegaw/Aigen Lake Test Data/sw (mm).csv


if __name__ == "__main__":
    # snow_path_name = str(input("Path name for snow data: "))
    snow_path_name = "/Users/athan/Documents/Wegaw/Aigen Lake Test Data/Snow-Depth-(cm).csv"
    snow_data = read_and_clean_data(snow_path_name, "snow depth (cm)")

    no_data_sets = int(input("How many more data sets? "))
    for i in range(no_data_sets):
        data_type = str(input("Data type (with units): "))
        path_name = str(input("Path name: "))

        data = read_and_clean_data(path_name, data_type)
        snow_data = snow_data.merge(data)

    print(snow_data)
    if input("Download data? [y/n]: ") == "y":
        snow_data.to_csv("/Users/athan/Documents/Wegaw/Aigen Lake Test Data/combined_data.csv", index=False)

    # Plot
    for i in range(1, len(snow_data.columns)):
        plt.plot(snow_data["datetime"], snow_data[snow_data.columns[i]])
    plt.xlabel("Time")
    plt.legend(snow_data.columns[1:])
    plt.show()
