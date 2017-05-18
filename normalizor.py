import numpy as np


class Normalizer:
    def __init__(self, remove, denominator, minus):
        self.remove = remove
        self.denominator = denominator
        self.minus = minus

    @staticmethod
    def standard_deviation_normalization(data_value, normalizer=None):
        """ Data normalization using standard deviation
        Args:
            data_value: The data to be normalized
            normalizer: guide
        """
        data_rows, data_cols = data_value.shape
        if normalizer is None:
            data_col_means = data_value.mean(axis=0)
            data_col_standard_deviation = data_value.std(axis=0)
            remove = []
            for col in range(0, data_cols, 1):
                if data_col_standard_deviation[col] == 0:
                    remove.append(col)
                    print("standard = 0  :", col)
                else:
                    std = data_col_standard_deviation[col]
                    col_mean = data_col_means[col]
                    for row in range(0, data_rows, 1):
                        data_value[row][col] = (data_value[row][col] - col_mean) / std
            data_col_means = np.delete(data_col_means, remove, axis=0)
            data_col_standard_deviation = np.delete(data_col_standard_deviation, remove, axis=0)
            return np.delete(data_value, remove, axis=1), Normalizer(remove, data_col_standard_deviation, data_col_means)
        else:
            # print(data_value.shape)
            data_value = np.delete(data_value, normalizer.remove, axis=1)
            # print(data_value.shape)
            data_rows, data_cols = data_value.shape
            for col in range(0, data_cols, 1):
                col_mean = normalizer.minus[col]
                std = normalizer.denominator[col]
                for row in range(0, data_rows, 1):
                    print(row, col)
                    data_value[row][col] = (data_value[row][col] - col_mean) / std
            return data_value

    @staticmethod
    def max_min_normalization(data_value, data_col_max_values=None, data_col_min_values=None):
        """ Data normalization using max value and min value
        Args:
            data_value: The data to be normalized
            data_col_max_values: The maximum value of data's columns
            data_col_min_values: The minimum value of data's columns
        """
        data_rows, data_cols = data_value.shape
        if data_col_max_values is None:
            data_col_max_values = data_value.max(axis=0)
        if data_col_min_values is None:
            data_col_min_values = data_value.min(axis=0)
        for col in range(0, data_cols, 1):
            if data_col_max_values[col] != data_col_min_values[col]:
                value_range = data_col_max_values[col] - data_col_min_values[col]
                for row in range(0, data_rows, 1):
                    data_value[row][col] = (data_value[row][col] - data_col_min_values[col]) / value_range