import numpy as np


class Normalizer:
    def __init__(self, remove, minus, denominator):
        self.remove = remove
        self.denominator = denominator
        self.minus = minus

    @staticmethod
    def standard_deviation_normalization(data, normalizer=None):
        """ Data normalization using standard deviation
        Args:
            data: The data to be normalized
            normalizer: guide
        """
        data_rows, data_cols = data.shape
        if normalizer is None:
            data_col_means = data.mean(axis=0)
            data_col_standard_deviation = data.std(axis=0)
            remove = []
            for col in range(data_cols):
                if data_col_standard_deviation[col] == 0:
                    remove.append(col)
                    # print("standard = 0  :", col)
                else:
                    std = data_col_standard_deviation[col]
                    col_mean = data_col_means[col]
                    for row in range(data_rows):
                        data[row][col] = (data[row][col] - col_mean) / std
            data_col_means = np.delete(data_col_means, remove, axis=0)
            data_col_standard_deviation = np.delete(data_col_standard_deviation, remove, axis=0)
            return np.delete(data, remove, axis=1), Normalizer(remove, data_col_means, data_col_standard_deviation)
        else:
            # print(data_value.shape)
            data = np.delete(data, normalizer.remove, axis=1)
            # print(data_value.shape)
            data_rows, data_cols = data.shape
            for col in range(0, data_cols, 1):
                col_mean = normalizer.minus[col]
                std = normalizer.denominator[col]
                for row in range(0, data_rows, 1):
                    # print(row, col)
                    data[row][col] = (data[row][col] - col_mean) / std
            return data

    @staticmethod
    def max_min_normalization(data, normalizer=None):
        """ Data normalization using max value and min value
        Args:
            data: The data to be normalized
            normalizer:guide
        """
        # data_rows, data_cols = data_value.shape
        # if data_col_max_values is None:
        #     data_col_max_values = data_value.max(axis=0)
        # if data_col_min_values is None:
        #     data_col_min_values = data_value.min(axis=0)
        # for col in range(0, data_cols, 1):
        #     if data_col_max_values[col] != data_col_min_values[col]:
        #         value_range = data_col_max_values[col] - data_col_min_values[col]
        #         for row in range(0, data_rows, 1):
        #             data_value[row][col] = (data_value[row][col] - data_col_min_values[col]) / value_range
        data_rows, data_cols = data.shape
        if normalizer is None:
            data_col_max = data.max(axis=0)
            data_col_min = data.min(axis=0)
            remove = []
            data_col_range = []
            for col in range(data_cols):
                if data_col_max[col] == data_col_min[col]:
                    remove.append(col)
                    # print("standard = 0  :", col)
                else:
                    col_range = (data_col_max[col] - data_col_min[col])/100.0
                    data_col_range.append(col_range)
                    col_min = data_col_min[col]
                    for row in range(data_rows):
                        data[row][col] = (data[row][col] - col_min) / col_range
            data_col_min = np.delete(data_col_min, remove, axis=0)
            data_col_range = np.delete(data_col_range, remove, axis=0)
            return np.delete(data, remove, axis=1), Normalizer(remove, data_col_min, data_col_range)
        else:
            # print(data_value.shape)
            data = np.delete(data, normalizer.remove, axis=1)
            # print(data_value.shape)
            data_rows, data_cols = data.shape
            for col in range(data_cols):
                col_min = normalizer.minus[col]
                col_range = normalizer.denominator[col]
                for row in range(data_rows):
                    print("data[row][col]=", data[row][col], "    col_min=", col_min, "   col_range", col_range)
                    data[row][col] = (data[row][col] - col_min) / col_range
            return data
