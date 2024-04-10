import numpy as np
import pandas as pd


def create_linear_model(x: np.ndarray, y: np.ndarray):
    """
    USAGE: x and y both are 1d numpy array.
           Thus, x is single variable.
    """
    x_mul_y = np.dot(x, y)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    total_obs = len(x)
    x_mul_x = np.sum(x**2)

    m = (total_obs * x_mul_y - x_sum * y_sum) / (total_obs * x_mul_x - x_sum**2)
    c = (y_sum - m * x_sum) / total_obs

    return m, c


def general_linear_model(x: np.ndarray, y: np.ndarray):
    """
    USAGE: x should be numpy array with feature being column and
           each observation being rows.
           y is target variable which should be of shape (n, 1)
           where n is number of observations.

    How did i get this: Refer README.md for more info
    """
    assert y.shape[1] == 1

    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    M = np.dot(x.T, x)
    b = np.dot(x.T, y)

    assert np.linalg.det(M) != 0

    value = np.dot(np.linalg.inv(M), b)

    # or using pseudo inverse as (A^T.A)^(-1) . A^T is pseudo inverse of A
    # value = np.dot(np.linalg.pinv(x), y)

    return value


if __name__ == "__main__":
    x, y = "NOX", "DIS"
    data = pd.read_csv("./HousingData.csv")
    data_x = np.array(data[x])
    data_y = np.array(data[y])

    # m, c = create_linear_model(data_x, data_y)
    # print(m, c)

    data_x = data_x.reshape((-1, 1))
    data_y = data_y.reshape((-1, 1))
    res = general_linear_model(data_x, data_y)
    print(res)
