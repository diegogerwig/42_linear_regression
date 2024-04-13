import pandas as pd
import numpy as np
import os
import sys
from time import sleep

DATA_FILE_PATH = "./data/data.csv"
PARAMS_FILE_PATH = "./data/params.csv"
DEFAULT_THETAS = {'theta0': 0, 'theta1': 0}


def load_thetas():
    if os.path.isfile(PARAMS_FILE_PATH):
        return pd.read_csv(PARAMS_FILE_PATH)
    else:
        pd.DataFrame(DEFAULT_THETAS, index=[0]).\
            to_csv(PARAMS_FILE_PATH, index=False)
        return pd.DataFrame(DEFAULT_THETAS, index=[0])


def Mean_Squared_Error(theta0, theta1, mileage, price):
    """
    Calculate the Mean Squared Error (MSE) between predicted price and
      actual price.
    """
    mse = ((1 / len(mileage)) *
           sum(((theta0 + theta1 * mileage) - price) ** 2))
    return mse


def Mean_Absolute_Error(theta0, theta1, mileage, price):
    """
    Calculate the Mean Absolute Error (MAE) between predicted price and
      actual price.
    """
    mae = (1 / len(mileage)) * sum(abs(theta1 * mileage + theta0 - price))
    return mae


def Mean_Percentage_Error(theta0, theta1, mileage, price):
    """
    Calculate the Mean Percentage Error (MPE) between predicted price and
      actual price.
    """
    mpe = (1 / len(mileage)) * \
        sum(abs(theta1 * mileage + theta0 - price) / price) * 100
    return mpe


def calculate_r_squared(data_true, data_pred):
    """
    Calculate the coefficient of determination (R²) for a regression model.

    Is a statistical measure that provides an indication of how well the
      predicted values by a regression model fit the observed real values.

    R² = 1: The model explains all variability in the data and fits
      the observed data perfectly.
    R² = 0: The model explains no variability in the data and does not fit
      the observed data.
    """
    # Calculate the mean of the true target variable
    data_mean = np.mean(data_true)

    # Calculate the total sum of squares (SS_total)
    ss_total = np.sum((data_true - data_mean) ** 2)

    # Calculate the residual sum of squares (SS_res)
    ss_res = np.sum((data_true - data_pred) ** 2)

    # Calculate R^2
    r_squared = 1 - (ss_res / ss_total)

    return r_squared


def precision_stats(df: pd.DataFrame, theta0: float, theta1: float):
    price = df.loc[:, 'price']
    mileage = df.loc[:, 'km']
    mse = Mean_Squared_Error(theta0, theta1, mileage, price)
    rmse = mse ** (1/2)
    r2 = calculate_r_squared(price, theta0 + theta1 * mileage)
    mae = Mean_Absolute_Error(theta0, theta1, mileage, price)
    mpe = Mean_Percentage_Error(theta0, theta1, mileage, price)

    print("\n=============  PRECISION STADISTICS  ====================")
    print(" MEAN SQUARED ERROR (MSE):                  {:<.5f}".format(mse))
    print(" ROOT MEAN SQUARED ERROR (RMSE):            {:<.5f}".format(rmse))
    print(" R-SQUARED (coefficient of determination):  {:<.5f}".format(r2))
    print(" MEAN ABSOLUTE ERROR (MAE):                 {:<.5f}".format(mae))
    print(" MEAN PERCENTAGE ERROR (MPE):               {:<.5f} %".format(mpe))
    print("=========================================================\n")
    sleep(5)


def main():
    try:
        assert len(sys.argv) == 1, "❗️ Wrong number of arguments"

        df = pd.read_csv(DATA_FILE_PATH)
        assert df is not None, "❌ The dataset is wrong"

        theta = load_thetas()
        theta0, theta1 = theta.loc[0, ['theta0', 'theta1']]

        assert (theta1 is not None and theta0 is not None), "thetas not num"
        precision_stats(df, theta0, theta1)
        return

    except AssertionError as msg:
        print("❌ Error:", msg)

    except Exception as error:
        print("❌ Error: ", error)


if __name__ == "__main__":
    main()
