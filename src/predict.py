import pandas as pd
import os
import sys

PARAMS_FILE_PATH = "./data/params.csv"
ERRORS_FILE_PATH = "./data/errors.csv"
DEFAULT_THETAS = {'theta0': 0, 'theta1': 0}


def is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def load_thetas():
    if os.path.isfile(PARAMS_FILE_PATH):
        return pd.read_csv(PARAMS_FILE_PATH)
    else:
        pd.DataFrame(DEFAULT_THETAS, index=[0]).\
            to_csv(PARAMS_FILE_PATH, index=False)
        return pd.DataFrame(DEFAULT_THETAS, index=[0])


def load_errors(path):
    try:
        errors_df = pd.read_csv(path)
        return errors_df
    except FileNotFoundError:
        return None


def check_mileage():
    while True:
        mileage_str = input("\n🟠 Enter the mileage of the car in km: ")
        if is_integer(mileage_str) and int(mileage_str) >= 0:
            return int(mileage_str)
        else:
            print("❗️ Enter a valid non-negative integer value for mileage.")


def main():
    try:
        assert len(sys.argv) == 1, "❗️ Wrong number of arguments"

        theta = load_thetas()
        theta0, theta1 = theta.loc[0, ['theta0', 'theta1']]

        mileage = check_mileage()

        estim_price = theta0 + (theta1 * mileage)
        print("\n🟢 Estimated price for mileage {} km -> {:.2f} €".
              format(mileage, estim_price))
        print("   theta0: {:.4f} | theta1: {:.4f}\n".format(theta0, theta1))

        errors_df = load_errors(ERRORS_FILE_PATH)
        if errors_df is not None:
            errors = errors_df.loc[:, 'errors']
            mean_abs_error = errors.abs().mean()
            print("   Mean Absolute Error: {:.2f} €\n".format(mean_abs_error))
            estim_prices_range = (estim_price - mean_abs_error,
                                  estim_price + mean_abs_error)
            print("   Estimated Price Range (± Error): {:.2f} € - {:.2f} €\n".
                  format(estim_prices_range[0], estim_prices_range[1]))

        return

    except FileNotFoundError:
        print("❗️ The file '{}' does not exist.".format(PARAMS_FILE_PATH))

    except ValueError:
        print("❗️ Thetas in '{}' must be numeric.".format(PARAMS_FILE_PATH))

    except Exception as error:
        print("❌ Error: ", error)


if __name__ == "__main__":
    main()
