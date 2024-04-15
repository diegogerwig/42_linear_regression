import pandas as pd
import sys
import time
from plotter import plot_price_vs_mileage, \
    plot_regression_animation, \
    plot_price_vs_mileage_with_linear_reg, \
    plot_cost_evolution, \
    plot_theta0_and_theta1, \
    plot_normal_distribution

DATA_FILE_PATH = "./data/data.csv"
PARAMS_FILE_PATH = "./data/params.csv"
ERRORS_FILE_PATH = "./data/errors.csv"
MAX_ITERATIONS = 100000
CONVERGED_THRESHOLD = 0.0000001
PERCENTEGE_THRESHOLD = 0.00000001
LEARNING_RATE = 0.01


def Mean_Squared_Error(theta0, theta1, mileage, price):
    """
    Calculate the Mean Squared Error (MSE) between predicted price and
      actual price.
    """
    mse = ((1 / len(mileage)) *
           sum(((theta0 + theta1 * mileage) - price) ** 2))
    return mse


def update_params(mileageN, price, theta0, theta1, lr):
    theta0 -= lr * (1 / len(mileageN)) *\
        sum(theta0 + theta1 * mileageN - price)
    theta1 -= lr * (1 / len(mileageN)) *\
        sum((theta0 + theta1 * mileageN - price) * mileageN)
    return theta0, theta1


def standardize_data(data):
    """
    Standardize the given data by subtracting the mean and dividing by the
      standard deviation.

    This function takes an array-like input `data` and performs normalization
      on it. Normalization is a common preprocessing step in machine learning
      where the data is rescaled to have a mean of 0 and a standard deviation
      of 1, making it easier to train models as it helps in converging faster
      and prevents features with larger scales from dominating those with
      smaller scales.
    """
    # print(data)
    dataMean = data.mean()
    dataStd = data.std()
    dataS = (data - dataMean) / dataStd
    # print(dataS)
    return dataS, dataMean, dataStd


def gradient_descent(mileageS, mileageMean, mileageStd, price, lr):
    """
    Perform gradient descent optimization to find optimal parameters for
      linear regression.

    This function implements gradient descent optimization to find the optimal
      parameters (theta0, theta1) for a linear regression model. It iteratively
      updates the parameters based on the mean squared error (MSE) between
      predicted and actual prices, gradually minimizing the error.
    """
    theta0 = 0
    theta1 = 0
    theta0_S = 0
    theta1_S = 0

    result = pd.DataFrame(columns=['theta0', 'theta1',
                                   'theta0_S', 'theta1_S', 'Cost'])
    result.loc[len(result)] =\
        [theta0, theta1, theta0_S, theta1_S,
            Mean_Squared_Error(theta0_S, theta1_S,
                               mileageS, price)]

    for i in range(MAX_ITERATIONS):
        prev_theta0_S, prev_theta1_S = theta0_S, theta1_S

        theta0_S, theta1_S =\
            update_params(mileageS, price, theta0_S, theta1_S, lr)
        result.loc[len(result)] =\
            [theta0, theta1, theta0_S, theta1_S,
                Mean_Squared_Error(theta0_S, theta1_S,
                                   mileageS, price)]

        # Destandardize parameters
        theta0 = theta0_S - (theta1_S * mileageMean / mileageStd)
        theta1 = theta1_S / mileageStd
        sys.stdout.write("\r{}\t->   theta0: {:.4f} | theta1: {:.4f}".
                         format(i+1, theta0, theta1))

        # Check if change in parameters is small as a percentage
        change_theta0 = abs(prev_theta0_S - theta0_S)\
            / max(abs(prev_theta0_S), PERCENTEGE_THRESHOLD)
        change_theta1 = abs(prev_theta1_S - theta1_S)\
            / max(abs(prev_theta1_S), PERCENTEGE_THRESHOLD)
        if change_theta0 < CONVERGED_THRESHOLD and\
                change_theta1 < CONVERGED_THRESHOLD:
            print("\n⭐️ Convergence reached. Stop gradient descent.")
            break

        if i == MAX_ITERATIONS - 1:
            print("\n⭐️ Number of iterations reached. Stop gradient descent.")
            break

    return theta0, theta1, result


def main():
    try:

        assert len(sys.argv) == 1, "❗️ Wrong number of arguments"

        # Load data
        data = pd.read_csv(DATA_FILE_PATH)
        assert data is not None, "❌ The dataset is wrong"

        print("📊 'data.csv' info() ")
        data.info()
        print(data.describe())
        input("\nPress Enter to continue...\n")

        mileage = data.loc[:, 'km']
        price = data.loc[:, 'price']

        plot_price_vs_mileage(mileage, price)

        # Standardize data
        mileageS, mileageMean, mileageStd = standardize_data(mileage)

        start_time = time.time()

        # Train model
        theta0, theta1, result = \
            gradient_descent(mileageS, mileageMean, mileageStd,
                             price, LEARNING_RATE)

        errors = (theta0 + theta1 * mileage - price).values

        errors_df = pd.DataFrame({'errors': errors})
        errors_df.to_csv(ERRORS_FILE_PATH, index=False)

        print(result)

        print("🟢 Training model result ")
        print("     theta0 -> {:.4f}".format(theta0))
        print("     theta1 -> {:.4f}".format(theta1))

        # Save parameters
        params = pd.DataFrame({'theta0': [theta0], 'theta1': [theta1]})
        params.to_csv(PARAMS_FILE_PATH, index=False)

        end_time = time.time()
        training_time = end_time - start_time

        print("\n⌛️ Training completed in {:.2f} seconds".
              format(training_time))

        plot_regression_animation(mileage, price, result)
        plot_price_vs_mileage_with_linear_reg(mileage, price, theta0, theta1)
        plot_cost_evolution(result)
        plot_theta0_and_theta1(result)
        plot_normal_distribution(errors)

        calculated_data = \
            pd.DataFrame({'km': mileage,
                          'price': price,
                          'estimated price': theta0 + theta1 * mileage,
                          'error': errors,
                          '% error': (errors / price) * 100})
        print("\n📊 Calculated Data:")
        print(calculated_data)
        print(calculated_data.describe())
        input("\nPress Enter to continue...\n")

        return

    except AssertionError as msg:
        print("❌ Error: ", msg)

    except FileNotFoundError:
        print("❌ Error: File not found.")

    except Exception as error:
        print("❌ Error: ", error)


if __name__ == "__main__":
    main()
