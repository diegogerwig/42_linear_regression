import pandas as pd
import time
from time import sleep
from train import gradient_descent, standardize_data, DATA_FILE_PATH
from plotter import plot_learning_rate_test

PLOTS_DIR = "./plots"

LEARNING_RATES = [0.001, 0.01, 0.1, 1.0]


def main():
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        assert df is not None, "❌ The dataset is wrong"

        price = df.loc[:, 'price']
        mileage = df.loc[:, 'km']
        mileageS, mileageMean, mileageStd = standardize_data(mileage)

        mse_history = []

        for lr in LEARNING_RATES:
            print(f"\n🏁 Training model with learning rate: {lr}")
            start_time = time.time()

            theta0, theta1, result = gradient_descent(
                mileageS, mileageMean, mileageStd, price, lr)

            mse_history.append(result)

            print("🟢 Training model result ")
            print("     theta0 -> {:.4f}".format(theta0))
            print("     theta1 -> {:.4f}".format(theta1))
            print("     Mean Squared Error (MSE): {:.4f}".
                  format(result['Cost'].iloc[-1]))

            end_time = time.time()
            training_time = end_time - start_time

            print("\n⌛️ Training completed in {:.2f} seconds".
                  format(training_time))

            sleep(2)

        plot_learning_rate_test(mse_history, LEARNING_RATES)

    except FileNotFoundError:
        print("❌ Error: File not found.")

    except Exception as error:
        print("❌ Error: ", error)


if __name__ == "__main__":
    main()
