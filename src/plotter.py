import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import seaborn as sns

PLOTS_DIR = "./plots"
time_to_wait = 3


def plot_price_vs_mileage(mileage, price):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.plot(mileage, price, 'o', label="Dataset")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.title("Data price of car with mileage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(time_to_wait)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, "01_plot_price_vs_mileage.png")
    plt.savefig(save_path)
    print("\n⚪️ Plot saved as: {}\n".format(save_path))
    input("\nPress Enter to continue...\n")
    plt.close()


def plot_regression_animation(mileage, price, result):
    plt.figure(figsize=(10, 7))

    for i in range(0, len(result), 10):
        plt.clf()
        theta0 = result['theta0'].iloc[i]
        theta1 = result['theta1'].iloc[i]
        plt.scatter(mileage, price, color='blue')
        plt.plot(mileage, theta0 + theta1 * mileage, color='red', linewidth=3)
        plt.xlabel('Mileage (km)')
        plt.ylabel('Price (€)')
        plt.title('Linear Regression Animation (Iteration {}): '
                  'Theta0 = {:.4f} // Theta1 = {:.4f}'.
                  format(i+1, theta0, theta1))
        plt.grid(True)
        if i == 0:
            plt.pause(time_to_wait)
        if i < 400:
            pause_duration = 0.5 - (0.49 / 400) * i
        else:
            pause_duration = 0.01
        plt.pause(pause_duration)
    plt.close()
    input("\nPress Enter to continue...\n")
    plt.close()


def plot_price_vs_mileage_with_linear_reg(mileage, price, theta0, theta1):
    plt.figure(figsize=(10, 7))
    x = np.linspace(min(mileage), max(mileage), 2)
    plt.plot(mileage, price, 'o', label="Dataset")
    plt.plot(x, theta0 + theta1 * x, 'green', label="Linear Regression",
             linewidth=3)
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.title("Price evolution of car with mileage (Linear Regression)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(time_to_wait)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR,
                             "02_plot_price_vs_mileage_with_linear_reg.png")
    plt.savefig(save_path)
    print("\n⚪️ Plot saved as: {}\n".format(save_path))
    input("\nPress Enter to continue...\n")
    plt.close()


def plot_cost_evolution(result):
    plt.figure(figsize=(10, 7))
    x = range(len(result))
    plt.plot(x, result.loc[:, 'Cost'], 'green', label="Mean Square Error",
             linewidth=3)
    plt.xlabel("Algo iterations")
    plt.ylabel("Mean Square Error")
    plt.title("Evolution of 'MSE' with gradient descent iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(time_to_wait)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, "03_plot_MSE_evolution.png")
    plt.savefig(save_path)
    print("\n⚪️ Plot saved as: {}\n".format(save_path))
    input("\nPress Enter to continue...\n")
    plt.close()


def plot_theta0_and_theta1(result):
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    x = range(1, len(result))
    plt.plot(x, result.loc[1:, 'theta0'], 'green', label="theta0", linewidth=3)
    plt.xlabel("Algorithm iterations")
    plt.ylabel("theta0")
    plt.title("Evolution of 'theta0' with gradient descent iterations")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, result.loc[1:, 'theta1'], 'blue', label="theta1", linewidth=3)
    plt.xlabel("Algorithm iterations")
    plt.ylabel("theta1")
    plt.title("Evolution of 'theta1' with gradient descent iterations")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(time_to_wait)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, "04_plot_theta0_&_theta1.png")
    plt.savefig(save_path)
    print("\n⚪️ Plot saved as: {}\n".format(save_path))
    input("\nPress Enter to continue...\n")
    plt.close()


def plot_normal_distribution(errors):
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    x = np.linspace(mean_error - 3*std_error, mean_error + 3*std_error, 100)
    y = norm.pdf(x, mean_error, std_error)

    plt.figure(figsize=(10, 7))
    sns.kdeplot(errors, color='g', label='Errors Distribution',
                linewidth=3)
    plt.plot(x, y, label='Normal Distribution')
    plt.title('Errors vs. Normal Distribution')
    plt.xlabel('Errors')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(time_to_wait)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, "05_plot_errors.png")
    plt.savefig(save_path)
    print("\n⚪️ Plot saved as: {}\n".format(save_path))
    input("\nPress Enter to continue...\n")
    plt.close()


def plot_learning_rate_test(mse_history, learning_rates):
    plt.figure(figsize=(10, 7))
    for lr, mse_df in zip(learning_rates, mse_history):
        plt.plot(range(1, len(mse_df) + 1),
                 mse_df.loc[:, 'Cost'],
                 label=f"Learning Rate: {lr}")
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs. Iterations for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(time_to_wait)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, "06_plot_test_learning_rates.png")
    plt.savefig(save_path)
    print("\n⚪️ Plot saved as: {}\n".format(save_path))
    input("\nPress Enter to continue...\n")
    plt.close()
