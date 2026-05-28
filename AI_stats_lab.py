import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


# -------------------------------------------------
# Question 1: Dataset generation and visualization
# -------------------------------------------------

def generate_clean_data(
    n_samples=500,
    noise=20,
    random_state=42
):
    """
    Generate a clean synthetic regression dataset.

    Return:
        X, y, true_coef

    Requirements:
    - n_samples = 500 by default
    - n_features = 1
    - n_informative = 1
    - noise = 20
    - coef = True
    - random_state = 42
    """
    pass


def add_outliers(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Add artificial outliers to the first n_outliers observations.

    Use:
        X[:n_outliers] = 10 + 0.75 * random_normal_values
        y[:n_outliers] = -15 + 20 * random_normal_values

    Return:
        X_out, y_out

    Important:
    Do not modify the original X and y directly.
    Make copies first.
    """
    pass


def plot_dataset_with_outliers(
    X,
    y,
    n_outliers=25
):
    """
    Plot the dataset and highlight the first n_outliers observations.

    Return:
        matplotlib Figure object

    Requirements:
    - normal observations and artificial outliers should be visually different
    - include title
    - include x-label
    - include y-label
    - include legend
    """
    pass


# -------------------------------------------------
# Question 2: Fit regression models
# -------------------------------------------------

def fit_linear_regression(X, y):
    """
    Fit ordinary Linear Regression.

    Return:
        fitted coefficient as a float
    """
    pass


def fit_huber_regression(X, y):
    """
    Fit Huber Regression.

    Return:
        fitted coefficient as a float
    """
    pass


def fit_ransac_regression(X, y, random_state=42):
    """
    Fit RANSAC Regression.

    Return:
        fitted coefficient as a float

    Hint:
    RANSAC stores the final linear model in estimator_.
    """
    pass


def fit_theilsen_regression(X, y, random_state=42):
    """
    Fit Theil-Sen Regression.

    Return:
        fitted coefficient as a float
    """
    pass


def coefficient_errors(coef_dict, true_coef):
    """
    Given a dictionary of coefficients and the true coefficient,
    return a dictionary of absolute coefficient errors.

    Example input:
        {
            "linear_regression": 8.7,
            "huber_regression": 37.5,
            "ransac_regression": 62.8,
            "theilsen_regression": 59.4
        }

    Return:
        {
            "linear_regression": abs(...),
            ...
        }
    """
    pass


def best_robust_model(errors):
    """
    Return the name of the robust model with the smallest error.

    Only compare:
        huber_regression
        ransac_regression
        theilsen_regression

    Do not include ordinary linear_regression in this comparison.
    """
    pass


def ransac_outlier_summary(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Fit RANSAC and return:

        total_outliers_detected, added_outliers_detected

    total_outliers_detected:
        total number of samples classified as outliers by RANSAC

    added_outliers_detected:
        number of artificial outliers among the first n_outliers
        that RANSAC classified as outliers
    """
    pass


# -------------------------------------------------
# Question 2: Visualization functions
# -------------------------------------------------

def plot_regression_fits(
    X,
    y,
    random_state=42
):
    """
    Plot fitted regression lines for:
    - Linear Regression
    - Huber Regression
    - RANSAC Regression
    - Theil-Sen Regression

    Return:
        matplotlib Figure object

    Requirements:
    - scatter plot of data
    - fitted line for each model
    - title
    - x-label
    - y-label
    - legend
    """
    pass


def plot_ransac_inliers_outliers(
    X,
    y,
    random_state=42
):
    """
    Fit RANSAC and visualize inliers vs outliers.

    Return:
        matplotlib Figure object

    Requirements:
    - inliers and outliers should be visually different
    - title
    - x-label
    - y-label
    - legend
    """
    pass
