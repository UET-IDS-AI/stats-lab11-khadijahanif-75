import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from AI_stats_lab import (
    generate_clean_data,
    add_outliers,
    plot_dataset_with_outliers,
    fit_linear_regression,
    fit_huber_regression,
    fit_ransac_regression,
    fit_theilsen_regression,
    coefficient_errors,
    best_robust_model,
    ransac_outlier_summary,
    plot_regression_fits,
    plot_ransac_inliers_outliers
)


# -------------------------------------------------
# Question 1 tests
# -------------------------------------------------

def test_generate_clean_data_shapes():
    X, y, true_coef = generate_clean_data()

    assert X.shape == (500, 1)
    assert y.shape == (500,)
    assert isinstance(float(true_coef), float)


def test_generate_clean_data_true_coefficient_range():
    X, y, true_coef = generate_clean_data()

    assert 50 < float(true_coef) < 80


def test_add_outliers_shapes_and_copy():
    X, y, true_coef = generate_clean_data()
    X_original = X.copy()
    y_original = y.copy()

    X_out, y_out = add_outliers(X, y)

    assert X_out.shape == X.shape
    assert y_out.shape == y.shape

    assert np.allclose(X, X_original)
    assert np.allclose(y, y_original)


def test_add_outliers_effect():
    X, y, true_coef = generate_clean_data()
    X_out, y_out = add_outliers(X, y)

    assert not np.allclose(X_out[:25], X[:25])
    assert not np.allclose(y_out[:25], y[:25])

    assert np.mean(X_out[:25]) > 8

    assert np.allclose(X_out[25:], X[25:])
    assert np.allclose(y_out[25:], y[25:])


def test_plot_dataset_with_outliers_returns_figure():
    X, y, true_coef = generate_clean_data()
    X_out, y_out = add_outliers(X, y)

    fig = plot_dataset_with_outliers(X_out, y_out)

    assert isinstance(fig, Figure)

    ax = fig.axes[0]
    assert ax.get_title() != ""
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""

    plt.close(fig)


# -------------------------------------------------
# Question 2 tests
# -------------------------------------------------

def test_fit_model_coefficients_are_floats():
    X, y, true_coef = generate_clean_data()
    X_out, y_out = add_outliers(X, y)

    lr_coef = fit_linear_regression(X_out, y_out)
    huber_coef = fit_huber_regression(X_out, y_out)
    ransac_coef = fit_ransac_regression(X_out, y_out)
    theilsen_coef = fit_theilsen_regression(X_out, y_out)

    assert isinstance(float(lr_coef), float)
    assert isinstance(float(huber_coef), float)
    assert isinstance(float(ransac_coef), float)
    assert isinstance(float(theilsen_coef), float)


def test_ordinary_linear_regression_is_badly_affected():
    X, y, true_coef = generate_clean_data()
    X_out, y_out = add_outliers(X, y)

    lr_coef = fit_linear_regression(X_out, y_out)

    error = abs(lr_coef - true_coef)

    assert error > 40


def test_robust_models_improve_over_linear_regression():
    X, y, true_coef = generate_clean_data()
    X_out, y_out = add_outliers(X, y)

    coefs = {
        "linear_regression": fit_linear_regression(X_out, y_out),
        "huber_regression": fit_huber_regression(X_out, y_out),
        "ransac_regression": fit_ransac_regression(X_out, y_out),
        "theilsen_regression": fit_theilsen_regression(X_out, y_out),
    }

    errors = coefficient_errors(coefs, true_coef)

    assert errors["ransac_regression"] < errors["linear_regression"]
    assert errors["theilsen_regression"] < errors["linear_regression"]


def test_coefficient_errors_dictionary():
    true_coef = 10.0

    coefs = {
        "linear_regression": 7.0,
        "huber_regression": 8.0,
        "ransac_regression": 9.5,
        "theilsen_regression": 11.0,
    }

    errors = coefficient_errors(coefs, true_coef)

    assert np.isclose(errors["linear_regression"], 3.0)
    assert np.isclose(errors["huber_regression"], 2.0)
    assert np.isclose(errors["ransac_regression"], 0.5)
    assert np.isclose(errors["theilsen_regression"], 1.0)


def test_best_robust_model():
    errors = {
        "linear_regression": 50.0,
        "huber_regression": 20.0,
        "ransac_regression": 2.0,
        "theilsen_regression": 5.0,
    }

    assert best_robust_model(errors) == "ransac_regression"


def test_ransac_detects_added_outliers():
    X, y, true_coef = generate_clean_data()
    X_out, y_out = add_outliers(X, y)

    total_outliers, added_outliers = ransac_outlier_summary(X_out, y_out)

    assert total_outliers >= 25
    assert added_outliers == 25


def test_plot_regression_fits_returns_figure():
    X, y, true_coef = generate_clean_data()
    X_out, y_out = add_outliers(X, y)

    fig = plot_regression_fits(X_out, y_out)

    assert isinstance(fig, Figure)

    ax = fig.axes[0]
    assert ax.get_title() != ""
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""

    # At least four fitted model lines should be present.
    assert len(ax.lines) >= 4

    plt.close(fig)


def test_plot_ransac_inliers_outliers_returns_figure():
    X, y, true_coef = generate_clean_data()
    X_out, y_out = add_outliers(X, y)

    fig = plot_ransac_inliers_outliers(X_out, y_out)

    assert isinstance(fig, Figure)

    ax = fig.axes[0]
    assert ax.get_title() != ""
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""

    plt.close(fig)
