from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import shutil
import os
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def explore_target_transformations(
    df, target_col="Exam_Score", test_size=0.2, random_state=42, show_results=False
):
    """
    Explore different target transformations using ONLY training data for decisions.
    Test data is only shown for verification, not decision making.
    """
    print("=" * 70)
    print("TARGET TRANSFORMATION ANALYSIS (NO DATA LEAKAGE)")
    print("=" * 70)

    # Split data first
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    print(f"Train set size: {len(df_train)}")
    print(f"Test set size: {len(df_test)}")

    y_train = df_train[target_col].copy()
    y_test = df_test[target_col].copy()

    original_skew = y_train.skew()
    print(f"\nOriginal skewness (TRAIN ONLY): {original_skew:.3f}")

    # Store transformations - DECISION BASED ONLY ON TRAINING DATA
    transformations = {"Original": y_train}

    # Only apply transformations if significantly skewed (based on TRAIN only)
    if original_skew > 0.5:
        if y_train.min() >= 0:
            transformations["Square Root"] = np.sqrt(y_train)
            print(
                f"Square Root skewness (TRAIN): {transformations['Square Root'].skew():.3f}"
            )

        if y_train.min() > 0:
            log_transformed = np.log(y_train)
            if not log_transformed.isna().any():
                transformations["Log"] = log_transformed
                print(f"Log skewness (TRAIN): {transformations['Log'].skew():.3f}")

    # Choose best transformation based ONLY on training data
    best_name = min(
        transformations.keys(), key=lambda x: abs(transformations[x].skew())
    )
    best_skew = transformations[best_name].skew()

    print(f"\n🎯 BEST TRANSFORMATION (based on TRAIN only): {best_name}")
    print(f"   Training skewness: {best_skew:.3f}")

    # Apply chosen transformation to test set for verification ONLY
    if best_name == "Square Root":
        y_test_transformed = np.sqrt(y_test)
    elif best_name == "Log":
        y_test_transformed = np.log(y_test)
    else:
        y_test_transformed = y_test

    print(f"   Test skewness (for verification): {y_test_transformed.skew():.3f}")

    # Plot only training data for decision making
    plt.figure(figsize=(15, 5))

    for i, (name, data) in enumerate(transformations.items()):
        plt.subplot(1, len(transformations), i + 1)
        color = "lightgreen" if name == best_name else "lightcoral"
        plt.hist(data, bins=30, alpha=0.7, color=color, edgecolor="black")
        title = f"{name}\n(skew: {data.skew():.3f})"
        if name == best_name:
            title += "\n⭐ CHOSEN"
        plt.title(title)
        plt.xlabel(target_col)
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

    plt.suptitle("TRANSFORMATION SELECTION (Training Data Only)", fontsize=14)
    plt.tight_layout()
    plt.show()

    return {
        "best_transformation": best_name,
        "train_skew": best_skew,
        "test_skew": y_test_transformed.skew(),  # For information only
    }


def train_linear_regression(
    X,
    y,
    regularization=None,
    grid_search=False,
    cv_folds=5,
    test_size=0.2,
    random_state=42,
    verbose=True,
):
    """
    Train and evaluate linear regression with optional regularization and grid search.

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.array
        Feature matrix
    y : pandas.Series or numpy.array
        Target variable
    regularization : str or None, default=None
        Type of regularization: None, 'ridge', 'lasso', 'elastic'
    grid_search : bool, default=False
        Whether to perform grid search for hyperparameters
    cv_folds : int, default=5
        Number of cross-validation folds
    test_size : float, default=0.2
        Proportion of dataset for testing
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Whether to print results

    Returns:
    --------
    dict : Dictionary containing model, predictions, and metrics
    """

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Define model and parameters based on regularization type
    if regularization is None:
        model = LinearRegression()
        param_grid = {}
        model_name = "Linear Regression"

    elif regularization == "ridge":
        model = Ridge(random_state=random_state)
        param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}
        model_name = "Ridge Regression"

    elif regularization == "lasso":
        model = Lasso(random_state=random_state, max_iter=1000)
        param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        model_name = "Lasso Regression"

    elif regularization == "elastic":
        model = ElasticNet(random_state=random_state, max_iter=1000)
        param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.7, 0.9]}
        model_name = "ElasticNet Regression"

    else:
        raise ValueError("regularization must be None, 'ridge', 'lasso', or 'elastic'")

    # Perform grid search if requested and parameters exist
    if grid_search and param_grid:
        if verbose:
            print(f"🔍 Performing Grid Search for {model_name}...")
            print(f"Parameter grid: {param_grid}")

        grid_search_cv = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=cv_folds, scoring="r2", n_jobs=-1
        )

        grid_search_cv.fit(X_train, y_train)
        best_model = grid_search_cv.best_estimator_

        if verbose:
            print(f"✅ Best parameters: {grid_search_cv.best_params_}")
            print(f"✅ Best CV R² score: {grid_search_cv.best_score_:.4f}")
    else:
        if verbose and grid_search and not param_grid:
            print(f"ℹ️  No hyperparameters to tune for standard Linear Regression")

        best_model = model
        best_model.fit(X_train, y_train)
        grid_search_cv = None

    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring="r2")

    # Create results dictionary
    results = {
        "model": best_model,
        "model_name": model_name,
        "grid_search_results": grid_search_cv,
        "metrics": {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
        },
        "predictions": {"y_train_pred": y_train_pred, "y_test_pred": y_test_pred},
        "data": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
    }

    # Print results
    if verbose:
        print(f"\n{'='*50}")
        print(f"📊 {model_name.upper()} RESULTS")
        print(f"{'='*50}")
        print(f"📈 R² Score:")
        print(f"   Training:   {train_r2:.4f}")
        print(f"   Testing:    {test_r2:.4f}")
        print(f"   CV Mean:    {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"\n📏 RMSE:")
        print(f"   Training:   {train_rmse:.4f}")
        print(f"   Testing:    {test_rmse:.4f}")
        print(f"\n📐 MAE:")
        print(f"   Training:   {train_mae:.4f}")
        print(f"   Testing:    {test_mae:.4f}")

        # Overfitting check
        if train_r2 - test_r2 > 0.1:
            print(f"\n⚠️  Warning: Possible overfitting!")
            print(f"   Train-Test R² gap: {train_r2 - test_r2:.4f}")
        else:
            print(f"\n✅ Model generalizes well")

    return results


def train_polynomial_regression(
    X,
    y,
    degree=2,
    grid_search=False,
    cv_folds=5,
    test_size=0.2,
    random_state=42,
    verbose=True,
):
    """
    Train and evaluate polynomial regression with optional grid search.

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.array
        Feature matrix
    y : pandas.Series or numpy.array
        Target variable
    degree : int, default=2
        Degree of polynomial features (ignored if grid_search=True)
    grid_search : bool, default=False
        Whether to perform grid search for optimal degree
    cv_folds : int, default=5
        Number of cross-validation folds
    test_size : float, default=0.2
        Proportion of dataset for testing
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Whether to print results

    Returns:
    --------
    dict : Dictionary containing model, predictions, and metrics
    """

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create polynomial regression pipeline
    poly_pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )

    # Define parameter grid for grid search
    if grid_search:
        param_grid = {"poly__degree": [1, 2, 3, 4, 5]}  # Test degrees 1-5

        if verbose:
            print(f"🔍 Performing Grid Search for Polynomial Regression...")
            print(f"Testing polynomial degrees: {param_grid['poly__degree']}")

        grid_search_cv = GridSearchCV(
            estimator=poly_pipeline,
            param_grid=param_grid,
            cv=cv_folds,
            scoring="r2",
            n_jobs=-1,
        )

        grid_search_cv.fit(X_train, y_train)
        best_model = grid_search_cv.best_estimator_
        best_degree = grid_search_cv.best_params_["poly__degree"]

        if verbose:
            print(f"✅ Best polynomial degree: {best_degree}")
            print(f"✅ Best CV R² score: {grid_search_cv.best_score_:.4f}")

            # Show all CV scores for different degrees
            print(f"\n📊 Grid Search Results:")
            results_df = pd.DataFrame(grid_search_cv.cv_results_)
            for i, degree in enumerate(param_grid["poly__degree"]):
                mean_score = results_df.iloc[i]["mean_test_score"]
                std_score = results_df.iloc[i]["std_test_score"]
                print(f"   Degree {degree}: {mean_score:.4f} (±{std_score:.4f})")
    else:
        if verbose:
            print(f"🔧 Training Polynomial Regression with degree={degree}")

        # Set the degree and train
        poly_pipeline.set_params(poly__degree=degree)
        poly_pipeline.fit(X_train, y_train)
        best_model = poly_pipeline
        best_degree = degree
        grid_search_cv = None

    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Cross-validation scores for final model
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring="r2")

    # Get number of features created
    poly_features = best_model.named_steps["poly"]
    n_original_features = X.shape[1]
    n_poly_features = poly_features.transform(X_train).shape[1]

    # Create results dictionary
    results = {
        "model": best_model,
        "model_name": f"Polynomial Regression (degree={best_degree})",
        "best_degree": best_degree,
        "grid_search_results": grid_search_cv,
        "feature_info": {
            "original_features": n_original_features,
            "polynomial_features": n_poly_features,
            "feature_expansion": f"{n_original_features} → {n_poly_features}",
        },
        "metrics": {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
        },
        "predictions": {"y_train_pred": y_train_pred, "y_test_pred": y_test_pred},
        "data": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
    }

    # Print results
    if verbose:
        print(f"\n{'='*50}")
        print(f"📊 POLYNOMIAL REGRESSION RESULTS")
        print(f"{'='*50}")
        print(f"🔢 Polynomial degree: {best_degree}")
        print(f"🔧 Feature expansion: {results['feature_info']['feature_expansion']}")
        print(f"\n📈 R² Score:")
        print(f"   Training:   {train_r2:.4f}")
        print(f"   Testing:    {test_r2:.4f}")
        print(f"   CV Mean:    {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"\n📏 RMSE:")
        print(f"   Training:   {train_rmse:.4f}")
        print(f"   Testing:    {test_rmse:.4f}")
        print(f"\n📐 MAE:")
        print(f"   Training:   {train_mae:.4f}")
        print(f"   Testing:    {test_mae:.4f}")

        # Overfitting and complexity warnings
        if train_r2 - test_r2 > 0.1:
            print(f"\n⚠️  Warning: Possible overfitting!")
            print(f"   Train-Test R² gap: {train_r2 - test_r2:.4f}")

        if best_degree > 3:
            print(
                f"\n⚠️  Warning: High polynomial degree ({best_degree}) may cause overfitting"
            )

        if n_poly_features > 100:
            print(
                f"\n⚠️  Warning: Many features created ({n_poly_features}), consider regularization"
            )
        else:
            print(f"\n✅ Model complexity seems reasonable")

    return results


# def train_and_evaluate_model(
#     X,
#     y,
#     model_type="compare",
#     test_size=0.2,
#     random_state=42,
#     apply_target_transform=None,
# ):
#     """
#     Train a model and evaluate its performance with optional target transformation
#     """

#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state
#     )

#     # Handle target transformation
#     inverse_transform_func = None
#     y_test_original = y_test.copy()

#     if apply_target_transform:
#         method = apply_target_transform.get("method")
#         params = apply_target_transform.get("params", {})

#         if method == "Square Root":
#             y_train = np.sqrt(y_train)
#             y_test_transformed = np.sqrt(y_test)
#             inverse_transform_func = lambda x: x**2

#         elif method == "Log":
#             min_score = params.get("min_score", y_train.min())
#             y_train = np.log(y_train + 1 - min_score)
#             y_test_transformed = np.log(y_test + 1 - min_score)
#             inverse_transform_func = lambda x: np.exp(x) + min_score - 1

#         print(f"Applied {method} transformation to target")

#     # Define models
#     models = {
#         "Linear": LinearRegression(),
#         "Ridge": Ridge(alpha=1.0, random_state=random_state),
#         "Random Forest": RandomForestRegressor(
#             n_estimators=100, random_state=random_state
#         ),
#         "XGBoost": XGBRegressor(n_estimators=100, random_state=random_state),
#     }

#     if model_type != "compare":
#         models = {model_type: models.get(model_type, LinearRegression())}

#     # Train and evaluate models
#     best_model_name = None
#     best_r2 = -float("inf")

#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         # Calculate metrics on original scale
#         if inverse_transform_func:
#             y_pred_original = inverse_transform_func(y_pred)
#             y_test_eval = y_test_original
#         else:
#             y_pred_original = y_pred
#             y_test_eval = y_test

#         r2 = r2_score(y_test_eval, y_pred_original)
#         rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_original))
#         mae = mean_absolute_error(y_test_eval, y_pred_original)

#         print(f"\n{name} Results:")
#         print(f"R² Score: {r2:.4f}")
#         print(f"RMSE: {rmse:.4f}")
#         print(f"MAE: {mae:.4f}")

#         if r2 > best_r2:
#             best_r2 = r2
#             best_model_name = name

#     if model_type == "compare":
#         print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")
#         if inverse_transform_func:
#             print(
#                 "📝 Metrics calculated on original scale after inverse transformation"
#             )
