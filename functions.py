from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import kagglehub
import shutil
import os
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def encode_categorical_features(df):
    """
    Encode categorical variables to numerical values

    Args:
        df: Input dataframe to encode

    Returns:
        df_encoded: Dataframe with categorical features encoded to numbers
    """

    # Create a copy to avoid modifying original dataframe
    df_encoded = df.copy()

    # Ordinal mappings
    ordinal_mappings = {
        "Parental_Involvement": {"Low": 1, "Medium": 2, "High": 3},
        "Access_to_Resources": {"Low": 1, "Medium": 2, "High": 3},
        "Motivation_Level": {"Low": 1, "Medium": 2, "High": 3},
        "Family_Income": {"Low": 1, "Medium": 2, "High": 3},
        "Teacher_Quality": {"Low": 1, "Medium": 2, "High": 3},
        "Parental_Education_Level": {"High School": 1, "College": 2, "Postgraduate": 3},
        "Distance_from_Home": {"Near": 1, "Moderate": 2, "Far": 3},
        "Peer_Influence": {"Negative": 1, "Neutral": 2, "Positive": 3},
    }

    # Binary mappings
    binary_mappings = {
        "Extracurricular_Activities": {"No": 0, "Yes": 1},
        "Internet_Access": {"No": 0, "Yes": 1},
        "School_Type": {"Public": 0, "Private": 1},
        "Learning_Disabilities": {"No": 0, "Yes": 1},
        "Gender": {"Male": 0, "Female": 1},
    }

    # Apply ordinal mappings
    for col, mapping in ordinal_mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
            print(f"Encoded {col}")

    # Apply binary mappings
    for col, mapping in binary_mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
            print(f"Encoded {col}")

    print("\nEncoding completed!")
    print(
        f"Categorical columns encoded: {len(ordinal_mappings) + len(binary_mappings)}"
    )

    return df_encoded


# its obvious that there are missing values in three columns:
# these cols are Teacher_Quality, Parental_Education_Level, Distance_from_Home
# Teacher_Quality -> ordered categorical
# Parental_Education_Level -> ordered categorical
# Distance_from_Home -> ordered categorical


def fill_missing_values(df):
    """
    Fill missing values in the dataframe

    Args:
        df: Input dataframe with missing values (should be already encoded)

    Returns:
        df_filled: Dataframe with missing values filled
    """

    # Create a copy to avoid modifying original dataframe
    df_filled = df.copy()

    print("Missing values before filling:")
    print(df_filled.isnull().sum())

    # Columns with missing values (after encoding these should be numeric)
    ordinal_cols_with_missing = [
        "Teacher_Quality",
        "Parental_Education_Level",
        "Distance_from_Home",
    ]

    # Fill missing values with median for ordinal columns -> produced worse results than the org missing vlaues
    for col in ordinal_cols_with_missing:
        if col in df_filled.columns and df_filled[col].isnull().sum() > 0:
            median_value = df_filled[col].median()
            df_filled[col].fillna(median_value, inplace=True)
            print(f"Filled {col} missing values with median: {median_value}")

    # # Fill missing values with mode for ordinal columns
    # for col in ordinal_cols_with_missing:
    #     if col in df_filled.columns and df_filled[col].isnull().sum() > 0:
    #         mode_value = df_filled[col].mode()[0]
    #         df_filled[col].fillna(mode_value, inplace=True)
    #         print(f"Filled {col} missing values with mode: {mode_value}")

    # Fill any missing values in columns
    # For numerical columns, use median -> we don't have any numerical columns with missing values
    # For any  categorical columns ->  use mode
    categorical_cols = df_filled.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df_filled[col].isnull().sum() > 0:
            mode_value = df_filled[col].mode()[0]
            df_filled[col].fillna(mode_value, inplace=True)
            print(f"Filled {col} missing values with mode: {mode_value}")

    print("\nMissing values after filling:")
    print(df_filled.isnull().sum())

    print(f"\nTotal missing values remaining: {df_filled.isnull().sum().sum()}")

    return df_filled


def train_and_evaluate_model(
    X,
    y,
    model_type="compare",
    test_size=0.2,
    random_state=42,
    # show_feature_importance=False,
):
    """
    Train a Linear Regression model and evaluate its performance

    Args:
        X: Feature matrix
        y: Target variable
        model_name: Name for the model (for display purposes)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        dict: Dictionary containing model, predictions, and metrics
    """

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train the model
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    if model_type == "linear":
        model = LinearRegression()

    elif model_type == "ridge":
        model = Ridge(alpha=1.0, random_state=random_state)

    elif model_type == "lasso":
        model = Lasso(alpha=0.1, random_state=random_state)

    elif model_type == "elastic":
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state)

    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    elif model_type == "xgb":
        model = XGBRegressor(n_estimators=100, random_state=random_state)

    elif model_type == "svr":
        model = SVR(kernel="rbf", C=1.0)

    elif model_type == "compare":
        models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=random_state),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=random_state
            ),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=random_state),
        }

        results = {}
        best_model = None
        best_r2 = -float("inf")

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)

            results[name] = {
                "model": model,
                "r2": r2,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
            }

            if r2 > best_r2:
                best_r2 = r2
                best_model = name

            print(f"\n{name} Results:")
            print(f"R² Score: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")

        print(f"\n🏆 Best Model: {best_model} (R² = {best_r2:.4f})")

        return results

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print results
    print("\n" + "=" * 50)
    print("PERFORMANCE")
    print("=" * 50)
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    # if show_feature_importance:
    #     # Feature importance (coefficients)
    #     feature_importance = pd.DataFrame(
    #         {"Feature": X.columns, "Coefficient": model.coef_}
    #     ).sort_values("Coefficient", key=abs, ascending=False)

    #     print(f"\nTop 10 Most Important Features:")
    #     print(feature_importance.head(10))

    return model
    # # Return everything for further use
    # return {
    #     'model': model,
    #     'X_train': X_train,
    #     'X_test': X_test,
    #     'y_train': y_train,
    #     'y_test': y_test,
    #     'y_pred': y_pred,
    #     'metrics': {
    #         'r2': r2,
    #         'mse': mse,
    #         'rmse': rmse,
    #         'mae': mae
    #     }
    # }
