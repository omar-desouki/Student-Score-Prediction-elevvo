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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


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


def normalize_features(
    df_train,
    df_test,
    method="standard",
    normalize_target=False,
    target_col="Exam_Score",
):
    """
    Normalize features using train/test split approach
    Fits scaler on training data and transforms both train and test sets

    Args:
        df_train: Training dataframe with features to normalize
        df_test: Test dataframe with features to normalize
        method: Normalization method ('standard', 'minmax', 'robust')
        normalize_target: Whether to normalize the target column
        target_col: Name of the target column to normalize if applicable

    Returns:
        df_train_normalized: Normalized training dataframe
        df_test_normalized: Normalized test dataframe
        scalers: Dictionary of scalers used for normalization
    """

    # Create copies to avoid modifying original dataframes
    df_train_normalized = df_train.copy()
    df_test_normalized = df_test.copy()
    scalers = {}

    # Select numerical columns for normalization (excluding target if not normalizing it)
    numerical_cols = df_train_normalized.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # Remove target column from features to normalize if not specified
    if not normalize_target and target_col in numerical_cols:
        numerical_cols.remove(target_col)

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(
            "Invalid normalization method. Choose 'standard', 'minmax', or 'robust'."
        )

    # Fit scaler on training data only and transform both train and test
    if numerical_cols:
        scaler.fit(df_train_normalized[numerical_cols])

        df_train_normalized[numerical_cols] = scaler.transform(
            df_train_normalized[numerical_cols]
        )
        df_test_normalized[numerical_cols] = scaler.transform(
            df_test_normalized[numerical_cols]
        )

        # Store the scaler for later use
        scalers["features"] = scaler

    # # Normalize target column if specified
    # if normalize_target and target_col in df_train_normalized.columns:
    #     target_scaler = StandardScaler()  # Use standard scaling for target
    #     target_scaler.fit(df_train_normalized[[target_col]])
    #     df_train_normalized[target_col] = target_scaler.transform(
    #         df_train_normalized[[target_col]]
    #     )
    #     df_test_normalized[target_col] = target_scaler.transform(
    #         df_test_normalized[[target_col]]
    #     )
    #     scalers["target"] = target_scaler

    print(f"\nNormalized {len(numerical_cols)} features using {method} scaling.")
    print(f"Features normalized: {numerical_cols}")
    if normalize_target:
        print(f"Target '{target_col}' also normalized.")

    return df_train_normalized, df_test_normalized, scalers


def plot_feature_distribution(df, column, title=None, figsize=(12, 6)):
    """
    Plot the distribution of a feature column

    Args:
        df: DataFrame containing the data
        column: Column name to plot
        title: Custom title for the plot (optional)
        figsize: Figure size tuple (width, height)
    """

    # Set default title if not provided
    if title is None:
        title = f"Distribution of {column}"

    # Check if column exists
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in DataFrame")
        return

    # Get the data
    data = df[column].dropna()

    # Determine if data is categorical or numerical
    if df[column].dtype == "object" or len(data.unique()) <= 10:
        # Categorical data - use bar plot
        plt.figure(figsize=figsize)

        # Count values and plot
        value_counts = data.value_counts().sort_index()

        plt.subplot(1, 2, 1)
        value_counts.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title(f"{title} - Count")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        # Percentage plot
        plt.subplot(1, 2, 2)
        percentages = value_counts / len(data) * 100
        percentages.plot(kind="bar", color="lightcoral", edgecolor="black")
        plt.title(f"{title} - Percentage")
        plt.xlabel(column)
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print(f"\n📊 Summary for {column}:")
        print(f"Unique values: {len(value_counts)}")
        print(
            f"Most common: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)"
        )
        print(f"Total non-null values: {len(data)}")
        print(f"\nValue counts:")
        for val, count in value_counts.items():
            percentage = count / len(data) * 100
            print(f"  {val}: {count} ({percentage:.1f}%)")

    else:
        # Numerical data - use histogram and box plot
        plt.figure(figsize=figsize)

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(
            data,
            bins=min(30, len(data.unique())),
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
        )
        plt.title(f"{title} - Histogram")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(
            data, patch_artist=True, boxprops=dict(facecolor="lightcoral", alpha=0.7)
        )
        plt.title(f"{title} - Box Plot")
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print(f"\n📊 Summary for {column}:")
        print(f"Count: {len(data)}")
        print(f"Mean: {data.mean():.2f}")
        print(f"Median: {data.median():.2f}")
        print(f"Std: {data.std():.2f}")
        print(f"Min: {data.min()}")
        print(f"Max: {data.max()}")
        print(f"Range: {data.max() - data.min():.2f}")

        # Check for outliers using IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        print(
            f"Potential outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)"
        )


def explore_target_transformations(
    df, target_col="Exam_Score", test_size=0.2, random_state=42, show_results=False
):
    """
    Explore different target transformations to reduce skewness without data leakage.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Name of target column to transform
    test_size : float
        Proportion of data for test set
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    dict : Dictionary containing transformation results and fitted transformers
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.stats import boxcox, yeojohnson
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("TARGET COL TRANSFORMATION ANALYSIS")
    print("=" * 70)

    # Split data first - CRITICAL for avoiding data leakage
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    print(f"Train set size: {len(df_train)}")
    print(f"Test set size: {len(df_test)}")
    print(f"Target column: {target_col}")

    # Original distribution analysis (TRAIN only)
    original_skew = df_train[target_col].skew()
    print(f"\nOriginal skewness (TRAIN): {original_skew:.3f}")

    # Store transformation results
    results = {
        "train_data": df_train,
        "test_data": df_test,
        "transformations": {},
        "transform_params": {},
        "best_transformation": None,
    }

    # 1. Square root transformation
    if original_skew > 0:
        sqrt_train = np.sqrt(df_train[target_col])
        results["transformations"]["Square Root"] = sqrt_train
        results["transform_params"]["Square Root"] = {}
        print(f"Square Root skewness (TRAIN): {sqrt_train.skew():.3f}")

    # 2. Log transformation
    if original_skew > 0:
        min_score_train = df_train[target_col].min()
        log_train = np.log(df_train[target_col] + 1 - min_score_train)
        results["transformations"]["Log"] = log_train
        results["transform_params"]["Log"] = {"min_score": min_score_train}
        print(f"Log skewness (TRAIN): {log_train.skew():.3f}")

    # # 3. Quantile transformation
    # try:
    #     qt = QuantileTransformer(
    #         output_distribution="normal", random_state=random_state
    #     )
    #     quantile_train = qt.fit_transform(
    #         df_train[target_col].values.reshape(-1, 1)
    #     ).flatten()
    #     results["transformations"]["Quantile (Normal)"] = pd.Series(quantile_train)
    #     results["transform_params"]["Quantile (Normal)"] = {"transformer": qt}
    #     print(
    #         f"Quantile transformation skewness (TRAIN): {pd.Series(quantile_train).skew():.3f}"
    #     )
    # except Exception as e:
    #     print(f"Quantile transformation failed: {e}")

    # Find best transformation
    if results["transformations"]:
        best_transform = min(
            results["transformations"].items(), key=lambda x: abs(x[1].skew())
        )
        best_name = best_transform[0]
        results["best_transformation"] = best_name
        print(
            f"\n🎯 BEST TRANSFORMATION: {best_name} (skewness: {best_transform[1].skew():.3f})"
        )

    # Plot transformations (TRAIN data only)
    n_plots = len(results["transformations"]) + 1  # +1 for original
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    axes = axes.flatten()

    # Original distribution
    axes[0].hist(
        df_train[target_col], bins=30, alpha=0.7, color="skyblue", edgecolor="black"
    )
    axes[0].set_title(f"Original TRAIN\n(skew: {original_skew:.3f})", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel("Frequency")

    # Transformed distributions
    for i, (name, data) in enumerate(results["transformations"].items(), 1):
        if i < len(axes):
            color = (
                "lightgreen" if name == results["best_transformation"] else "lightcoral"
            )
            axes[i].hist(data, bins=30, alpha=0.7, color=color, edgecolor="black")
            title = f"{name} TRAIN\n(skew: {data.skew():.3f})"
            if name == results["best_transformation"]:
                title += "\n⭐ BEST"
            axes[i].set_title(title, fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlabel(f"{target_col} ({name})")
            axes[i].set_ylabel("Frequency")

    # Hide empty subplots
    for i in range(len(results["transformations"]) + 1, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

    if show_results:
        print(f"results = {results}")

    # return results


def apply_transformation_to_sets(df_train, df_test, target_col, method, params):
    """
    Apply transformation to both train and test sets using parameters from train set.
    """
    import numpy as np
    from scipy.special import inv_boxcox
    from scipy.stats import yeojohnson

    df_train_transformed = df_train.copy()
    df_test_transformed = df_test.copy()

    if method == "Square Root":
        df_train_transformed[target_col] = np.sqrt(df_train[target_col])
        df_test_transformed[target_col] = np.sqrt(df_test[target_col])

    elif method == "Log":
        min_score = params["min_score"]
        df_train_transformed[target_col] = np.log(df_train[target_col] + 1 - min_score)
        df_test_transformed[target_col] = np.log(df_test[target_col] + 1 - min_score)

    # elif method == "Box-Cox":
    #     lambda_param = params["lambda"]
    #     from scipy.special import boxcox

    #     df_train_transformed[target_col] = boxcox(df_train[target_col], lambda_param)
    #     df_test_transformed[target_col] = boxcox(df_test[target_col], lambda_param)

    # elif method == "Yeo-Johnson":
    #     lambda_param = params["lambda"]
    #     df_train_transformed[target_col] = yeojohnson(
    #         df_train[target_col], lmbda=lambda_param
    #     )
    #     df_test_transformed[target_col] = yeojohnson(
    #         df_test[target_col], lmbda=lambda_param
    #     )

    # elif method == "Quantile (Normal)":
    #     transformer = params["transformer"]
    #     # Transform train
    #     df_train_transformed[target_col] = transformer.transform(
    #         df_train[target_col].values.reshape(-1, 1)
    #     ).flatten()
    #     # Transform test using same fitted transformer
    #     df_test_transformed[target_col] = transformer.transform(
    #         df_test[target_col].values.reshape(-1, 1)
    #     ).flatten()

    return df_train_transformed, df_test_transformed


def get_inverse_transform_function(method, params):
    """
    Get function to inverse transform predictions back to original scale.
    """
    import numpy as np
    from scipy.special import inv_boxcox
    from scipy.stats import yeojohnson

    if method == "Square Root":
        return lambda x: x**2

    elif method == "Log":
        min_score = params["min_score"]
        return lambda x: np.exp(x) - 1 + min_score

    # elif method == "Box-Cox":
    #     lambda_param = params["lambda"]
    #     return lambda x: inv_boxcox(x, lambda_param)

    # elif method == "Yeo-Johnson":
    #     lambda_param = params["lambda"]
    #     return lambda x: yeojohnson(x, lmbda=lambda_param)

    # elif method == "Quantile (Normal)":
    #     transformer = params["transformer"]
    #     return lambda x: transformer.inverse_transform(x.reshape(-1, 1)).flatten()

    return lambda x: x  # Identity function as fallback


def train_and_evaluate_model(
    X,
    y,
    model_type="compare",
    test_size=0.2,
    random_state=42,
    apply_target_transform=None,  # NEW PARAMETER: None/False = no transform, dict = apply transform
    # show_feature_importance=False,
):
    """
    Train a model and evaluate its performance with optional target transformation

    Args:
        X: Feature matrix
        y: Target variable
        model_type: Type of model to train ('compare', 'linear', 'ridge', etc.)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        apply_target_transform: None/False for no transform, or dict with keys:
                              {'method': 'Log', 'params': {'min_score': 55}}

    Returns:
        dict: Dictionary containing model, predictions, and metrics
    """

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Handle target transformation
    transform_applied = False
    inverse_transform_func = None

    if apply_target_transform:
        transform_method = apply_target_transform.get("method")
        transform_params = apply_target_transform.get("params")

        if transform_method and transform_params:
            print(f"🔄 Applying {transform_method} transformation to target...")

            # Apply transformation to target
            if transform_method == "Square Root":
                y_train_transformed = np.sqrt(y_train)
                y_test_transformed = np.sqrt(y_test)

            elif transform_method == "Log":
                min_score = transform_params["min_score"]
                y_train_transformed = np.log(y_train + 1 - min_score)
                y_test_transformed = np.log(y_test + 1 - min_score)

            # elif transform_method == "Box-Cox":
            #     from scipy.special import boxcox

            #     lambda_param = transform_params["lambda"]
            #     y_train_transformed = boxcox(y_train, lambda_param)
            #     y_test_transformed = boxcox(y_test, lambda_param)

            # elif transform_method == "Yeo-Johnson":
            #     from scipy.stats import yeojohnson

            #     lambda_param = transform_params["lambda"]
            #     y_train_transformed = yeojohnson(y_train, lmbda=lambda_param)
            #     y_test_transformed = yeojohnson(y_test, lmbda=lambda_param)

            # elif transform_method == "Quantile (Normal)":
            #     transformer = transform_params["transformer"]
            #     y_train_transformed = transformer.transform(
            #         y_train.values.reshape(-1, 1)
            #     ).flatten()
            #     y_test_transformed = transformer.transform(
            #         y_test.values.reshape(-1, 1)
            #     ).flatten()

            else:
                print(f"⚠️ Unknown transformation method: {transform_method}")
                y_train_transformed = y_train
                y_test_transformed = y_test

            # Get inverse transform function
            inverse_transform_func = get_inverse_transform_function(
                transform_method, transform_params
            )
            transform_applied = True

            # Replace original targets with transformed ones for training
            y_train = y_train_transformed
            y_test = y_test_transformed

            print(f"✅ Transformation applied. Training on transformed target.")

    # Model selection
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

            # If target was transformed, inverse transform for evaluation
            if transform_applied and inverse_transform_func:
                # Inverse transform for evaluation in original scale
                y_pred_original = inverse_transform_func(y_pred)
                y_test_original = inverse_transform_func(y_test)

                # Calculate metrics on original scale
                r2 = r2_score(y_test_original, y_pred_original)
                mse = mean_squared_error(y_test_original, y_pred_original)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_original, y_pred_original)

                results[name] = {
                    "model": model,
                    "r2": r2,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "y_pred_original": y_pred_original,
                    "y_pred_transformed": y_pred_transformed,
                    "transform_applied": True,
                }

                print(f"\n{name} Results:")
                print(f"R² Score (Original Scale): {r2:.4f}")
                print(f"RMSE (Original Scale): {rmse:.4f}")
                print(f"MAE (Original Scale): {mae:.4f}")

            else:
                # Standard evaluation without transformation
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
                    "y_pred": y_pred,
                    "transform_applied": False,
                }

                print(f"\n{name} Results:")
                print(f"R² Score: {r2:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model = name

        print(f"\n🏆 Best Model: {best_model} (R² = {best_r2:.4f})")

        if transform_applied:
            print(
                f"📝 Note: Metrics calculated on original scale after inverse transformation"
            )

        # return results
        return

    # Single model training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Handle target transformation for single model
    if transform_applied and inverse_transform_func:
        y_pred_original = inverse_transform_func(y_pred)
        y_test_original = inverse_transform_func(y_test)

        # Calculate metrics on original scale
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)

        print("\n" + "=" * 50)
        print("PERFORMANCE")
        print("=" * 50)
        print(f"R² Score (Original Scale): {r2:.4f}")
        print(f"Mean Squared Error (Original Scale): {mse:.4f}")
        print(f"Root Mean Squared Error (Original Scale): {rmse:.4f}")
        print(f"Mean Absolute Error (Original Scale): {mae:.4f}")
        print(f"📝 Note: Primary metrics calculated on original scale")

    else:
        # Standard evaluation
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n" + "=" * 50)
        print("PERFORMANCE")
        print("=" * 50)
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")

    # return model
    return


# def train_with_transformation_results(
#     X, y, transformation_results, model_type="compare", test_size=0.2, random_state=42
# ):
#     """
#     Convenience function to use transformation results from explore_target_transformations()

#     Args:
#         X: Feature matrix
#         y: Original target variable
#         transformation_results: Results dict from explore_target_transformations()
#         model_type: Type of model to train
#         test_size: Test set proportion
#         random_state: Random seed

#     Returns:
#         Model results with transformation applied and inverse transformation for evaluation
#     """

#     best_method = transformation_results["best_transformation"]
#     transform_params = transformation_results["transform_params"][best_method]

#     # Create transform config
#     transform_config = {"method": best_method, "params": transform_params}

#     # Train model with transformation
#     results = train_and_evaluate_model(
#         X=X,
#         y=y,
#         model_type=model_type,
#         test_size=test_size,
#         random_state=random_state,
#         apply_target_transform=transform_config,
#     )

#     return results


# def train_and_evaluate_model(
#     X,
#     y,
#     model_type="compare",
#     test_size=0.2,
#     random_state=42,

#     # show_feature_importance=False,
# ):
#     """
#     Train a Linear Regression model and evaluate its performance

#     Args:
#         X: Feature matrix
#         y: Target variable
#         model_name: Name for the model (for display purposes)
#         test_size: Proportion of data to use for testing
#         random_state: Random seed for reproducibility

#     Returns:
#         dict: Dictionary containing model, predictions, and metrics
#     """

#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state
#     )

#     # Train the model
#     # model = LinearRegression()
#     # model.fit(X_train, y_train)
#     if model_type == "linear":
#         model = LinearRegression()

#     elif model_type == "ridge":
#         model = Ridge(alpha=1.0, random_state=random_state)

#     elif model_type == "lasso":
#         model = Lasso(alpha=0.1, random_state=random_state)

#     elif model_type == "elastic":
#         model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state)

#     elif model_type == "rf":
#         model = RandomForestRegressor(n_estimators=100, random_state=random_state)

#     elif model_type == "xgb":
#         model = XGBRegressor(n_estimators=100, random_state=random_state)

#     elif model_type == "svr":
#         model = SVR(kernel="rbf", C=1.0)

#     elif model_type == "compare":
#         models = {
#             "Linear": LinearRegression(),
#             "Ridge": Ridge(alpha=1.0, random_state=random_state),
#             "Random Forest": RandomForestRegressor(
#                 n_estimators=100, random_state=random_state
#             ),
#             "XGBoost": XGBRegressor(n_estimators=100, random_state=random_state),
#         }

#         results = {}
#         best_model = None
#         best_r2 = -float("inf")

#         for name, model in models.items():
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             r2 = r2_score(y_test, y_pred)
#             mse = mean_squared_error(y_test, y_pred)
#             rmse = np.sqrt(mse)
#             mae = mean_absolute_error(y_test, y_pred)

#             results[name] = {
#                 "model": model,
#                 "r2": r2,
#                 "mse": mse,
#                 "rmse": rmse,
#                 "mae": mae,
#             }

#             if r2 > best_r2:
#                 best_r2 = r2
#                 best_model = name

#             print(f"\n{name} Results:")
#             print(f"R² Score: {r2:.4f}")
#             print(f"RMSE: {rmse:.4f}")
#             print(f"MAE: {mae:.4f}")

#         print(f"\n🏆 Best Model: {best_model} (R² = {best_r2:.4f})")

#         return results

#     # Make predictions
#     y_pred = model.predict(X_test)

#     # Calculate regression metrics
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     # Print results
#     print("\n" + "=" * 50)
#     print("PERFORMANCE")
#     print("=" * 50)
#     print(f"R² Score: {r2:.4f}")
#     print(f"Mean Squared Error: {mse:.4f}")
#     print(f"Root Mean Squared Error: {rmse:.4f}")
#     print(f"Mean Absolute Error: {mae:.4f}")

#     # if show_feature_importance:
#     #     # Feature importance (coefficients)
#     #     feature_importance = pd.DataFrame(
#     #         {"Feature": X.columns, "Coefficient": model.coef_}
#     #     ).sort_values("Coefficient", key=abs, ascending=False)

#     #     print(f"\nTop 10 Most Important Features:")
#     #     print(feature_importance.head(10))

#     return model
#     # # Return everything for further use
#     # return {
#     #     'model': model,
#     #     'X_train': X_train,
#     #     'X_test': X_test,
#     #     'y_train': y_train,
#     #     'y_test': y_test,
#     #     'y_pred': y_pred,
#     #     'metrics': {
#     #         'r2': r2,
#     #         'mse': mse,
#     #         'rmse': rmse,
#     #         'mae': mae
#     #     }
#     # }
