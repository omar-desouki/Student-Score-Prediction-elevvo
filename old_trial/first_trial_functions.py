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
import seaborn as sns


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


def train_and_evaluate_model(
    X,
    y,
    model_type="compare",
    test_size=0.2,
    random_state=42,
    apply_target_transform=None,
):
    """
    Train a model and evaluate its performance with optional target transformation
    """

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Handle target transformation
    inverse_transform_func = None
    y_test_original = y_test.copy()

    if apply_target_transform:
        method = apply_target_transform.get("method")
        params = apply_target_transform.get("params", {})

        if method == "Square Root":
            y_train = np.sqrt(y_train)
            y_test_transformed = np.sqrt(y_test)
            inverse_transform_func = lambda x: x**2

        elif method == "Log":
            min_score = params.get("min_score", y_train.min())
            y_train = np.log(y_train + 1 - min_score)
            y_test_transformed = np.log(y_test + 1 - min_score)
            inverse_transform_func = lambda x: np.exp(x) + min_score - 1

        print(f"Applied {method} transformation to target")

    # Define models
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state
        ),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=random_state),
    }

    if model_type != "compare":
        models = {model_type: models.get(model_type, LinearRegression())}

    # Train and evaluate models
    best_model_name = None
    best_r2 = -float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics on original scale
        if inverse_transform_func:
            y_pred_original = inverse_transform_func(y_pred)
            y_test_eval = y_test_original
        else:
            y_pred_original = y_pred
            y_test_eval = y_test

        r2 = r2_score(y_test_eval, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_original))
        mae = mean_absolute_error(y_test_eval, y_pred_original)

        print(f"\n{name} Results:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    if model_type == "compare":
        print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")
        if inverse_transform_func:
            print(
                "📝 Metrics calculated on original scale after inverse transformation"
            )


def plot_two_columns(df, col1, col2, plot_type="auto", figsize=(12, 8), title=None):
    """
    Create visualizations for two columns in a dataframe

    Args:
        df: DataFrame containing the data
        col1: First column name
        col2: Second column name
        plot_type: Type of plot ('auto', 'scatter', 'box', 'violin', 'bar', 'heatmap')
        figsize: Figure size tuple
        title: Custom title for the plot

    Returns:
        None (displays the plot)
    """

    # Check if columns exist
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Error: One or both columns not found in dataframe")
        return

    # Determine data types
    col1_numeric = pd.api.types.is_numeric_dtype(df[col1])
    col2_numeric = pd.api.types.is_numeric_dtype(df[col2])

    # Auto-select plot type based on data types
    if plot_type == "auto":
        if col1_numeric and col2_numeric:
            plot_type = "scatter"
        elif col1_numeric or col2_numeric:
            plot_type = "box"
        else:
            plot_type = "heatmap"

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "scatter":
        # Scatter plot for two numeric variables
        plt.scatter(df[col1], df[col2], alpha=0.6)
        plt.xlabel(col1)
        plt.ylabel(col2)

        # Add correlation coefficient
        corr = df[col1].corr(df[col2])
        plt.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

    elif plot_type == "box":
        # Box plot for categorical vs numeric
        if col1_numeric and not col2_numeric:
            sns.boxplot(data=df, x=col2, y=col1)
            plt.xticks(rotation=45)
        else:
            sns.boxplot(data=df, x=col1, y=col2)
            plt.xticks(rotation=45)

    elif plot_type == "violin":
        # Violin plot for categorical vs numeric
        if col1_numeric and not col2_numeric:
            sns.violinplot(data=df, x=col2, y=col1)
            plt.xticks(rotation=45)
        else:
            sns.violinplot(data=df, x=col1, y=col2)
            plt.xticks(rotation=45)

    elif plot_type == "bar":
        # Bar plot for categorical data
        if col1_numeric:
            # Group by col2 and get mean of col1
            grouped = df.groupby(col2)[col1].mean().sort_values(ascending=False)
            grouped.plot(kind="bar")
            plt.ylabel(f"Mean {col1}")
            plt.xlabel(col2)
        else:
            # Group by col1 and get mean of col2
            grouped = df.groupby(col1)[col2].mean().sort_values(ascending=False)
            grouped.plot(kind="bar")
            plt.ylabel(f"Mean {col2}")
            plt.xlabel(col1)
        plt.xticks(rotation=45)

    elif plot_type == "heatmap":
        # Heatmap for categorical vs categorical
        crosstab = pd.crosstab(df[col1], df[col2])
        sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues")
        plt.xlabel(col2)
        plt.ylabel(col1)

    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f"{col1} vs {col2}")

    plt.tight_layout()
    plt.show()


def train_polynomial_regression(
    X,
    y,
    test_size=0.2,
    random_state=42,
    apply_target_transform=None,
    grid_search=True,
    cv_folds=5,
):
    """
    Train polynomial regression with optional target transformation and hyperparameter tuning

    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        apply_target_transform: Dict with transformation config {'method': 'Log'/'Square Root', 'params': {}}
        grid_search: Whether to perform grid search for best hyperparameters
        cv_folds: Number of cross-validation folds for grid search

    Returns:
        dict: Results containing best model, metrics, and hyperparameters
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Ridge

    print("🔄 Training Polynomial Regression Model")
    print("=" * 50)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Handle target transformation
    inverse_transform_func = None
    y_test_original = y_test.copy()

    if apply_target_transform:
        method = apply_target_transform.get("method")
        params = apply_target_transform.get("params", {})

        if method == "Square Root":
            y_train = np.sqrt(y_train)
            inverse_transform_func = lambda x: x**2

        elif method == "Log":
            min_score = params.get("min_score", y_train.min())
            y_train = np.log(y_train + 1 - min_score)
            inverse_transform_func = lambda x: np.exp(x) + min_score - 1

        print(f"✅ Applied {method} transformation to target")

    # Create polynomial regression pipeline
    poly_pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(include_bias=False)),
            ("ridge", Ridge(random_state=random_state)),
        ]
    )

    if grid_search:
        # Define parameter grid for grid search
        param_grid = {
            "poly__degree": [1, 2, 3, 4],
            "ridge__alpha": [0.1, 1.0, 10.0, 100.0],
        }

        print(f"🔍 Performing Grid Search with {cv_folds}-fold CV...")
        print(
            f"Parameter combinations to test: {len(param_grid['poly__degree']) * len(param_grid['ridge__alpha'])}"
        )

        # Perform grid search
        grid_search_cv = GridSearchCV(
            poly_pipeline, param_grid, cv=cv_folds, scoring="r2", n_jobs=-1, verbose=1
        )

        grid_search_cv.fit(X_train, y_train)
        best_model = grid_search_cv.best_estimator_
        best_params = grid_search_cv.best_params_
        best_cv_score = grid_search_cv.best_score_

        print(f"✅ Best CV R² Score: {best_cv_score:.4f}")
        print(f"✅ Best Parameters: {best_params}")

    else:
        # Use default parameters
        best_model = poly_pipeline
        best_params = {"poly__degree": 2, "ridge__alpha": 1.0}
        best_cv_score = None

        print("📝 Using default parameters (degree=2, alpha=1.0)")
        best_model.fit(X_train, y_train)

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Calculate metrics on original scale
    if inverse_transform_func:
        y_pred_original = inverse_transform_func(y_pred)
        y_test_eval = y_test_original
        scale_note = " (Original Scale)"
    else:
        y_pred_original = y_pred
        y_test_eval = y_test
        scale_note = ""

    # Calculate evaluation metrics
    r2 = r2_score(y_test_eval, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_original))
    mae = mean_absolute_error(y_test_eval, y_pred_original)

    # Display results
    print("\n📊 Test Set Performance")
    print("=" * 30)
    print(f"R² Score{scale_note}: {r2:.4f}")
    print(f"RMSE{scale_note}: {rmse:.4f}")
    print(f"MAE{scale_note}: {mae:.4f}")

    if inverse_transform_func:
        print("📝 Metrics calculated after inverse transformation")

    # Prepare results dictionary
    results = {
        "model": best_model,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "test_r2": r2,
        "test_rmse": rmse,
        "test_mae": mae,
        "predictions": y_pred_original,
        "true_values": y_test_eval,
        "transformation_applied": apply_target_transform is not None,
        "inverse_transform_func": inverse_transform_func,
    }

    return results
