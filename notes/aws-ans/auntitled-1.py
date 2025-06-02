# %%
# Regression Problem

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import datetime
from datetime import datetime

# %%
# Get current date in YYYY-MM-DD format
current_date = datetime.now().strftime('%Y-%m-%d')

# %%
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# %%
df_train.head()

# %%
df_test.head()

# %%
df_train.info()

# %%
#df_train.isnull().sum()
# Display only the column names where there are missing values
df_train_columns_with_missing_values = df_train.columns[df_train.isnull().sum() > 0]
print(df_train_columns_with_missing_values)

# %%
# Identify categorical columns
categorical_columns = df_train.select_dtypes(include=['object']).columns
print("Categorical columns:", list(categorical_columns))

# %%
# Identify numerical columns
numerical_columns = df_train.select_dtypes(include=['int64', 'float64']).columns
print("\nNumerical columns:", list(numerical_columns))

# %%
# Print value counts for each categorical column
print("\nValue counts for each categorical column:")
print("==========================================")
for categorical_column in categorical_columns:
    print(f"\nColumn: {categorical_column}")
    print(df_train[categorical_column].value_counts())
    print("Null values:", df_train[categorical_column].isnull().sum())
    print("------------------------------------------------------")

# %%
#Initialize transformers
# cat_imputer = SimpleImputer(strategy='most_frequent')
# num_imputer = SimpleImputer(strategy='median')
# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# scaler = StandardScaler()

# Initialize transformers
cat_imputer = SimpleImputer(strategy='constant', fill_value='None')
num_imputer = SimpleImputer(strategy='median')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler = StandardScaler()

# %%
def get_categorical_numerical_columns(df):
    """Identify categorical and numerical columns with improved column selection"""
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Exclude ID and target variables
    numerical_columns = numerical_columns.drop(['Id', 'SalePrice'] if 'SalePrice' in numerical_columns else ['Id'])

    # Identify ordinal columns (could be treated differently)
    ordinal_columns = [
        'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',
        'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
        'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
    ]

    return list(categorical_columns), list(numerical_columns), ordinal_columns


# %%
def process_numerical(df, is_training=True):
    """Enhanced numerical processing with safety checks"""
    global num_imputer, scaler
    _, numerical_columns, _ = get_categorical_numerical_columns(df)

    # Handle specific columns differently
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    if is_training:
        # Handle outliers for specific columns
        outlier_columns = ['GrLivArea', 'TotalBsmtSF', 'LotArea']
        for col in outlier_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        num_imputer = SimpleImputer(strategy='median')
        df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])
    else:
        df[numerical_columns] = num_imputer.transform(df[numerical_columns])

    return df

# %%
def create_interaction_features(df):
    """Create interaction features with safety checks"""
    df = df.copy()

    # Add small epsilon to prevent division by zero
    epsilon = 1e-8

    # Quality-related interactions
    df['Quality_Space'] = df['OverallQual'] * df['GrLivArea']
    df['Quality_Age'] = df['OverallQual'] * df['Age']
    df['Quality_Garage'] = df['OverallQual'] * df['GarageCars']

    # Area-related interactions (with safety checks)
    df['Living_Lot_Ratio'] = (df['GrLivArea'] / (df['LotArea'] + epsilon)).clip(0, 1e6)
    df['Total_Bath_Bed_Ratio'] = (df['TotalBaths'] / (df['BedroomAbvGr'] + epsilon)).clip(0, 1e2)

    # Age-related interactions (with safety checks)
    df['Age_Quality'] = df['Age'] * df['OverallQual']
    df['Remod_Age_Ratio'] = (df['RemodAge'] / (df['Age'] + epsilon)).clip(-1e6, 1e6)

    return df

# %%
def feature_engineering(df):
    """Enhanced feature engineering with safety checks"""
    df = df.copy()

    # Basic features
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBaths'] = df['FullBath'] + (0.5 * df['HalfBath']) + \
                       df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + \
                         df['3SsnPorch'] + df['ScreenPorch']

    # New features
    df['TotalQual'] = df['OverallQual'] + df['OverallCond']
    # Add epsilon to prevent division by zero
    df['AvgRoomSize'] = (df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1e-8)).clip(0, 1e6)
    df['HasHighQuality'] = (df['OverallQual'] >= 8).astype(int)
    df['IsNew'] = (df['YearBuilt'] == df['YrSold']).astype(int)
    df['HasRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

    # Binary features
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)

    # Create interaction features
    df = create_interaction_features(df)

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

# %%
# First, let's modify the process_categorical function to ensure all categorical variables are encoded
def process_categorical(df, is_training=True):
    """Enhanced categorical processing with complete encoding"""
    global cat_imputer, encoder
    categorical_columns, _, _ = get_categorical_numerical_columns(df)

    # Get all object columns that remain in the dataframe
    remaining_cat_cols = df.select_dtypes(include=['object']).columns

    if is_training:
        cat_imputer = SimpleImputer(strategy='constant', fill_value='None')
        df[remaining_cat_cols] = cat_imputer.fit_transform(df[remaining_cat_cols])

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(df[remaining_cat_cols])
    else:
        df[remaining_cat_cols] = cat_imputer.transform(df[remaining_cat_cols])
        encoded_features = encoder.transform(df[remaining_cat_cols])

    encoded_feature_names = encoder.get_feature_names_out(remaining_cat_cols)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)

    # Drop original categorical columns and add encoded ones
    df = df.drop(columns=remaining_cat_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df

# %%
def prepare_data(df, is_training=True):
    """Main function with improved pipeline"""
    df = df.copy()

    # Feature engineering first
    df = feature_engineering(df)

    # Process numerical data
    df = process_numerical(df, is_training)

    # Process categorical data
    df = process_categorical(df, is_training)

    # Scale numerical features
    _, numerical_columns, _ = get_categorical_numerical_columns(df)
    if is_training:
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    else:
        df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Drop unnecessary columns
    columns_to_drop = ['Id'] + ['MiscFeature', 'Fence', 'PoolQC', 'Alley']
    if not is_training and 'SalePrice' in df.columns:
        columns_to_drop.append('SalePrice')

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    return df

# %%
# Now let's create a function to analyze correlations
def analyze_correlations(df_train_processed, target='SalePrice'):
    """Analyze and visualize correlations"""
    # Get numerical columns only
    numerical_cols = df_train_processed.select_dtypes(include=['float64', 'int64']).columns

    # Calculate correlations for numerical columns only
    correlation_matrix = df_train_processed[numerical_cols].corr()

    # Print correlations with target
    if target in numerical_cols:
        print("\nTop 10 Correlations with {}:".format(target))
        print(correlation_matrix[target].sort_values(ascending=False)[:10])

    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return correlation_matrix


# %%
df_train_processed = prepare_data(df_train.copy(), is_training=True)
df_test_processed = prepare_data(df_test.copy(), is_training=False)

# Log transform the target variable
df_train_processed['SalePrice'] = np.log1p(df_train['SalePrice'])

# %%
df_train_processed.isnull().sum()

# %%
df_train_processed.info()

# %%
df_test_processed.isnull().sum()

# %%
df_test_processed.info()

# %%
# Analyze correlations
correlation_matrix = analyze_correlations(df_train_processed)

#correlation_matrix_train = df_train_processed.corr()
#print(correlation_matrix_train)

# %%
# Prepare training data
X = df_train_processed.drop('SalePrice', axis=1)
y = df_train_processed['SalePrice']

# %%
# Save test IDs
test_ids = df_test['Id']
X_test = df_test_processed

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

# Print shapes
print("\nData Shapes:")
print("Training features shape:", X_train.shape)
print("Validation features shape:", X_val.shape)
print("Test features shape:", X_test.shape)

# %%
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Ensure balanced split for binary classification
)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

# %%
# Visualize important features
def plot_important_features(df, target='SalePrice'):
    """Plot important feature relationships"""
    important_numerical = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']

    # Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for idx, feature in enumerate(important_numerical):
        sns.scatterplot(data=df, x=feature, y=target, ax=axes[idx])
        axes[idx].set_title(f'{feature} vs {target}')

    plt.tight_layout()
    plt.show()

    # Distribution of target
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df_train['SalePrice'], ax=ax1)
    ax1.set_title('Original SalePrice Distribution')

    sns.histplot(df_train_processed['SalePrice'], ax=ax2)
    ax2.set_title('Log-transformed SalePrice Distribution')

    plt.tight_layout()
    plt.show()

# Plot important features
plot_important_features(df_train_processed)


# %%
# Scatter plots of important numerical features vs SalePrice
def plot_scatter_with_price(df, features, target='SalePrice'):
    n = len(features)
    fig, axes = plt.subplots(n//2, 2, figsize=(12, 4*n//2))
    axes = axes.ravel()

    for idx, feature in enumerate(features):
        sns.scatterplot(data=df, x=feature, y=target, ax=axes[idx])
        axes[idx].set_title(f'{feature} vs {target}')

    plt.tight_layout()
    plt.show()

important_features = ['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageArea']
plot_scatter_with_price(df_train_processed, important_features)

# %%
# Detect and print outliers
def analyze_outliers(df, features, target='SalePrice', n_std=3):
    """Analyze outliers in important features"""
    for feature in features:
        mean = df[feature].mean()
        std = df[feature].std()
        outliers = df[(df[feature] < mean - n_std * std) | 
                     (df[feature] > mean + n_std * std)]

        if len(outliers) > 0:
            print(f"\nOutliers in {feature}:")
            print(outliers[[feature, target]].head())

# Analyze outliers
important_features = ['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageArea']
analyze_outliers(df_train_processed, important_features)

# %%
### Train

# %%
# Cell 10: Train XGBoost model
import xgboost as xgb

# %%
# Define hyperparameters for regression with some adjustments
hyperparameters = {
    'n_estimators': 1000,  # Increased from 500
    'max_depth': 4,        # Reduced from 5 to prevent overfitting
    'learning_rate': 0.01,
    'subsample': 0.7,      # Reduced from 0.8 to prevent overfitting
    'colsample_bytree': 0.8,  # Reduced from 0.9
    'min_child_weight': 3,    # Added to prevent overfitting
    'gamma': 0.1,            # Added to prevent overfitting
    'reg_alpha': 0.1,        # L1 regularization
    'reg_lambda': 1.0,       # L2 regularization
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'early_stopping_rounds': 50,  # Increased from 20
    'verbosity': 1,
    'objective': 'reg:squarederror'
}

# %%
# Create and train the model
model = xgb.XGBRegressor(**hyperparameters)

# Fit with evaluation set
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100  # Print every 100 iterations
)

# %%
# Make predictions
y_pred_val = model.predict(X_val)
y_pred_train = model.predict(X_train)

# %%
# Calculate metrics for both training and validation sets
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
val_r2 = r2_score(y_val, y_pred_val)

print("\nModel Performance:")
print("=================")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation R² Score: {val_r2:.4f}")

# %%
# Plot actual vs predicted
plt.figure(figsize=(10, 5))

# Training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual log(SalePrice)')
plt.ylabel('Predicted log(SalePrice)')
plt.title('Training: Actual vs Predicted')

# Validation set
plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_val, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Actual log(SalePrice)')
plt.ylabel('Predicted log(SalePrice)')
plt.title('Validation: Actual vs Predicted')

plt.tight_layout()
plt.show()

# %%
# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.show()

# %%
# Print top 20 features
print("\nTop 20 Most Important Features:")
print("==============================")
print(feature_importance.head(20))

# %%

# Make predictions on test set
test_predictions = model.predict(X_test)

# Convert predictions back to original scale
test_predictions_original = np.expm1(test_predictions)

# %%
# Create submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions_original
})

# %%
# Save processed training data with target
df_train_processed_with_target = X.copy()
df_train_processed_with_target['SalePrice'] = y
df_train_processed_with_target.to_csv(f'data-out/train-processed-{current_date}.csv', index=False)

# Save processed test data
X_test.to_csv(f'data-out/test-processed-{current_date}.csv', index=False)

# %%
# Save submission file
submission.to_csv(f'data-out/prediction_{current_date}_v2.csv', index=False)
print(f"\nSaved submission to: data-out/prediction_{current_date}.csv")


# %%
# Additional visualization: Feature importance distribution
plt.figure(figsize=(10, 6))
sns.histplot(feature_importance['importance'], bins=50)
plt.title('Distribution of Feature Importance Values')
plt.xlabel('Importance Score')
plt.ylabel('Count')
plt.show()


# %%
# Print top 20 most important features
print("\nTop 20 Most Important Features:")
print("===============================")
print(feature_importance.head(20))

# %%
# Optional: Learning curves
evals_result = model.evals_result()

plt.figure(figsize=(10, 6))
plt.plot(evals_result['validation_0']['rmse'], label='Training')
plt.plot(evals_result['validation_1']['rmse'], label='Validation')
plt.xlabel('Boosting Round')
plt.ylabel('RMSE')
plt.title('XGBoost Learning Curves')
plt.legend()
plt.show()

# %%



