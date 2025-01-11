# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. Load and Clean the Data
# ------------------------------------------------------------------------------
# Load your dataset
file_path = '/Users/willjones/Desktop/DataScience/diss_double_2.csv'  # Replace with your CSV file's location
df = pd.read_csv(file_path)

# Display original column names
print("Original Columns:")
print(df.columns.tolist())

# Clean column names: remove leading/trailing spaces and convert to lowercase
df.columns = df.columns.str.strip().str.lower()
print("\nCleaned Columns:")
print(df.columns.tolist())

# ------------------------------------------------------------------------------
# 2. Drop Unwanted Columns
# ------------------------------------------------------------------------------
# Specify columns to drop (ensure they are in lowercase)
columns_to_drop = [
    'finished', 'manager_name', 'team_name', 'season_name', 'start_time', 'lose', 'draw', 'points',
    'manager_change_last_1', 'manager_change_last_3', 'manager_change_last_5',
    'manager_change_last_10', 'manager_change_last_15', 'manager_change_last_20',
    'points_j_to_3', 'prop_home_j_to_3', 'prop_home_j_to_5', 'prop_home_j_to_10',
    'prop_home_j_to_15', 'prop_home_j_to_20'
]

# Drop the specified columns, ignore if some columns are missing
df = df.drop(columns=columns_to_drop, axis=1, errors='ignore')
print("\nColumns After Dropping Unwanted Columns:")
print(df.columns.tolist())

# ------------------------------------------------------------------------------
# 3. Manual Train-Test Split
# ------------------------------------------------------------------------------
# Define manual train and test indices
train_start, train_end = 18242, 18639  # Inclusive
test_start, test_end = 18640, 18659    # Inclusive

# Select train and test DataFrames
train_df = df.iloc[train_start:train_end + 1].copy()  # +1 because iloc is exclusive at the end
test_df = df.iloc[test_start:test_end + 1].copy()

# ------------------------------------------------------------------------------
# 4. Handle Missing Values
# ------------------------------------------------------------------------------
# For the training set: drop rows where 'win' is NaN
train_df = train_df.dropna(subset=['win'])
print("\nAfter Dropping NaNs in 'win' from Train Set:")
print(f"Train shape: {train_df.shape}")

# For the test set: **do not** drop rows with NaN in 'win' since you want to predict these
# However, ensure that other feature columns do not have NaNs
# Optionally, you can handle NaNs in features here if needed
print("\nTest shape before handling NaNs in features:", test_df.shape)

# Check for NaNs in feature columns of test set
# Assuming 'win' is the last column after dropping unwanted columns
feature_columns = test_df.columns.tolist()
if 'win' in feature_columns:
    feature_columns.remove('win')  # Exclude 'win' since it's to be predicted

# Handle NaNs in features
for col in feature_columns:
    if test_df[col].dtype in ['float64', 'int64']:
        # Fill numerical columns with mean
        test_df[col].fillna(test_df[col].mean(), inplace=True)
    else:
        # Fill categorical columns with mode
        if not test_df[col].mode().empty:
            test_df[col].fillna(test_df[col].mode()[0], inplace=True)
        else:
            # If mode is empty (all values are NaN), fill with a placeholder
            test_df[col].fillna('Unknown', inplace=True)

# Verify no NaNs remain in test features
print("Test shape after handling NaNs in features:", test_df.shape)
print("Any NaNs in test features?", test_df[feature_columns].isnull().values.any())

# ------------------------------------------------------------------------------
# 5. Define Features and Target
# ------------------------------------------------------------------------------
# Define target variable for training
y_train = train_df['win']

# Define feature sets by dropping the target
X_train = train_df.drop('win', axis=1)
X_test = test_df.drop('win', axis=1)

# ------------------------------------------------------------------------------
# 6. Train the Random Forest Model
# ------------------------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    random_state=42,
    class_weight='balanced'  # Helps with class imbalance
)
rf_model.fit(X_train, y_train)
print("\nRandom Forest model trained successfully.")

# ------------------------------------------------------------------------------
# 7. Make Predictions on Test Set
# ------------------------------------------------------------------------------
# Predict class labels for test set
y_pred = rf_model.predict(X_test)
print("\nPredictions on Test Set:")
print(y_pred)

# Generate predicted probabilities for test set
y_proba = rf_model.predict_proba(X_test)

# ------------------------------------------------------------------------------
# 8. Apply Custom Probability Threshold
# ------------------------------------------------------------------------------
# Define a custom threshold
custom_threshold = 0.35

# Confirm the classes
print("\nClasses in the model:", rf_model.classes_)

# Assuming '1' is the positive class for 'win'
# Find the index of the positive class
if 1 in rf_model.classes_:
    positive_class_index = list(rf_model.classes_).index(1)
else:
    # If '1' is not present, use the last class as positive
    positive_class_index = -1  # Modify based on your specific case

y_proba_win = y_proba[:, positive_class_index]

# Apply the custom threshold
y_pred_custom = (y_proba_win >= custom_threshold).astype(int)
print("\nPredictions with Custom Threshold:")
print(y_pred_custom)

# ------------------------------------------------------------------------------
# 9. Create a Predictions DataFrame with Row Numbers
# ------------------------------------------------------------------------------
# To include row numbers, we'll use the original DataFrame's indices
# Since test_df is a slice of df, its index corresponds to df's index

# Create a DataFrame for predictions
predictions_df = test_df.copy()
predictions_df['predicted_win'] = y_pred_custom
predictions_df['predicted_win_probability'] = y_proba_win

# Reset index to have row numbers as a column
predictions_df_reset = predictions_df.reset_index()

# Rename 'index' column to 'row_number' for clarity
predictions_df_reset.rename(columns={'index': 'row_number'}, inplace=True)

# Select relevant columns to display
# Assuming you want to see the row number, features, and predictions
# You can adjust which columns to display as needed
columns_to_display = ['row_number', 'predicted_win', 'predicted_win_probability']

# Display the first few predictions with row numbers
print("\nPredictions with Row Numbers:")
print(predictions_df_reset[columns_to_display].head(20))  # Display first 20 predictions

# ------------------------------------------------------------------------------
# 10. (Optional) Save Predictions to a CSV File
# ------------------------------------------------------------------------------
# If you want to save the predictions along with test data identifiers, do the following:
# Ensure your test_df has an identifier column (e.g., 'match_id', 'index', etc.)
# Here, we'll use the original DataFrame's index

# Save the predictions DataFrame to CSV
output_file_path = '/Users/willjones/Desktop/DataScience/predictions_with_row_numbers.csv'  # Replace with desired path
predictions_df_reset.to_csv(output_file_path, index=False)  # Set index=False to avoid duplicating the index
print(f"\nPredictions with row numbers saved to {output_file_path}")

# ------------------------------------------------------------------------------
# 11. (Optional) Visualize Predicted Probabilities
# ------------------------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.hist(y_proba_win, bins=20, edgecolor='k', alpha=0.7)
plt.axvline(custom_threshold, color='r', linestyle='dashed', linewidth=1, label=f'Threshold = {custom_threshold}')
plt.title("Distribution of Predicted Probabilities for 'Win'")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# End of Script
# ------------------------------------------------------------------------------
