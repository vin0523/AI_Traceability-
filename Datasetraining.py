import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
file_path = "/Users/vinaybommanahalliumesha/Desktop/Synthetic_Dataset.csv"
df = pd.read_csv(file_path)

# Step 2: Check for missing values
print("Checking for missing values in the dataset:")
print(df.isnull().sum())

# Step 3: Encode the target variable ('Test Execution Status') into numeric values
label_encoder = LabelEncoder()

# Encode the target column (Test Execution Status)
df['Test Execution Status'] = label_encoder.fit_transform(df['Test Execution Status'])

# Encode the 'Previous Execution Status' feature, but don't inverse_transform later
previous_execution_encoder = LabelEncoder()
df['Previous Execution Status'] = previous_execution_encoder.fit_transform(df['Previous Execution Status'])

# Step 4: Define features and target
features = ['Previous Execution Status', 'Test Execution Count']  # Example features
target = 'Test Execution Status'  # The column to predict (Pass/Fail)

X = df[features]  # Features
y = df[target]    # Target

# Step 5: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the sizes of training and testing sets
print(f"Training data size: {len(X_train)}")
print(f"Testing data size: {len(X_test)}")

# Step 6: Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Collect prediction insights
prediction_insights = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Prediction Correct': y_test == y_pred
})

# Inverse transform the predictions and actual labels back to their original values
prediction_insights['Actual'] = label_encoder.inverse_transform(prediction_insights['Actual'])
prediction_insights['Predicted'] = label_encoder.inverse_transform(prediction_insights['Predicted'])

# Step 10: Save prediction insights to CSV
prediction_insights.to_csv('prediction_insights.csv', index=False)
print("\nPrediction insights saved to 'prediction_insights.csv'.")
