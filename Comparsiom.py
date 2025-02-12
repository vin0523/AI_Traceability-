import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load model function (you should replace this with actual AI model)
def train_model(df):
    X = df[['Test Execution Count']]  # Example feature
    y = df['Test Execution Status'].apply(lambda x: 1 if x == 'Fail' else 0)  # Binary classification (0=Pass, 1=Fail)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    return model, predictions, y_test, X_test

# Streamlit UI elements
st.title("Test Failure Prediction and Traceability Matrix")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    # Read the uploaded dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")

    # Show the first few rows of the dataset
    st.write(df.head())

    # Check for missing values
    st.write("Checking for missing values in the dataset:")
    st.write(df.isnull().sum())

    # Run AI Prediction
    if st.button("Run AI Prediction"):
        st.write("Running AI-powered test failure prediction...")
        model, predictions, y_test, X_test = train_model(df)
        
        # Display the classification report
        st.write("Prediction Results:")
        st.write(f"Accuracy: {np.mean(predictions == y_test)}")
        
        # Display classification report
        report = classification_report(y_test, predictions, target_names=['Pass', 'Fail'])
        st.write(report)

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)

        # Visualize the confusion matrix as a heatmap
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pass', 'Fail'], yticklabels=['Pass', 'Fail'])
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # Create a dataframe to compare predictions with actual test status
        prediction_comparison = pd.DataFrame({
            'Actual': ['Fail' if label == 1 else 'Pass' for label in y_test],
            'Predicted': ['Fail' if pred == 1 else 'Pass' for pred in predictions]
        })

        # Display the comparison in a table
        st.write("Prediction vs Actual Test Status Comparison")
        st.dataframe(prediction_comparison)

        # Visualize with a bar plot
        st.write("Visualizing Prediction Results (Pass vs Fail)")
        fig, ax = plt.subplots(figsize=(8, 5))
        prediction_comparison['Predicted'].value_counts().plot(kind='bar', color=["red", "green"], alpha=0.7, ax=ax)
        ax.set_title('Predicted Test Failures vs Passes')
        ax.set_xlabel('Test Status')
        ax.set_ylabel('Count')
        ax.set_xticklabels(['Pass', 'Fail'], rotation=0)
        st.pyplot(fig)

        # Optionally, visualize the misclassified predictions
        misclassified = prediction_comparison[prediction_comparison['Actual'] != prediction_comparison['Predicted']]

        st.write("Misclassified Predictions:")
        st.dataframe(misclassified)

        # Option to download prediction insights as CSV
        if st.button("Download Prediction Insights"):
            prediction_comparison.to_csv('prediction_insights.csv', index=False)
            st.write("Download your results: [prediction_insights.csv](./prediction_insights.csv)")

    # Option to generate Traceability Matrix (optional)
    if st.button("Generate Traceability Matrix"):
        traceability_matrix = df.pivot_table(index='Requirement ID', columns='Test Case ID', values='Test Execution Status', aggfunc='first', fill_value='Not Executed')
        st.write("Traceability Matrix:")
        st.dataframe(traceability_matrix)  # Display matrix as a table

        # Optional: Create a heatmap for better visualization of the matrix
        st.write("Visualizing Traceability Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(traceability_matrix == 'Pass', annot=True, cmap="YlGnBu", cbar=False, ax=ax)
        st.pyplot(fig)
