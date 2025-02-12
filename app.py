import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load model function (you should replace this with actual AI model)
def train_model(df):
    X = df[['Test Execution Count']]  # Example feature
    y = df['Test Execution Status'].apply(lambda x: 1 if x == 'Fail' else 0)  # Binary classification (0=Pass, 1=Fail)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Predict on the entire dataset (not just the test set)
    predictions = model.predict(X)  # Predicting on the entire dataset
    
    return model, predictions, y, X  # Return y (target), predictions, and the full feature set

# Function to display traceability matrix
def generate_traceability_matrix(df):
    traceability_matrix = df.pivot_table(index='Requirement ID', columns='Test Case ID', values='Test Execution Status', aggfunc='first', fill_value='Not Executed')
    return traceability_matrix

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

    # Generate Traceability Matrix button
    if st.button("Generate Traceability Matrix"):
        traceability_matrix = generate_traceability_matrix(df)
        st.write("Traceability Matrix:")
        st.dataframe(traceability_matrix)  # Display matrix as a table
        
        # Optional: Create a heatmap for better visualization of the matrix
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(traceability_matrix == 'Pass', annot=True, cmap="YlGnBu", cbar=False, ax=ax)
        st.pyplot(fig)

    # AI Prediction button
    if st.button("Run AI Prediction"):
        st.write("Running AI-powered test failure prediction...")
        model, predictions, y, X = train_model(df)
        
        # Display the classification report
        st.write("Prediction Results:")
        accuracy = np.mean(predictions == y)
        st.write(f"Accuracy: {accuracy:.2f}")
        
        # Display classification report
        report = classification_report(y, predictions, target_names=['Pass', 'Fail'], output_dict=True)
        st.write(report)

        # Prediction insights
        prediction_insights = pd.DataFrame({
            'Test Execution Count': df['Test Execution Count'],
            'Predicted Status': ['Fail' if pred == 1 else 'Pass' for pred in predictions]
        })
        st.write("Prediction Insights:")
        st.dataframe(prediction_insights)

        # Visualizing prediction results as a bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        prediction_insights['Predicted Status'].value_counts().plot(kind='bar', ax=ax, color=["green", "red"])
        ax.set_title("Prediction Results (Pass vs Fail)")
        st.pyplot(fig)
