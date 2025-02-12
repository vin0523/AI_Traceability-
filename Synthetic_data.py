import pandas as pd
import random

# Function to generate synthetic data with a specified number of records
def generate_synthetic_data(num_tests=1000):
    data = {
        'Requirement ID': [f"REQ-{i % 20 + 1}" for i in range(num_tests)],  # Repeating requirements
        'Test Case ID': [f"TC-{i + 1}" for i in range(num_tests)],  # Sequential Test Case IDs
        'Previous Execution Status': [random.choice(['Pass', 'Fail']) for _ in range(num_tests)],
        'Test Execution Count': [random.randint(1, 10) for _ in range(num_tests)],  # Number of executions between 1 and 10
    }
    
    df = pd.DataFrame(data)
    
    # Simulate the current Test Execution Status based on previous status and execution count
    # Rule: If it has failed multiple times, it’s likely to fail again
    df['Test Execution Status'] = df.apply(lambda row: 'Fail' if row['Previous Execution Status'] == 'Fail' and row['Test Execution Count'] > 3 else 'Pass', axis=1)
    
    # Randomly set some "Not Executed" values for diversity
    df.loc[df.sample(frac=0.1).index, 'Test Execution Status'] = 'Not Executed'
    
    return df

# Generate synthetic dataset with 100 rows
df_synthetic = generate_synthetic_data(1000)

# Display the first few rows of the synthetic dataset
print(df_synthetic.head())

# Save the synthetic dataset to a CSV file
file_path = "/Users/vinaybommanahalliumesha/Desktop/Synthetic_Dataset.csv"  # Adjust the path as needed
df_synthetic.to_csv(file_path, index=False)

print(f"Dataset saved to {file_path}")
