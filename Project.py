import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/vinaybommanahalliumesha/Desktop/Project Tracebaility/Dummy-Data.csv"  
df = pd.read_csv(file_path, encoding='ISO-8859-1')  

print("Dataset loaded successfully!")
print(df.head())

# Step 3: Create the Traceability Matrix with Execution Status
traceability_matrix = df.pivot_table(
    index='Requirement ID',  
    columns='Test Case ID',  
    values='Test Execution Status',  # Use execution status instead of description
    aggfunc=lambda x: ', '.join(x)  
)

# Fill missing values with "Not Executed"
traceability_matrix = traceability_matrix.fillna("Not Executed")

print("\nUpdated Traceability Matrix with Execution Status:")
print(traceability_matrix)

# Save to Excel
output_file = "traceability_matrix.xlsx"  
traceability_matrix.to_excel(output_file)
print(f"\nTraceability Matrix has been saved to {output_file}")

# Step 4: Compute Test Coverage Per Requirement
df['Passed'] = df['Test Execution Status'].apply(lambda x: 1 if x == 'Pass' else 0)
df['Total'] = 1  

coverage = df.groupby('Requirement ID').agg({'Passed': 'sum', 'Total': 'count'})
coverage['Coverage (%)'] = (coverage['Passed'] / coverage['Total']) * 100

print("\nTest Coverage Report:")
print(coverage)

# Save Coverage Report
coverage_output_file = "test_coverage_report.xlsx"
coverage.to_excel(coverage_output_file)
print(f"\nTest Coverage Report has been saved to {coverage_output_file}")

# Step 5: Visualize Test Coverage as a Bar Chart
plt.figure(figsize=(10, 5))
sns.barplot(x=coverage.index, y=coverage["Coverage (%)"], palette="Blues_d")

plt.xlabel("Requirement ID")
plt.ylabel("Test Coverage (%)")
plt.title("Test Coverage per Requirement")
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.show()
