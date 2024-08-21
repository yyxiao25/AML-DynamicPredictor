import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
data = pd.read_excel('C:/Users/xyy/Desktop/nodata4ASTALT/AML_allnew_merged_patient_data_processed.xlsx')

# List of features to include in the correlation matrix
features_to_include = [
    'Pre_treatment_Mean_PLT', 'Pre_treatment_Mean_HGB', 'Pre_treatment_Mean_WBC', 'Pre_treatment_Mean_NEUT',
    'Admission_Value_PLT', 'Admission_Value_HGB', 'Admission_Value_WBC', 'Admission_Value_NEUT',
    'CHE', 'B', 'LDH1', 'ALB', 'CRP', 'SOD',
    'FIB-C', 'PA', 'DBIL', 'TP', 'C1q', 'CYSC',
    'BUN', 'SAA', 'SA', 'UBIL','A/G'
]

# Dictionary for renaming features
feature_names = {
    'Pre_treatment_Mean_PLT': 'Pre-treatment mean value of PLT',
    'Pre_treatment_Mean_HGB': 'Pre-treatment mean value of HGB',
    'Pre_treatment_Mean_WBC': 'Pre-treatment mean value of WBC',
    'Pre_treatment_Mean_NEUT': 'Pre-treatment mean value of NEUT',
    'Admission_Value_PLT': 'Admission value of PLT',
    'Admission_Value_HGB': 'Admission value of HGB',
    'Admission_Value_WBC': 'Admission value of WBC',
    'Admission_Value_NEUT': 'Admission value of NEUT',
    'CHE': 'CHE', 'B': 'B', 'LDH1': 'LDH1', 'ALB': 'ALB', 'CRP': 'CRP', 'SOD': 'SOD',
    'FIB-C': 'FIB-C', 'PA': 'PA', 'DBIL': 'DBIL', 'TP': 'TP', 'C1q': 'C1q', 'CYSC': 'CYSC',
    'BUN': 'BUN', 'SAA': 'SAA', 'SA': 'SA', 'UBIL': 'UBIL','A/G':'A/G'
}

# Calculate the correlation matrix for the selected features
correlation_matrix = data[features_to_include].corr()

# Rename the rows and columns of the correlation matrix
correlation_matrix.rename(columns=feature_names, index=feature_names, inplace=True)

# Plot the correlation matrix
plt.figure(figsize=(16, 16))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Specified Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()  # Adjust the layout to fit the figure size
plt.show()

# # Save the figure
# plt.savefig('D:/OneDrive/ModelParameterClassification/Results/Model_parameters_and_clinical_features_correlation_matrix.png')
# plt.close()  # Close the figure after saving to avoid displaying it in the notebook
