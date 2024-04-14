
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define paths to CSV files
train_file_path = os.path.join(current_directory, 'data', 'train.csv')
test_file_path = os.path.join(current_directory, 'data', 'test.csv')

# Load train and test data
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Combine train and test data
combined_data = pd.concat([train_data, test_data], axis=0)

# Feature Engineering
def preprocess_assets_liabilities(df):
    df['Total Assets'] = df['Total Assets'].str.replace(' Crore\+', 'e7', regex=True).str.replace(' Lac\+', 'e5', regex=True).str.replace(' Thou\+', 'e3', regex=True).str.replace(' Hund\+', 'e2', regex=True).astype(float)
    df['Liabilities'] = df['Liabilities'].str.replace(' Crore\+', 'e7', regex=True).str.replace(' Lac\+', 'e5', regex=True).str.replace(' Thou\+', 'e3', regex=True).str.replace(' Hund\+', 'e2', regex=True).astype(float)
    return df

combined_data = preprocess_assets_liabilities(combined_data)

# Impute missing or zero values
imputer = SimpleImputer(strategy='median')
combined_data[['Total Assets', 'Liabilities']] = imputer.fit_transform(combined_data[['Total Assets', 'Liabilities']])

# Feature Engineering (create 'Total_Liabilities' feature)
# combined_data['Total_Liabilities'] = combined_data['Total Assets'] - combined_data['Liabilities']

# Preprocessing
combined_data.drop(columns=['ID', 'Candidate'], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
combined_data['Constituency ∇'] = label_encoder.fit_transform(combined_data['Constituency ∇'])
combined_data['Party'] = label_encoder.fit_transform(combined_data['Party'])
combined_data['state'] = label_encoder.fit_transform(combined_data['state'])

# Handle 'Criminal Case' column
combined_data['Criminal Case'] = pd.to_numeric(combined_data['Criminal Case'], errors='coerce').fillna(0)

# Split data into train and test
train_data = combined_data[:len(train_data)]
test_data = combined_data[len(train_data):]

# Split train data into features and target variable
X_train = train_data.drop(columns=['Education'])
y_train = train_data['Education']

# Define the RandomForestClassifier model
rf_classifier = RandomForestClassifier(random_state=42)

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters:")
print(grid_search.best_params_)

# Make predictions using the best model
best_rf_model = grid_search.best_estimator_
predictions = best_rf_model.predict(test_data.drop(columns=['Education']))

# Save the predictions to a CSV file
submission_df = pd.DataFrame({'ID': test_data.index, 'Education': predictions})
submission_file_path = os.path.join(current_directory, 'data', 'submission.csv')
submission_df.to_csv(submission_file_path, index=False)




# PLOTTING GRAPHS



# Function to preprocess wealth data and convert it into numerical format
def preprocess_wealth(df):
    # Convert wealth strings to numerical format
    df['Total Assets'] = df['Total Assets'].str.replace(' Crore\+', 'e7', regex=True).str.replace(' Lac\+', 'e5', regex=True).str.replace(' Thou\+', 'e3', regex=True).str.replace(' Hund\+', 'e2', regex=True).astype(float)
    return df

# Load initial train data
initial_train_data = pd.read_csv(train_file_path)
# Load initial test data
initial_test_data = pd.read_csv(test_file_path)

# Preprocess the wealth data
test_data_education = preprocess_wealth(initial_test_data)
initial_train_data = preprocess_wealth(initial_train_data)

# Load submission file to get predicted education values
submission_df = pd.read_csv(submission_file_path)

# Add education column to initial test data and fill it with predicted values
test_data_education['Education'] = submission_df['Education']

# Drop 'ID' and 'Candidate' columns from both train and test data
test_data_education.drop(columns=['ID', 'Candidate'], inplace=True)
initial_train_data.drop(columns=['ID', 'Candidate'], inplace=True)

# Combine train and test data
combined_data = pd.concat([initial_train_data, test_data_education], axis=0)

# Save the combined dataset to a new CSV file
combined_data_file_path = os.path.join(current_directory, 'data', 'combined_data.csv')
combined_data.to_csv(combined_data_file_path, index=False)



# GRAPH-1: Percentage of Candidates with High Criminal Records by Party

# Set the threshold for high criminal record
criminal_record_threshold = 5  # Change this threshold as needed

# Calculate the percentage of candidates for each party who have high criminal records
party_criminal_record_percentage = (combined_data[combined_data['Criminal Case'] >= criminal_record_threshold]
                                    .groupby('Party')['Criminal Case']
                                    .count() / combined_data.groupby('Party')['Criminal Case'].count()) * 100

# Plotting
plt.figure(figsize=(10, 6))
party_criminal_record_percentage.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title(f'Percentage of Candidates with High Criminal Records by Party (Threshold={criminal_record_threshold} Criminal Cases)')
plt.xlabel('Party')
plt.ylabel('Percentage')
plt.show()



# GRAPH-2: Percentage of Candidates with High Wealth by Party 

# Set the threshold for high wealth
wealth_threshold = 5e7  # Change this threshold as needed (e.g., 10 million)

# Calculate the percentage of candidates for each party who have high wealth
party_wealth_percentage = (combined_data[combined_data['Total Assets'] >= wealth_threshold]
                           .groupby('Party')['Total Assets']
                           .count() / combined_data.groupby('Party')['Total Assets'].count()) * 100

# Plotting
plt.figure(figsize=(10, 6))
party_wealth_percentage.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title(f'Percentage of Candidates with High Wealth by Party (Threshold={wealth_threshold/1e7} Crore+)')
plt.xlabel('Party')
plt.ylabel('Percentage')
plt.show()

