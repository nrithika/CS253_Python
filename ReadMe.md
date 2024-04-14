# Education predictions

## Introduction:
In this assignment, we aim to predict the education level of winners in state and union territory (UT) elections across India using machine learning techniques. The dataset contains information on various features of election winners, sourced from the Election Commission of India website. We will perform multi-class classification to predict the education level of the winners.

## Dataset:
The dataset consists of a training dataset and a test dataset. Each row represents a winner in a state or UT election, with features including constituency, party, state, total assets, liabilities, criminal cases, and education level. The given test data set doesn't include education and our task is to predict the education levels of candidates in test data set by training a machine learning model using training set.

## Task
1. **Objective**: Predict the education level of election winners.
2. **Classification Models**: Utilize machine learning models such as SVM, KNN, DecisionTree, RandomForest, etc., for multi-class classification and then improvise the model to get a better F1 score.

## Models and Hyperparameters
We will train a RandomForestClassifier model for this task. The hyperparameters used for the model training are as follows:

- `n_estimators`: Number of trees in the forest (100, 200)
- `max_depth`: Maximum depth of the trees (None, 10)
- `min_samples_split`: Minimum number of samples required to split an internal node (2, 5)
- `min_samples_leaf`: Minimum number of samples required to be at a leaf node (1, 2)

## Data Analysis Plots
1. **Percentage Distribution of Parties with Candidates Having the Most Criminal Records**: This plot shows the percentage distribution of parties with candidates having a high number of criminal records. The threshold for a high criminal record is set to 5 cases.
2. **Percentage Distribution of Parties with the Most Wealthy Candidates**: This plot displays the percentage distribution of parties with candidates having high wealth. The threshold for high wealth is set to 50 million rupees (5e7).

## Libraries Used
The following libraries are used for data preprocessing, model training, and analysis:
- pandas
- scikit-learn
- matplotlib

## Project Structure:

The project consists of a single file:

```CS253_Python/
│
├── data/
│   ├── train.csv          # CSV file containing training data
│   ├── test.csv           # CSV file containing test data
│   ├── combined_data.csv  # CSV file containing combined data of both train and test data
│   └── submission.csv     # CSV file containing submission data
├── graphs/
│   ├── graph1.png         # Graph of Percentage of candidates with high criminal records
│   ├── graph2.png         # Graph of Percentage of candidates with high wealth
├── prediction.py          # Main source file for the program
└── README.md              # README file containing project information

```
# Instructions

1. **Clone the Repository**: Open your terminal or command prompt and navigate to the directory where you want to clone the repository. Then, execute the following command to clone the repository:
```
https://github.com/nrithika/CS253_Python.git
```

2. **Navigate to the Directory**: Change your current directory to the cloned `CS253_Python` directory
```
cd CS253_Python
```

3. **Install Required Libraries**: Ensure you have Python installed on your system. You'll need to install the following libraries if they are not already installed:
- pandas
- scikit-learn
- matplotlib

You can install these libraries using pip:
```
pip install pandas scikit-learn matplotlib
```

4. **Run the Script**: Execute the Python script `prediction.py` to train the model and generate the data analysis plots:
```
python prediction.py
```
