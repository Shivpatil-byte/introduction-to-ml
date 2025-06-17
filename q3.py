

import pandas as pd
import numpy as np

# Step 1: Create a DataFrame with random scores
np.random.seed(42)  # For reproducibility
num_students = 10  # Number of students
data = {'Student': [f'Student{i+1}' for i in range(num_students)],
        'Score': np.random.randint(50, 101, size=num_students)}  # Random scores between 50 and 100
df = pd.DataFrame(data)

# Step 2: Define a function to assign grades based on score
def assign_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

# Step 3: Apply the function to assign grades
df['Grade'] = df['Score'].apply(assign_grade)

# Step 4: Print the DataFrame
print(df)
def filter_grades(df, grades_to_keep=['A', 'B']):
    
    if 'Grade' not in df.columns:
        raise ValueError("The DataFrame must have a 'Grade' column.")
    
    return df[df['Grade'].isin(grades_to_keep)]
print(filter_grades(df, grades_to_keep=['A', 'B']))