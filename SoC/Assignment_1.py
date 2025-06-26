import numpy as np
import pandas as pd
from sklearn import linear_model
df=pd.read_csv('/content/Salary_dataset.csv')
df.head()
model=linear_model.LinearRegression()
model.fit(df[['YearsExperience']],df.Salary)
model.score(df[['YearsExperience']],df.Salary)
model.predict([[1.3]])
