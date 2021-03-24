# -*- coding: utf-8 -*-
"""
Modelling the cleaned jobs data

by Susheel Patel

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.read_pickle("cleaned_jobs_2.pk1")


"""
General steps for modelling

1. Select required variables/features
2. Create dummy data/ encode vategorical variables
3. Split into train and test
4. Models to implement
    Lasso regression
    Multiple linear regression
    Random forest
"""
df.columns

df_model = df[['Rating','Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 
       'employer provided flag', 'job_state',
       'hq_state', 'job_in_HQ_flag', 'age_company', 'sas_flag', 'spark_flag',
       'python_flag', 'matlab_flag', 'tensorflow_flag', 'tableau_flag',
       'aws_flag', 'hadoop_flag', 'r_flag', 'job_simplified', 'seniority',
       'num_competitors', 'min_salary_updated','job_desc_len',
       'max_salary_updated', 'avg_salary']]

df_dum = pd.get_dummies(df_model)

X = df_dum.drop('avg_salary',axis = 1)
y = df_dum['avg_salary']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)