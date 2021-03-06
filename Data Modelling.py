# -*- coding: utf-8 -*-
"""
Modelling the cleaned jobs data

by Susheel Patel

"""
###########################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



###########################################################################################

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
###########################################################################################
df.columns

df_model = df[['Rating','Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 
       'employer provided flag', 'job_state',
       'hq_state', 'job_in_HQ_flag', 'age_company', 'sas_flag', 'spark_flag',
       'python_flag', 'matlab_flag', 'tensorflow_flag', 'tableau_flag',
       'aws_flag', 'hadoop_flag', 'r_flag', 'job_simplified', 'seniority',
       'num_competitors','job_desc_len', 'avg_salary']]

df_dum = pd.get_dummies(df_model)

X = df_dum.drop('avg_salary',axis = 1)
y = df_dum['avg_salary']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)


# X_train.iloc[1,:].values.reshape(1,-1)
###########################################################################################

# Stats models OLS regression. Since it doesnt create a constant by default, we add it in

X_train_stats = sm.add_constant(X_train)

model = sm.OLS(y_train,X_train_stats.astype(float))

model.fit().summary()

###########################################################################################
# RUn sklearn models

###########################################################################################
# Function for Cross validation of model


def cross_validate(model, cv, scoring_func):
    val_score = cross_val_score(estimator = model, X = X_train, y = y_train, cv=cv, scoring=scoring_func)
    
    # Printing pretty output
    
    print("Negative MAE: {:.2f}".format(val_score.mean()))
    print("Std deviation: {:.2f}".format(val_score.std()))
    print("Min: {:.2f}".format(val_score.min()))
    print("Max: {:.2f}".format(val_score.max()))

###########################################################################################
# Run OLS on SKlearn library
# No need to add constant value as SKlearn handles it

lm = LinearRegression()

cross_validate(lm, 3, 'neg_mean_absolute_error')

lm.fit(X_train,y_train)

###########################################################################################
# Run Lasso Model and plot error curve to optimise the alpha value based on CV score

error_array = []
alpha_arr = []

for i in range(1,100,1):
    j =  i/100
    val_score = cross_val_score(estimator = Lasso(alpha = j), X = X_train, y = y_train, cv=3, scoring='neg_mean_absolute_error')
    error_array.append(val_score.mean())
    alpha_arr.append(j)

plt.plot(alpha_arr, error_array)  

# Put error list in a data frame for identifying optimal alpha
error_tuple = tuple(zip(alpha_arr,error_array))

lasso_error_df = pd.DataFrame(error_tuple, columns = ["lasso_alpha", "neg_MAE"])

best_alpha = lasso_error_df.loc[lasso_error_df['neg_MAE'] == max(lasso_error_df['neg_MAE']),'lasso_alpha']

print("Best alpha for lasso is %f" %best_alpha)

lm_lasso = Lasso(alpha = 0.09)

lm_lasso.fit(X_train, y_train)

###########################################################################################

# Random forest regressor
    
cross_validate(RandomForestRegressor(), 3, 'neg_mean_absolute_error')

# Since Random forest seems to be providing the best results, we will run Grid search to find optimal parameters

# Setting a parameter dictionary

params = {
    "n_estimators": range(10,100,10),
    "criterion": ["mae"],
    #"min_samples_leaf": np.arange(0.1,0.5,0.1),
    "random_state":[1]
    }

ds_rf = GridSearchCV(estimator = RandomForestRegressor(),param_grid=params, scoring='neg_mean_absolute_error', cv= 3)

ds_rf.fit(X_train,y_train)

print("Best Estimator: ")
print(ds_rf.best_estimator_)

print("Best score: ")
print(ds_rf.best_score_)

###########################################################################################

# Testing ensemble score

test_pred_lm = lm.predict(X_test)
test_pred_lm_lasso = lm_lasso.predict(X_test)
test_pred_rf = ds_rf.predict(X_test)

array_preds = [test_pred_lm,test_pred_lm_lasso,test_pred_rf]

for prediction in array_preds:
    
    print(mean_absolute_error(y_test, prediction))
    
test_pred_ensemble = (test_pred_lm + test_pred_lm_lasso + test_pred_rf)/3
    
mean_absolute_error(y_test, test_pred_ensemble)

###########################################################################################

# Pickle our model so we can use it with flask for web hosting

import pickle

filename = 'final_jobs_rf_model.p'
pickle.dump(ds_rf.best_estimator_, open(filename, 'wb'))