# Learning Data Science - Data Science positions salary prediction

- Created a salary prediction tool for Data Science positions to help data scientists negotiate better incomes while seeking employment
- Scraped ~900 job descriptions from Glassdoor website using selenium and python
- Engineered features by extracting important text based parameters from Jo descriptions and position listings
- Used GridSearchCV to optimize Linear, Lasso and Random forest models
- Implemented a basic API for client side implementation using flask

## References and resources

**Project guide for learning resources:** KenJee's Youtube channel: https://www.youtube.com/watch?v=nUOh_lDMHOU&t=1526s
**Python version:** 3.8
**Packages:** pandas, numpy, matplotlib, selenium, flask, pickle, json
**Scraper github (forked)**: https://github.com/arapfaik/scraping-glassdoor-selenium
**Scraper Guide:** https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905
**Flask implementation:** https://towardsdatascience.com/productionizing-your-machine-learning-model-221468b0726d


## Web scraping

Tweaked the forked github repo to account for Glassdoor API changes and processed ~900 Glassdoor listings. Following fields were extracted

- Job Title
- Salary estimate
- Job description
- Company rating
- Company location
- Company HQ location
- Company size
- Company founded date
- Company Ownership type
- Industry and Sector
- Revenue

## Data Cleaning

Post scraping the data, various fields were cleaned up to make them more useable for the model. Below is a summary of processing done:

- Removed entries with no salary available
- Extracted numeric data out of salary field (Min and Max of salary offered)
- Assigned flag to identify if estimate was employer provided/glassdoor estimate
- Identified entires with hourly wages and converted it to yearly based on ~40 hrs/week and 50 weeks per year
- Parse company ratings from text
- Calculated company age
- Identified keywords in job description and created features for job requirements:
  - Python
  - R
  - Excel
  - AWS
  - Spark
  - SQL 
- Simplified Job titles and seniority

## EDA

Created basic distributions for data based on categorical variables. TODO: Add charts

## Model Building

Categorical variables were converted into dummy variables and dataset was split into training and test groups (20% split).
Three models were chosen for testing and Mean absolute error metric was used for evaluation to ensure ease of understanding.

Models used:
1. OLS Multiple Linear regression: Baseline model
2. Lasso Regression: Since we have sparse data and significantly higher number of categorical variables, lasso may provide better results due to inherent normalization of variables
3. Random forest regression: With sparse data and high # of categorical variables

## Model Performance

Random forest clearly outperformed other models. Below are average cross validation MAE scores:
- **Random forest:** MAE - -15.8
- **Lasso Regression:** MAE - -20.1
- **OLS Multiple linear regression:** MAE - -24.2
