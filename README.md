# Python-Analytics-Project---HR-Analytics
Python Analytics project- HR Analytics to predict the attrition rate of employees based on various factors

This project focus on predicting the attrition rate based various factors.
Data was collected from Kaggle.com - https://www.kaggle.com/ludobenistant/hr-analytics/data
It has 10 columns and 14999 rows. The target feature is Turnover(left). The other features are:
=> satisfaction_level
=> last_evaluation
=> number_project
=> average_monthly_hours
=> time_spend_company
=> work_accident
=> promotion_last_5years
=> sales
=> salary

The modules required for the project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import model_selection

Data Pre-processing

Since the data did not have any missing value, just renamed the columns with meaningful names.

Data Analysis

The project gives the basic analysis of the data. Statistical analysis is done based on the target variable-Turnover. Also Correlation among various variable are found
and Histograms are plotted

Hypothesis Testing

Assumption is made that the Turnover ratio is same for people with good satisfaction rate and bas satisfaction rate.
TTest is conducted and it proves that the null hypothesis is wrong. People with good satisfaction are less likely to leave

Feature Selection

Feature selection with SelectKbest and ExtraTreesClassifier are done. The features that have more effect on turnover are
satisfaction_level,ProjectCount,last_evaluation and Experience

Logistic model

Data is split into train and test data on the ratio 75%-25%. Logistic model is fed with the train data and predictions are made on the test data. This model gives an accuracy of 76%
10 fold cross validation is also preformed which also gives an accuracy of 76%

Naive Bayes
Naive Bayes gave only 71% accuracy

Decision Tree
Decision Tree gave 89%

We can conclude that Decision Tree is the best alogorithm among three