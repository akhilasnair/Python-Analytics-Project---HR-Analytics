#import
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


#Read data from the file
data = pd.read_csv("C:/Users/akhil/Downloads/HR_comma_sep.csv",index_col=None)

#Rename some columns to meaningful names
data=data.rename(columns={"number_project":"ProjectCount","average_montly_hours":"MonthlyHours","time_spend_company":"Experience","left":"Turnover","sales":"Department"})


#preview of data
print(data.head())

#check for missing data
print(data.isnull().any())

#rearranging the order.Moving the Turnover column towards the front
turnover = data['Turnover']
data.drop(labels=['Turnover'],axis=1,inplace=True)
data.insert(0,'Turnover',turnover)
print(data.head())

#Describe the functions
print(data.describe(include='all'))

#check data types
print(data.dtypes)

#turnover ratio
print(data.Turnover.value_counts())

#Datasummary based on turnover(mean and variance)
summary_data=data.groupby(data.Turnover)
print(summary_data.mean())
print(summary_data.var())

#The correlation between various
print(data.corr())

#visualization of the correlation
print(plt.matshow(data.corr()))

 #histograms
print(data.hist())

#hypothesis testing
test_val = data['satisfaction_level'][data['Turnover'] == 0].mean()
print(stats.ttest_1samp(a=  data[data['Turnover']==1]['satisfaction_level'],popmean = test_val))

#Splitting data into train and test data
train_data, test_data = train_test_split(data, test_size=0.25)
print(len(test_data))
print(len(train_data))
print(len(data))

#feature selection
X = ["satisfaction_level","last_evaluation","ProjectCount","MonthlyHours","Experience","Work_accident","promotion_last_5years"]
Y = ["Turnover"]
feature_test = SelectKBest(score_func=chi2, k=3)
fit = feature_test.fit(data[X],data[Y])
np.set_printoptions(precision=5)
print(fit.scores_)
#Feature selection using extratreesclassifier
feature_test = ExtraTreesClassifier()
y=data[Y]
feature_test.fit(data[X], y.values.ravel())
print(feature_test.feature_importances_)
#we are choosing the top 4 variables which are satisfaction_level,ProjectCount,last_evaluation and Experience
#Modelling Logistic regression
Xrel=["satisfaction_level","last_evaluation","ProjectCount","Experience"]
logit_model=sm.Logit(data[Y],data[Xrel])
result=logit_model.fit()
print(result.summary()) #all values are relevant

Xrel_train, Xrel_test, y_train, y_test = train_test_split(data[Xrel], data[Y], test_size=0.25, random_state=0)
logicmodel = LogisticRegression()
logicmodel.fit(Xrel_train,y_train.values.ravel())
#prediction
pred = logicmodel.predict(Xrel_test)
print(logicmodel.score(Xrel_test,y_test)) #we get accuracy 76%

#verify the accuracy of logistic model with 10 fold validation
kfold = model_selection.KFold(n_splits=10)
modelLog = LogisticRegression()
scoring = 'accuracy'
result = model_selection.cross_val_score(modelLog, Xrel_train, y_train.values.ravel(), cv=kfold, scoring=scoring)
print(result.mean()) #it gives 76% prediction. This proves our model is not over trained. It generalize well

#confusion matrix
matrix = confusion_matrix(y_test,pred)
print(matrix)

#roc curve
roc_logistic = roc_auc_score(y_test, logicmodel.predict(Xrel_test))
logic_fpr, logic_tpr, logic_thresholds = roc_curve(y_test, logicmodel.predict_proba(Xrel_test)[:,1])
plt.figure()
plt.plot(logic_fpr, logic_tpr, label='Logistic Regression (area = %0.2f)' % roc_logistic)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()