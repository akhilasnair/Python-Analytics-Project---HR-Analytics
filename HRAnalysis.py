#import
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import model_selection
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

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

dummies = ['Department', 'salary']
for value in dummies:
    dummy_list = 'value' + '_' + value
    dummy_list = pd.get_dummies(data[value], prefix = value)
    datas = data.join(dummy_list)
    data = datas
data.drop(labels=['Department','salary'],axis=1,inplace=True)
data.drop(labels=['Department_RandD','salary_high'],axis=1,inplace=True)
cols = data.columns.values.tolist()
print(cols)
    
#feature selection
Y = ["Turnover"]
X = [i for i in cols if i not in Y]
print(X)
print(Y)
y=data[Y]
#feature selection using RFE
model = LogisticRegression()
rfe_checking = RFE(model,10)
fit = rfe_checking.fit(data[X],y.values.ravel())
print(fit.support_)
print(fit.ranking_)
#we are choosing the top 10 variables
#Modelling Logistic regression
Xrel=['satisfaction_level','Work_accident','promotion_last_5years','Department_accounting','Department_hr','Department_marketing','Department_support', 'Department_technical', 'salary_low', 'salary_medium']
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

#Naive Bayes
model1 = GaussianNB()
model1.fit(Xrel_train,y_train.values.ravel())
predicted = model1.predict(Xrel_test)
print("Naive Bayes {}".format(model1.score(Xrel_test,y_test)))
#Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xrel_train,y_train.values.ravel())
predit = clf.predict(Xrel_test)
print("decision tree {}".format(clf.score(Xrel_test,y_test)))

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

roc_naivebayes = roc_auc_score(y_test, model1.predict(Xrel_test))
fpr, tpr, thresholds = roc_curve(y_test, model1.predict_proba(Xrel_test)[:,1])
plt.figure()
plt.plot(logic_fpr, logic_tpr, label='Naive Bayes (area = %0.2f)' % roc_naivebayes)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

roc_decisiontree = roc_auc_score(y_test, clf.predict(Xrel_test))
logic_fpr, logic_tpr, logic_thresholds = roc_curve(y_test, clf.predict_proba(Xrel_test)[:,1])
plt.figure()
plt.plot(logic_fpr, logic_tpr, label='Decision Tree (area = %0.2f)' % roc_decisiontree)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()