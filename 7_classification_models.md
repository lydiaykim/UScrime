---
title: Classification Models
notebook: 7_classification_models.ipynb
nav_include: 4
---

## Contents
{:.no_toc}
*  
{: toc}


We develop classification models to predict whether the murder rate in each metropolitan area will be above 8.5 (the historical 90th percentile rate) in subsequent years. We choose to develop the binary categorical prediction model like this one, because we believe they will produce results that can be more useful to policy makers than continuous analysis alone. Policy makers in violent areas are interested in understanding the factors that make their area violent, so that they can develop policies that will reduce levels of violence among their constituency. For this reason, we developed a classification algorithm that can predict if an area will be unusually violent, so that the factors that lead to unusual violence can be isolated and addressed by policy makers.



```python
import numpy as np
import pandas as pd
import matplotlib
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
import sklearn.metrics as metrics
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from IPython.display import Image
from IPython.display import display
%matplotlib inline
from sklearn import tree
from sklearn import ensemble
import seaborn.apionly as sns
from sklearn import preprocessing
from sklearn.tree import export_graphviz
pd.options.mode.chained_assignment = None   
```


    /anaconda/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools
    /anaconda/lib/python3.6/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)


## Data set up



```python
df = pd.read_csv('cleaned data/merged_dataset.csv').drop('Unnamed: 0', axis=1)
df.shape
```





    (3893, 303)





```python
#create some vars
def makelog(dframe, varname, varname2, orig_var):
    dframe[varname] = np.log(dframe[orig_var])
    dframe[varname2] = (np.log(dframe[orig_var]))**2
    
    
makelog(df, 'log_pop', 'log_pop2', 'pop')
makelog(df, 'log_mean_inc_white', 'log_mean_inc_white2', 'mean_inc_white')
makelog(df, 'log_mean_inc_black', 'log_mean_inc_black2', 'mean_inc_black')
makelog(df, 'log_mean_inc_hispanic', 'log_mean_inc_hispanic2', 'mean_inc_hispanic')


df['poor_nonwhite']=df['poor']-df['poor_white']
df['LA'] = df['state'] == 'Louisiana' 
df['AK'] = df['state'] == 'Arkansas'
df['MI'] = df['state'] == 'Mississippi'
df['SC'] = df['state'] == 'South Carolina'
df['OK'] = df['state'] == 'Oklahoma'

df.shape
```





    (3893, 317)





```python
#split into training and test datasets 
np.random.seed(1818)
msk = np.random.rand(len(df)) < 0.66
dftrain = df[msk]
dftest = df[~msk]
```




```python
#set the predictors we will use 
df_gun_laws=pd.read_csv('raw data/state-firearms/raw_data.csv')
#gun_vars = list(df_gun_laws)[2:]
gun_vars = ['age21longgunpossess', 'ccbackgroundnics', 'ccrevoke', 'dealerh', 'drugmisdemeanor', 'incidentremoval',
            'lockd', 'loststolen', 'onepermonth', 'opencarryh', 'opencarrypermith', 'permitconcealed', 'personalized',
            'recordsallh', 'stalking']

#census vars
cen_vars1=['pct_hispanic', 'pct_white', 'pct_black', 'pct_indian', 'pct_asian', 'pct_hawaiian', 
           'pct_other', 'pct_mixed', 'pct_foreign_citizen']
cen_vars2 = ['log_pop', 'log_pop2', 'pop5_14', 'pop15_17', 'pop18_24', 'pop15_44', 'pop65up', 
             'median_age', 'sex_ratio',  'age_dep', 'oldage_dep', 'child_dep']
cen_vars3 = ['families_singledad', 'families_singlemom', 'family_size',  'pct_ownhouse']
cen_vars4 = ['pct15up_married', 'pct15up_widowed', 'pct15up_divorced', 'pct15up_separated', 
             'pct15up_nevermar']
cen_vars5 = ['pct_enroll_public', 'pct_enroll_private', 'pct15_17_enroll', 'pct18_19_enroll', 
             'pct18_24_lesshs', 'pct18_24_hs', 'pct18_24_somecol']
cen_vars6 = ['pct25up_less9','pct25up_nohs', 'pct25up_hs', 'pct25up_somecol', 'pct25up_somecol2', 
             'pct25up_col', 'pct25up_grad']
cen_vars7 = ['poor', 'poor_nonwhite', 'fam_cash_assist', 'fam_social_sec', 'log_mean_inc_white', 
             'log_mean_inc_white2','log_mean_inc_black', 'log_mean_inc_black2','income10lo', 'income10_15', 
             'income15_25', 'income25_35', 'income35_50', 'income50_75', 'income75_100', 'income100_150', 
             'income150_200', 'income200up']
cen_vars8 = ['pct18_veterans', 'pct_disabled20_64', 'employed20_64', 'unemployed20_64', 'employed_white', 
             'unemployed_white', 'employed20_64_m', 'unemployed20_64_m', 'employed20_64_f', 'unemployed20_64_f']
cen_vars9 = ['pct16_manuf', 'pct16_info', 'pct16_finance', 'pct16_prof', 'pct16_edhealth', 'LA', 'AK', 'MI', 'SC','OK']
cen_vars = cen_vars1 + cen_vars2 + cen_vars3 +cen_vars4 +cen_vars5+ cen_vars6 +cen_vars7 + cen_vars8 + cen_vars9

#all predictors
predictors = gun_vars + cen_vars

#set outcomes
outcomes=['Total', 'Estimated', 'Rate']

#all vars of interest
var_needed= predictors+outcomes
```




```python
#limit datasets to just subset of vars
dftrain=dftrain[var_needed]
dftest=dftest[var_needed]
print(dftrain.shape, dftest.shape)
```


    (2610, 100) (1283, 100)




```python
#Classification variable: 90 percentile murder rate in the training data
dftrain['Rate'].quantile([.90])
```





    0.9    8.5
    Name: Rate, dtype: float64





```python
#Create a dummy = 1 if rate > cutoff (8.5)
dftrain['High'] = (dftrain['Rate'] >= 8.5).astype(int) 
dftest['High'] = (dftest['Rate'] >= 8.5).astype(int) 
```




```python
dftest['High'].describe()
```





    count    1283.000000
    mean        0.102884
    std         0.303926
    min         0.000000
    25%         0.000000
    50%         0.000000
    75%         0.000000
    max         1.000000
    Name: High, dtype: float64





```python
xtrain = dftrain[dftrain.columns[:-4]]
ytrain = dftrain['High']

xtest = dftest[dftest.columns[:-4]]
ytest = dftest['High']
```


## General Classification Models

We first look at logistic models, kNN, LDA, and QDA to predict whether an MSA will have a high murder rate. Given that only about 10% of the training dataset has a murder rate higher than 8.5, we give class=1 greater weight. 



```python
score = lambda model, x_train, y_train: pd.Series([model.score(x_train, y_train), 
                                                 model.score(x_train[y_train==0], y_train[y_train==0]),
                                                 model.score(x_train[y_train==1], y_train[y_train==1])], 
                                                index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1'])
```




```python
#Unweighted Logistic model
cvals = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 10000, 100000]
Ulm = LogisticRegressionCV(Cs=cvals, cv=10)
Ulm = Ulm.fit(xtrain, ytrain)

Ulm_scores = score(Ulm, xtrain, ytrain)
Ulm_scores_test = score(Ulm, xtest, ytest)
```




```python
#Weighted Logistic model
lm = LogisticRegressionCV(Cs=cvals, cv=10, class_weight='balanced')
lm = lm.fit(xtrain, ytrain)

lm_scores = score(lm, xtrain, ytrain)
lm_scores_test = score(lm, xtest, ytest)
```




```python
#kNN
klist = [2, 5, 10, 20, 40, 50]
xval_scores = []

for k in klist:
    KNN = KNeighborsClassifier(n_neighbors=k)
    xval_scores.append(np.mean(cross_val_score(KNN, xtrain, ytrain, cv=10)))
    
imax = xval_scores.index(np.max(xval_scores))
kval = klist[imax]

#fit model
KNN = KNeighborsClassifier(n_neighbors=kval)
KNN.fit(xtrain, ytrain)

knn_scores = score(KNN, xtrain, ytrain)
knn_scores_test = score(KNN, xtest, ytest)
```




```python
#LDA
LDA = LinearDiscriminantAnalysis()
LDA.fit(xtrain, ytrain)

LDA_scores = score(LDA, xtrain, ytrain)
LDA_scores_test = score(LDA, xtest, ytest)
```




```python
#QDA
QDA = QuadraticDiscriminantAnalysis()
QDA.fit(xtrain, ytrain)

QDA_scores = score(QDA, xtrain, ytrain)
QDA_scores_test = score(QDA, xtest, ytest)
```


    /anaconda/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:695: UserWarning: Variables are collinear
      warnings.warn("Variables are collinear")


## Decision tree & Ensemble methods

We next explore single decision tree, Random Forest, gradient boosting and SVC. For the random forest model, we use cross validation to determine the optimal number of trees and the optimal number of predictors for splitting. For the gradient boosting method, we also use cross validation to choose optimal number of trees and and tree depth for the base learner. 



```python
#Single Decision Tree
xval_scores = []
depths = list(range(2, 11))

for d in depths:
    dt = DecisionTreeClassifier(max_depth = d, criterion='gini')
    xval_scores.append(np.mean(cross_val_score(dt, xtrain, ytrain, cv=5)))
    
imax = xval_scores.index(np.max(xval_scores))
opt_depth = depths[imax]

dt = DecisionTreeClassifier(max_depth = opt_depth, criterion='gini', class_weight='balanced')
dt.fit(xtrain, ytrain)
tree_scores = score(dt, xtrain, ytrain)
tree_scores_test = score(dt, xtest, ytest)
```


### Random Forest



```python
#Find the optimal number of trees
trees, train_score, test_score = [], [], []

for x in range(8):
    rf = RandomForestClassifier(n_estimators=2**(x+1), max_features='auto', class_weight='balanced')
    rf_fitted = rf.fit(xtrain, ytrain)
    trees.append(2**(x+1))
    train_score.append(rf_fitted.score(xtrain, ytrain))
    test_score.append(rf_fitted.score(xtest, ytest))
```




```python
rfdf = pd.DataFrame({'Trees': trees, 'Training score': train_score, 'Test score': test_score})
rfdf = rfdf[['Trees', 'Training score', 'Test score' ]]
rfdf
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Trees</th>
      <th>Training score</th>
      <th>Test score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.949425</td>
      <td>0.903352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>0.969732</td>
      <td>0.908028</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>0.975862</td>
      <td>0.901013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>0.981609</td>
      <td>0.907249</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>0.986973</td>
      <td>0.911925</td>
    </tr>
    <tr>
      <th>5</th>
      <td>64</td>
      <td>0.989272</td>
      <td>0.908807</td>
    </tr>
    <tr>
      <th>6</th>
      <td>128</td>
      <td>0.988889</td>
      <td>0.908807</td>
    </tr>
    <tr>
      <th>7</th>
      <td>256</td>
      <td>0.989655</td>
      <td>0.908028</td>
    </tr>
  </tbody>
</table>
</div>





```python
#Find the optimal number of predictors for splitting
param_grid = dict(num_pred = list(range(1, xtrain.shape[1])))
results = {}
estimators= {}

for f in param_grid['num_pred']:
    est = RandomForestClassifier(oob_score=True, class_weight='balanced', n_estimators=32, 
                                 max_features=f, max_depth=opt_depth, n_jobs=-1)
    est.fit(xtrain, ytrain)
    results[f] = est.oob_score_
    estimators[f] = est
opt_pred = max(results, key = results.get)
```




```python
rf = RandomForestClassifier(oob_score=True, n_estimators=32, class_weight='balanced',
                            max_features=opt_pred, max_depth=opt_depth, n_jobs=-1)
rf_fitted = rf.fit(xtrain, ytrain)

RF2_scores = score(rf_fitted, xtrain, ytrain)
RF2_scores_test = score(rf_fitted, xtest, ytest)
```


### Gradient Boosting



```python
accuracies_train = []
accuracies_test = []
for md in [1,2,]:
    depth_accuracies_train = []
    depth_accuracies_test = []
    ada=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=opt_depth),n_estimators=trees[-1], learning_rate=.05)
    ada.fit(xtrain,ytrain)
    ada_train_gen = ada.staged_predict(xtrain)
    ada_test_gen = ada.staged_predict(xtest)
    for train_stagepred in ada_train_gen:    
        depth_accuracies_train.append(metrics.accuracy_score(ytrain, train_stagepred))
    for test_stagepred in ada_test_gen:   
        depth_accuracies_test.append(metrics.accuracy_score(ytest, test_stagepred))
    accuracies_train.append(depth_accuracies_train)
    accuracies_test.append(depth_accuracies_test)
```




```python
#find the optimal number of trees and the optimal tree depth for the base learner
param_grid_boost = {
              'base_estimator__max_depth': list(range(1,11))
}
gb = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=300, learning_rate=.05)
gb_cv = GridSearchCV(gb, param_grid_boost, cv=5, n_jobs=-1)

gb_cv.fit(xtrain, ytrain)

begb = gb_cv.best_estimator_
begb
```





    AdaBoostClassifier(algorithm='SAMME.R',
              base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=1,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best'),
              learning_rate=0.05, n_estimators=300, random_state=None)





```python
test_scores=[]
for spred in begb.staged_predict(xtest):
    test_scores.append(metrics.accuracy_score(spred, ytest))
    
print ("Optimal # trees = ", range(1, 301)[np.argmax(test_scores)])
print ("Optimal depth = ", 1)
```


    Optimal # trees =  149
    Optimal depth =  1




```python
gb_optimized = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), 
                                  n_estimators=149, learning_rate=.05)

gb_optimized.fit(xtrain, ytrain).score(xtest, ytest)

boost_scores = score(gb_optimized, xtrain, ytrain)
boost_scores_test = score(gb_optimized, xtest, ytest)
```




```python
#SVC 
SVC = SVC(C=100, class_weight='balanced')
SVC.fit(xtrain, ytrain)
SVC_scores = score(SVC, xtrain, ytrain)
SVC_scores_test = score(SVC, xtest, ytest)
```


## Comparison of Results

**Training Scores**



```python
score_df = pd.DataFrame({'kNN': knn_scores, 'Weighted Logistic': lm_scores,'Unweighted Logistic': Ulm_scores,
                         'LDA': LDA_scores,'QDA': QDA_scores, 'Tree': tree_scores, 'Random Forest': RF2_scores,
                         'SVC': SVC_scores, 'Boosting': boost_scores})
score_df
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Boosting</th>
      <th>LDA</th>
      <th>QDA</th>
      <th>Random Forest</th>
      <th>SVC</th>
      <th>Tree</th>
      <th>Unweighted Logistic</th>
      <th>Weighted Logistic</th>
      <th>kNN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>overall accuracy</th>
      <td>0.909962</td>
      <td>0.919157</td>
      <td>0.671648</td>
      <td>0.860920</td>
      <td>0.988889</td>
      <td>0.852490</td>
      <td>0.913793</td>
      <td>0.793103</td>
      <td>0.896935</td>
    </tr>
    <tr>
      <th>accuracy on class 0</th>
      <td>0.991029</td>
      <td>0.962409</td>
      <td>0.633917</td>
      <td>0.864588</td>
      <td>0.987612</td>
      <td>0.858607</td>
      <td>0.987185</td>
      <td>0.793251</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>accuracy on class 1</th>
      <td>0.204461</td>
      <td>0.542751</td>
      <td>1.000000</td>
      <td>0.828996</td>
      <td>1.000000</td>
      <td>0.799257</td>
      <td>0.275093</td>
      <td>0.791822</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Test Scores**



```python
score_df = pd.DataFrame({'kNN': knn_scores_test, 'Weighted Logistic': lm_scores_test,'Unweighted Logistic': Ulm_scores_test,
                         'LDA': LDA_scores_test,'QDA': QDA_scores_test, 'Tree': tree_scores_test, 'Random Forest': RF2_scores_test,
                         'SVC': SVC_scores_test, 'Boosting': boost_scores_test})
score_df
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Boosting</th>
      <th>LDA</th>
      <th>QDA</th>
      <th>Random Forest</th>
      <th>SVC</th>
      <th>Tree</th>
      <th>Unweighted Logistic</th>
      <th>Weighted Logistic</th>
      <th>kNN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>overall accuracy</th>
      <td>0.909587</td>
      <td>0.904131</td>
      <td>0.656274</td>
      <td>0.849571</td>
      <td>0.890101</td>
      <td>0.831645</td>
      <td>0.910366</td>
      <td>0.795791</td>
      <td>0.897116</td>
    </tr>
    <tr>
      <th>accuracy on class 0</th>
      <td>0.990443</td>
      <td>0.960904</td>
      <td>0.629018</td>
      <td>0.865334</td>
      <td>0.987837</td>
      <td>0.849696</td>
      <td>0.982624</td>
      <td>0.800174</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>accuracy on class 1</th>
      <td>0.204545</td>
      <td>0.409091</td>
      <td>0.893939</td>
      <td>0.712121</td>
      <td>0.037879</td>
      <td>0.674242</td>
      <td>0.280303</td>
      <td>0.757576</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Since we are interested in predicting unusually high murder rates, we are most interested in classification accuracy on class 1 rather than overall accuracy. QDA and SVC predict high murder rates with perfect accuracy in the training data, while Random Forest performs next-best, with an accuracy score for class 1 of 0.83. Weighted logistic regression and single decision tree methods perform similarly, producing an accuracy score of about 0.80. In the test set, QDA and and weighted logistic regression have the highest accuracy scores on class 1, with Random forest coming in third with an accuracy of 0.71. It is likely that the SVC model was overfitted, given its low test score. 

Although the QDA model produces the highest test scores, we choose the weighted logistic regression as our model of choice because the QDA model generates warning of collinearity in our predictors. Although we used a correlation matrix and a stepwise method to determine which variables are collinear, we were not able to determine which variables were causing the issue. Since QDA does not have a penalty function built in like some of the other models, it is likely that our model is not reliable. 

### Weighted Logistic Model

Which variables are significant? As mentioned, we are interested in knowing which predictors are important for determining high murder rates. Thus, we look at which variables are significant in our preferred model.



```python
xtrain_c = sm.add_constant(xtrain)
xtest_c = sm.add_constant(xtest)
```




```python
iterations = 100

B_boot = np.zeros((xtrain_c.shape[1],100))

for i in range(iterations):
    #sample with replacement from X_train
    boot_rows = np.random.choice(range(xtrain_c.shape[0]), size=xtrain_c.shape[0], replace=True)
    X_train_boot = xtrain_c.values[boot_rows]
    y_train_boot = ytrain.values[boot_rows]

    #fit
    lm_boot = LogisticRegression(C=10000, class_weight='balanced',fit_intercept=False)
    lm_boot.fit(X_train_boot, y_train_boot)
    B_boot[:,i] = lm_boot.coef_
```




```python
B_ci_upper = np.percentile(B_boot, 97.5, axis=1)
B_ci_lower = np.percentile(B_boot, 2.5, axis=1)
```




```python
sig_i = []
print ("Logistic regression model: features significant at 5% level")

#if ci contains 0, then insignificant
for i in range(xtrain_c.shape[1]):
    if B_ci_upper[i]<0 or B_ci_lower[i]>0:
        sig_i.append(i)

#print significant predictors
for i in sig_i:
    print(xtrain_c.columns[i])
```


    Logistic regression model: features significant at 5% level
    age21longgunpossess
    dealerh
    drugmisdemeanor
    lockd
    permitconcealed
    personalized
    log_pop2
    sex_ratio
    pct15up_widowed
    pct15up_separated
    pct25up_nohs
    pct_disabled20_64
    pct16_manuf
    pct16_prof
    LA
    MI


Our model predicts that the following variables are important for determining high murder rates:
- Laws preventing possession of long guns until age 21
- Laws stipulating that state dealer licenses are required for sale of handguns
- Laws preventing firearm possession for people with a drug misdemeanor conviction
- Laws requiring a safety lock for handguns sold through licensed dealers
- Laws requiring a permit to carry concealed weapons
- Laws requiring review of personalized gun technology
- Log popuation 
- Sex ratio
- Percent of the population age 15 and up who are widowed
- Percent of the population age 15 and up who are separated
- Percent of the population age 25 and up who have completed less than high school
- Percent of the population age 20-64 who are disabled
- Percent of the population age 16 and up who work in the manufacturing sector
- Percent of the population age 16 and up who work in the professional, science, technology sector
- Residence in Louisiana
- Residence in Mississippi

