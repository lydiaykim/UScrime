---
title: Murder Rate Models
notebook: 5_continuous_models.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}



**What are the major predictors of murder rates in the United States?**

Here we look at murder rates as the outcome and use census and gun law data as predictors. We use four main models for this prediction. The models we explore are: 
- OLS 
- Ridge regression
- LASSO regression
- kNN

## Import libraries



```python
#import libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.api import OLS
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
%matplotlib inline
import seaborn.apionly as sns
import datetime
from datetime import tzinfo, timedelta, datetime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from scipy import stats


```


    /anaconda/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


## Data set up 



```python
#import data
df = pd.read_csv('cleaned data/merged_dataset.csv')
df.shape
```





    (3893, 304)





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





    (3893, 318)





```python
df.head()
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
      <th>Unnamed: 0</th>
      <th>MSA_name</th>
      <th>Total</th>
      <th>Estimated</th>
      <th>Rate</th>
      <th>MSA</th>
      <th>year</th>
      <th>state</th>
      <th>age18longgunpossess</th>
      <th>age18longgunsale</th>
      <th>...</th>
      <th>log_mean_inc_black</th>
      <th>log_mean_inc_black2</th>
      <th>log_mean_inc_hispanic</th>
      <th>log_mean_inc_hispanic2</th>
      <th>poor_nonwhite</th>
      <th>LA</th>
      <th>AK</th>
      <th>MI</th>
      <th>SC</th>
      <th>OK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Abilene, TX</td>
      <td>6.0</td>
      <td>6</td>
      <td>3.7</td>
      <td>10180</td>
      <td>2006</td>
      <td>Texas</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>9.384882</td>
      <td>88.076005</td>
      <td>9.312807</td>
      <td>86.728369</td>
      <td>4.2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2078</td>
      <td>Abilene, TX</td>
      <td>3.0</td>
      <td>3</td>
      <td>1.8</td>
      <td>10180</td>
      <td>2012</td>
      <td>Texas</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>9.146228</td>
      <td>83.653494</td>
      <td>9.390743</td>
      <td>88.186062</td>
      <td>4.5</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2805</td>
      <td>Abilene, TX</td>
      <td>10.0</td>
      <td>10</td>
      <td>5.9</td>
      <td>10180</td>
      <td>2014</td>
      <td>Texas</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>9.423029</td>
      <td>88.793484</td>
      <td>9.555135</td>
      <td>91.300605</td>
      <td>3.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1723</td>
      <td>Abilene, TX</td>
      <td>5.0</td>
      <td>5</td>
      <td>3.0</td>
      <td>10180</td>
      <td>2011</td>
      <td>Texas</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>9.573176</td>
      <td>91.645704</td>
      <td>9.471935</td>
      <td>89.717555</td>
      <td>1.9</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1367</td>
      <td>Abilene, TX</td>
      <td>5.0</td>
      <td>5</td>
      <td>3.1</td>
      <td>10180</td>
      <td>2010</td>
      <td>Texas</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>9.345570</td>
      <td>87.339685</td>
      <td>9.357380</td>
      <td>87.560563</td>
      <td>3.7</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 318 columns</p>
</div>





```python
#split into training and test datasets 
np.random.seed(1818)
msk = np.random.rand(len(df)) < 0.66
data_train = df[msk]
data_test = df[~msk]
```




```python
#set the predictors we will use 
df_gun_laws=pd.read_csv('raw data/state-firearms/raw_data.csv')
gun_vars = list(df_gun_laws)[2:]


#census vars
cen_vars1=['pct_hispanic', 'pct_white', 'pct_black', 'pct_indian', 'pct_asian', 'pct_hawaiian', 
           'pct_other', 'pct_mixed', 'pct_foreign_citizen']
cen_vars2 = ['log_pop', 'log_pop2', 'pop5_14', 'pop15_17', 'pop18_24', 'pop15_44', 'pop65up', 
             'median_age', 'sex_ratio',  'age_dep', 'oldage_dep', 'child_dep']
cen_vars3 = ['families_singledad', 'families_singlemom', 'family_size',  'pct_ownhouse']
cen_vars4 = ['pct15up_married', 'pct15up_widowed', 'pct15up_divorced', 'pct15up_separated', 
             'pct15up_nevermar', 'pct15up_married_f', 'pct15up_divorced_f', 'pct15up_separated_f' , 
             'pct15up_married_m', 'pct15up_divorced_m', 'pct15up_separated_m' ]
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
data_train=data_train[var_needed]
data_test=data_test[var_needed]
print(data_train.shape, data_test.shape)
```


    (2610, 225) (1283, 225)




```python
#set the three different outcome variables we will be looking at
y_total_train = data_train['Total']
y_total_test = data_test['Total']

y_est_train = data_train['Estimated']
y_est_test = data_test['Estimated']

y_rate_train = data_train['Rate']
y_rate_test = data_test['Rate']

#make x_test and x_train
X_train=data_train[predictors]
X_test=data_test[predictors]
```


## Variable selection using LASSO



```python
#Lasso regression
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5]

lasso_cv = LassoCV(alphas=alphas, fit_intercept=True, cv=10, normalize=True)
lasso_cv.fit(X_train, y_rate_train)

print('Lasso training R^2:', lasso_cv.score(X_train, y_rate_train))
print('Lasso test R^2:', lasso_cv.score(X_test, y_rate_test))
```


    /anaconda/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:484: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    Lasso training R^2: 0.503825551412
    Lasso test R^2: 0.504268619176




```python
betas = np.absolute(lasso_cv.coef_)
neg = np.where(betas < 1e-10)
neg[0]
zero_coef = []
for x in neg[0]:
    zero_coef.append(X_train.columns[x])
```




```python
#Which predictors did Lasso keep (ie not set to near 0)?
betas = np.absolute(lasso_cv.coef_)
neg = np.where(betas > 1e-10)
nonzero_coef = []
print("LASSO Predictors with coefficients not near 0:")
for x in neg[0]:
    nonzero_coef.append(X_train.columns[x])
    print(X_train.columns[x])
```


    LASSO Predictors with coefficients not near 0:
    age21longgunpossess
    ccbackgroundnics
    ccrevoke
    dealerh
    drugmisdemeanor
    incidentremoval
    lockd
    loststolen
    onepermonth
    opencarryh
    opencarrypermith
    permitconcealed
    personalized
    recordsallh
    stalking
    pct_white
    pct_black
    pct_indian
    pct_foreign_citizen
    log_pop
    pop15_17
    sex_ratio
    families_singledad
    families_singlemom
    pct15up_widowed
    pct15up_separated
    pct15up_married_m
    pct_enroll_private
    pct18_24_lesshs
    pct25up_nohs
    pct25up_somecol
    pct25up_somecol2
    pct25up_grad
    poor_nonwhite
    fam_cash_assist
    income25_35
    income35_50
    income50_75
    income200up
    pct_disabled20_64
    unemployed_white
    pct16_manuf
    LA
    AK
    MI
    OK




```python
#drop unimportant coefficients from LASSO
X_train = X_train.drop(zero_coef, axis=1)
X_test = X_test.drop(zero_coef, axis=1)
X_train.shape
```





    (2610, 46)



## Models



```python
#OLS
LinearReg = LinearRegression(fit_intercept=True)
LinearReg.fit(X_train, y_rate_train)
np.mean(cross_val_score(LinearReg, X_train, y_rate_train, cv=5))

OLS_scores = pd.Series([np.mean(cross_val_score(LinearReg, X_train, y_rate_train, cv=5)), 
            np.mean(cross_val_score(LinearReg, X_test, y_rate_test,cv=5))], 
            index=['Train R^2', 'Test R^2'])
```




```python
#Ridge regression
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
a={'alpha': alphas}
ridge=Ridge(fit_intercept=True, normalize=True)
ridge_cv = GridSearchCV(ridge, param_grid=[a], cv=10)
ridge_cv.fit(X_train, y_rate_train)

Ridge_scores = pd.Series([ridge_cv.score(X_train, y_rate_train), ridge_cv.score(X_test, y_rate_test)], 
            index=['Train R^2', 'Test R^2'])
```




```python
#Lasso regression
lasso_cv = LassoCV(alphas=alphas, fit_intercept=True, cv=10, normalize=True)
lasso_cv.fit(X_train, y_rate_train)

LASSO_scores = pd.Series([lasso_cv.score(X_train, y_rate_train), lasso_cv.score(X_test, y_rate_test)], 
            index=['Train R^2', 'Test R^2'])
```




```python
#kNN
klist = {'n_neighbors':[2, 5, 10, 20, 40, 50]}
knn_cv = GridSearchCV(KNeighborsRegressor(), param_grid=[klist], cv=10)
knn_cv.fit(X_train, y_rate_train)

kNN_scores = pd.Series([knn_cv.score(X_train, y_rate_train), knn_cv.score(X_test, y_rate_test)], 
            index=['Train R^2', 'Test R^2'])
```




```python
#Score Dataframe
score_df = pd.DataFrame({'OLS': OLS_scores, 'Ridge': Ridge_scores,
                         'LASSO': LASSO_scores,'kNN': kNN_scores})
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
      <th>LASSO</th>
      <th>OLS</th>
      <th>Ridge</th>
      <th>kNN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train R^2</th>
      <td>0.519474</td>
      <td>0.446783</td>
      <td>0.512779</td>
      <td>0.228430</td>
    </tr>
    <tr>
      <th>Test R^2</th>
      <td>0.518703</td>
      <td>0.437526</td>
      <td>0.517634</td>
      <td>0.169981</td>
    </tr>
  </tbody>
</table>
</div>





```python
xtrain_c = sm.add_constant(X_train)
xtest_c = sm.add_constant(X_test)
```




```python
iterations = 100

B_boot = np.zeros((xtrain_c.shape[1],100))

for i in range(iterations):
    #sample with replacement from X_train
    boot_rows = np.random.choice(range(xtrain_c.shape[0]), size=xtrain_c.shape[0], replace=True)
    X_train_boot = xtrain_c.values[boot_rows]
    y_train_boot = y_rate_train.values[boot_rows]

    #fit
    lm_boot = LinearRegression(fit_intercept=False)
    lm_boot.fit(X_train_boot, y_train_boot)
    B_boot[:,i] = lm_boot.coef_
```




```python
B_ci_upper = np.percentile(B_boot, 97.5, axis=1)
B_ci_lower = np.percentile(B_boot, 2.5, axis=1)
```




```python
sig_i = []
print ("OLS regression model: features significant at 5% level")

#if ci contains 0, then insignificant
for i in range(xtrain_c.shape[1]):
    if B_ci_upper[i]<0 or B_ci_lower[i]>0:
        sig_i.append(i)

#print significant predictors
for i in sig_i:
    print(xtrain_c.columns[i])
```


    OLS regression model: features significant at 5% level
    const
    age21longgunpossess
    ccbackgroundnics
    ccrevoke
    dealerh
    drugmisdemeanor
    lockd
    opencarryh
    opencarrypermith
    permitconcealed
    stalking
    pct_white
    pct_black
    pct_foreign_citizen
    log_pop
    sex_ratio
    families_singledad
    families_singlemom
    pct15up_widowed
    pct15up_separated
    pct_enroll_private
    pct25up_somecol
    fam_cash_assist
    pct_disabled20_64
    unemployed_white
    pct16_manuf
    AK
    MI
    OK

