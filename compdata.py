import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

comp = pd.read_csv("D:\\excelR\\Data science notes\\multilinear Regression\\assignment\\Computer_Data.csv")
comp.columns
comp.drop("Unnamed: 0",axis=1,inplace=True)
comp.columns
comp.isnull().sum() #no NULL values 

comp = pd.get_dummies(comp)
comp.columns
comp.drop(["cd_no","multi_no","premium_no"],axis=1,inplace=True)

comp.corr()
import seaborn as sns
sns.pairplot(comp)

plt.hist(np.log(comp.hd)) # to normalize
plt.hist(np.log(comp.ram)) # to normalize

comp.hd=np.log(comp.hd)
comp.ram=np.log(comp.ram)

import statsmodels.formula.api as smf 
         
# Preparing model                  
ml1 = smf.ols('price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=comp).fit() 
ml1.summary() # r_sqr=0.764

price_pred=ml1.predict()


# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(ml1.resid_pearson, dist="norm", plot=pylab)

#to check for influential record

import statsmodels.api as sm
fig, ax = plt.subplots(figsize=(10,20))
fig = sm.graphics.influence_plot(ml1, ax=ax)

comp_new=comp.drop(comp.index[[1440,1700,79,3]],axis=0)
ml1_new = smf.ols('price~ speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data = comp_new).fit()    
ml1_new.params
ml1_new.summary() ## 0.766

# Predicted values of MPG 
price_pred = ml1_new.predict(comp_new[['speed','hd','ram','screen','ads','trend','cd_yes','multi_yes','premium_yes']])
price_pred

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(comp_new.price,price_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
# histogram
plt.hist(ml1_new.resid_pearson) # Checking the standardized residuals are normally distributed
# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(ml1_new.resid_pearson, dist="norm", plot=pylab)

############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(price_pred,ml1_new.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
comp_train,comp_test  = train_test_split(comp,test_size = 0.2) # 20% size

# Preparing model for traning data               
comp_model = smf.ols("price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes",data=comp_train).fit() 
comp_model.summary() ## 0.762


# train_data prediction
train_pred = comp_model.predict(comp_train)

# train residual values 
train_resid  = train_pred - comp_train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
train_rmse ## 280.16

# prediction on test data set 
test_pred = comp_model.predict(comp_test)

# test residual values 
test_resid  = test_pred - comp_test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
test_rmse ## 291.45

