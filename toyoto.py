

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

toy=pd.read_csv("D:\\excelR\\Data science notes\\multilinear Regression\\assignment\\ToyotaCorolla.csv",header=0,encoding = 'unicode_escape')
toy.head(20)
toy.corr()
toy.columns

#sns.pairplot(toy)

# preparing model considering all the variables 
ml= smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data=toy).fit()
ml.corr()
ml.params
ml.summary() ## r-sqr = 0.864
colnames=toy.columns
a=toy.iloc[:,12:14]
a.corr()
# p-values for cc,doors are more than 0.05

# preparing model based only on cc
ml1=smf.ols('Price~cc',data = toy).fit()  
ml1.summary() # 0.016
# p-value <0.05 .. It is significant 

# Preparing model based only on doors
ml2=smf.ols('Price~Doors',data = toy).fit()  
ml2.summary() # 0.034

# Preparing model based only on cc & Doors
ml3=smf.ols('Price~cc+Doors',data = toy).fit()  
ml3.summary() # 0.047


# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(ml)

fig, ax = plt.subplots(figsize=(10,15))
fig = sm.graphics.influence_plot(ml, ax=ax)

# index 80 is showing high influence so we can exclude that entire row

toy_new=toy.drop(toy.index[80],axis=0)

ml_new= smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data=toy_new).fit()
plt.hist(ml_new.resid_pearson)
ml_new.corr()

ml_new.summary()  ## 0.869

# Confidence values 99%
print(ml_new.conf_int(0.01)) # 99% confidence level

price_pred = ml_new.predict(toy_new[['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']])
price_pred

toy_new.head()
# calculating VIF's values of independent variables
rsq_age = smf.ols('Price~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toy_new).fit().rsquared  
vif_age = 1/(1-rsq_age) # 3.001
vif_age
rsq_km = smf.ols('Price~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toy_new).fit().rsquared  
vif_km = 1/(1-rsq_km) # 6.733
vif_km
rsq_hp = smf.ols('Price~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight',data=toy_new).fit().rsquared  
vif_hp = 1/(1-rsq_hp) #  6.786
vif_hp
rsq_cc = smf.ols('Price~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight',data=toy_new).fit().rsquared  
vif_cc = 1/(1-rsq_cc) #  7.309
vif_cc
rsq_doors = smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=toy_new).fit().rsquared  
vif_doors = 1/(1-rsq_doors) #  7.653
vif_doors
rsq_gears=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight',data=toy_new).fit().rsquared  
vif_gears = 1/(1-rsq_gears) #  7.616
vif_gears
rsq_tax = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Weight',data=toy_new).fit().rsquared  
vif_tax = 1/(1-rsq_tax) #  7.445
vif_tax
rsq_wt = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax',data=toy_new).fit().rsquared  
vif_wt = 1/(1-rsq_wt) #  6.228
vif_wt

           # Storing vif values in a data frame
d1 = {'Variables':['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'],'VIF':[vif_age,vif_km,vif_hp,vif_cc,vif_doors,vif_gears,vif_tax,vif_wt]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# As Doors is having higher VIF value, we are not going to include this prediction model

# Added varible plot 
sm.graphics.plot_partregress_grid(ml_new)

# added varible plot for Doors is not showing any significance 

# final model
final_ml= smf.ols('Price~Age_08_04+KM+HP+cc+Quarterly_Tax+Weight',data = toy_new).fit()
final_ml.params
final_ml.summary() # 0.869

# As we can see that r-squared value has not increased 
price_pred1= final_ml.predict(toy_new)
price_pred1
import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(toy_new.Price,price_pred1,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(price_pred1,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
########    Normality plot for residuals ######

plt.hist(final_ml.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_ml.resid_pearson, dist="norm", plot=pylab)

############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(price_pred1,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
toy_train,toy_test  = train_test_split(toy_new,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train= smf.ols('Price~Age_08_04+KM+HP+cc+Quarterly_Tax+Weight',data = toy_train).fit()

# train_data prediction
train_pred = model_train.predict(toy_train)

# train residual values 
train_resid  = train_pred - toy_train.Price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
train_rmse  
# prediction on test data set 
test_pred = model_train.predict(toy_test)

# test residual values 
test_resid  = test_pred - toy_test.Price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
test_rmse 
