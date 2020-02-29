

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start=pd.read_csv("D:\\excelR\\Data science notes\\multilinear Regression\\assignment\\50_Startups.csv")
start.columns
#Converting string values to integer using LabelEncoding
start.rename(columns={"RDS": "RD","MS":"Marketing"},inplace=True)

x=start.iloc[:,3] #state


from sklearn.preprocessing import LabelEncoder
from numpy import array


#Converting strings from state to integer
values = array(x)
print(values)

label_encoder=LabelEncoder()
integerEncoded= label_encoder.fit_transform(values)
print(integerEncoded)


#Dropping columns with strings 
start.drop(["State"],axis =1,inplace=True)

#Adding string converted to integer columns to the dataset
df=pd.DataFrame(start)
df['State']=integerEncoded
start.columns
start.corr()
#High colinearity between R&D and Profit
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(start)

# preparing model considering all the variables 
import statsmodels.formula.api as smf 
                  
m1 = smf.ols('Profit~RD+Administration+Marketing+State',data=start).fit()

# Getting coefficients of variables               
m1.params

# Summary
m1.summary() #R squared value : 0.776
#Administration and State are not having significant values


#Preparing model only for Admiminstration
ml_ad=smf.ols('Profit~Administration',data = start).fit()  
ml_ad.summary()#0.162   R=0.40

#Preparing model only for State
ml_st=smf.ols('Profit~State',data =start).fit()  
ml_st.summary()#0.482  R=0.010

#Preparing model only for Marketing
ml_mk=smf.ols('Profit~Marketing',data = start).fit()  
ml_mk.summary() #R=0.550


#Preparing model for Administartion and State both
ml_as=smf.ols('Profit~Administration+State',data =start).fit()  
ml_as.summary()#0.661(State) R=0.50

#Preparing model for Administartion,State and Marketing
ml_asm=smf.ols('Profit~Administration+State+Marketing',data =start).fit()  
ml_asm.summary()#0.661(State)  R=0.611

#Influential Index plots for checking the influential values
import statsmodels.api as sm
sm.graphics.influence_plot(m1)
#Index 49,45,48 are having high infulence , so we can exclude their entire rows
start_new=start.drop(start.index[[49,45,48]],axis=0)

# Preparing  a new model                  
ml_new = smf.ols('Profit~RD+Administration+Marketing+State',data=start_new).fit()
    
ml_new.params

# Summary
ml_new.summary()  #R=0.946

# Predicted values of Profit
start_pred = ml_new.predict(start_new[['RD','Administration','Marketing','State']])
print(start_pred)

#Calculating VIFs of individual variable
rsq_RD = smf.ols('RD~Administration+Marketing+State',data=start_new).fit().rsquared  
vif_RD = 1/(1-rsq_RD)

rsq_Administration = smf.ols('Administration~Marketing+State+RD',data=start_new).fit().rsquared  
vif_Administration = 1/(1-rsq_Administration) 

rsq_Marketing = smf.ols('Marketing~RD+State+Administration',data=start_new).fit().rsquared  
vif_Marketing = 1/(1-rsq_Administration) 

rsq_State = smf.ols('State~RD+Marketing+Administration',data=start_new).fit().rsquared  
vif_State = 1/(1-rsq_Administration) 


# Storing vif values in a data frame
d1 = {'Variables':['RD','Administration','Marketing','State'],'VIF':[vif_RD,vif_Administration,vif_Marketing,vif_State]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# Added varible plot 
sm.graphics.plot_partregress_grid(ml_new)
 #Variable plot for State is not showing any significance

#Computing the final model 
 final_ml= smf.ols('Profit~Administration+Marketing+RD',data = start_new).fit()
final_ml.params
final_ml.summary() #R:0.964
 #Even after omitting the state column , there exists non significant codes,
 #Therefore,if administration or Marketing gets omitted,we get significant codes.
 final_ml= smf.ols('Profit~Administration+RD',data = start_new).fit()
final_ml.params
final_ml.summary()

#  Linearity
# Observed values VS Fitted values
plt.scatter(start_new.Profit,start_pred,c="r");
plt.xlabel("observed_values");
plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(start_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


#Splitting the data into train and test
from sklearn.model_selection import train_test_split
start_train,start_test  = train_test_split(start_new,test_size = 0.3) # 20% size

# preparing the model on train data 

model_train = smf.ols("Profit~RD+Administration",data=start_train).fit()

# train_data prediction
train_pred = model_train.predict(start_train)
print(train_pred)
# train residual values 
train_resid  = train_pred - start_train.Profit
print(train_resid)

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(start_test)
print(test_pred)

# test residual values 
test_resid  = test_pred - start_test.Profit
print(test_resid)
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

#R squared values for every model
R_SqrdValue = {'Models':['SLR','Administration','State','Marketing','Adminis./State','Adminis./State/Marketing','Final'],'R^2':[0.776,0.40,0.010,0.550,0.50,0.611,0.946]}
RSqrd_frame = pd.DataFrame(R_SqrdValue)  
Rsqrd_frame

startup.py
Displaying startup.py.
