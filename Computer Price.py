import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

comp=pd.read_csv("D:\\excelR\\Data science notes\\multilinear Regression\\assignment\\Computer_Data.csv")
comp.columns
comp.drop(["Unnamed: 0"],inplace=True,axis=1)
comp.columns
print(comp.corr())#Correlation between the variables

#Scatter plot between the variables



#Converting string values to integer using LabelEncoding

x=comp.iloc[:,5] #cd
y=comp.iloc[:,6]  #multi
z=comp.iloc[:,7]  #premium

from sklearn.preprocessing import LabelEncoder
from numpy import array


#Converting strings from cd to integer
values = array(x)
print(values)

label_encoder=LabelEncoder()
integerEncoded= label_encoder.fit_transform(values)
print(integerEncoded)

#Converting strings from multi to integer
values1 = array(y)
print(values1)

labelEncoder1=LabelEncoder()
integerEncoded1= labelEncoder1.fit_transform(values1)
print(integerEncoded1)

#Converting strings from premium to integerr

values2= array(z)
print(values2)

labelEncoder2=LabelEncoder()
integerEncoded2=labelEncoder2.fit_transform(values2)
print(integerEncoded2)

#Dropping columns with strings 
comp.drop(["cd","multi","premium"],axis =1,inplace=True)


#Adding string converted to integer columns to the dataset
df=pd.DataFrame(comp)
df['cd']=integerEncoded
df['multi']=integerEncoded1
df['premium']=integerEncoded2

comp=df
comp.columns
comp.corr()

#Scatter plot between the variables
import seaborn as sns
sns.pairplot(comp)

#Some linearity exists between ads and trend


 #Preparing model for all the variables
import statsmodels.formula.api as smf
m1=smf.ols('price~ram+hd+speed+screen+ads+trend+cd+multi+premium',data=comp).fit()
m1.params
m1.summary()  
comp.shape

#p values are not more than 0.05 for every column/input variable. Thus ,
#prediction can be done using this model.

# Predicted values of Price
m1_pred = m1.predict(comp[['ram','hd','speed','screen','ads','trend','cd','multi','premium']])
print(m1_pred)

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
comp_train,comp_test  = train_test_split(comp,test_size = 0.3) 

# preparing the model on train data 

model_train = smf.ols("price~ram+hd+speed+screen+ads+trend+cd+multi+premium",data=comp_train).fit()

# train_data prediction
train_pred = model_train.predict(comp_train)
print(train_pred)

# train residual values 
train_resid  = train_pred - comp_train.price
print(train_resid)

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(comp_test)
print(test_pred)

# test residual values 
test_resid  = test_pred - comp_test.price
print(test_resid)
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))









