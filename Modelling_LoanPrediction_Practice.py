# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 05:47:16 2019
@author: Manjunath G

Prediction of Loan - Practice problem from AnalyticsVidhya
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

loanData_train=pd.read_csv('train_ctrUa4K.csv')
loanData_test=pd.read_csv('test_lAUu6dG.csv')

# Get the Loan_Status as targets from the training set and map it to 1/0 (Y/N)
loanTargets=loanData_train['Loan_Status'].map({'Y':1,'N':0})

#Drop the Loan_Status from traing set and stack tain and test set together
#Will be easy to process the null values and datapreprocessing
loanData_train.drop('Loan_Status', axis=1, inplace=True)
loanData_all=pd.concat([loanData_train, loanData_test])

loanData_all['TotalIncome']=loanData_all['ApplicantIncome']+loanData_all['CoapplicantIncome']

#Drop Load_ID feature
loanData_all.drop(['Loan_ID'], axis=1, inplace=True)

### HANDLE NULL VALUES : IMPUTING ###
#Normalize the   features Income, and LoanAmount for optimal performanace 
cat_cols = ['Dependents', 'Gender', 'Married', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'LoanAmount']

for cat_f in cat_cols:
        loanData_all[cat_f].fillna(loanData_all[cat_f].mode()[0],inplace=True)

### MAP NON NUMERIC VALUES TO NUMERIC VALLUES ###
loanData_all['Gender'] = loanData_all['Gender'].map({'Male':1,'Female':0})
loanData_all['Married'] = loanData_all['Married'].map({'Yes':1,'No':0})
loanData_all['Education'] = loanData_all['Education'].map({'Graduate':1,'Not Graduate':0})
loanData_all['Self_Employed'] = loanData_all['Self_Employed'].map({'Yes':1,'No':0})
loanData_all['Dependents'] = loanData_all['Dependents'].map({'0':0,'1':1, '2':2, '3':3, '3+':3})

# Perform one-hot encoding on nominal feature Property
property_dummies = pd.get_dummies(loanData_all['Property_Area'], prefix='PropArea')
loanData_all = pd.concat([loanData_all, property_dummies], axis=1)
loanData_all.drop(['Property_Area'], axis=1, inplace=True)

R=0.080 # Rate of Interest 12% => 12/100*12 => 0.01 per month
loanData_all['EMI']=(loanData_all['LoanAmount']*1000*R*pow((1+R),loanData_all['Loan_Amount_Term']))/(pow((1+R),loanData_all['Loan_Amount_Term'])-1)
loanData_all['EIR']=loanData_all['EMI']/loanData_all['TotalIncome']

#print (loanData_all.isnull().sum())

#Standardize the   features Income, and LoanAmount for optimal performanace 
minmaxscaler=MinMaxScaler()
#Normalize the   features Income, and LoanAmount for optimal performanace 
num_cols = ['TotalIncome', 'EMI', 'EIR', 'LoanAmount', 'Loan_Amount_Term']

for num_f in num_cols:
        loanData_all[num_f]=minmaxscaler.fit_transform(np.array(loanData_all[num_f]).astype(float).reshape(len(loanData_all[num_f]),1))
        
#Sperate Train and  Test set
loanTrain=loanData_all[0:loanData_train.shape[0]]
loanTest=loanData_all[loanData_train.shape[0]:]

### USING RandomForestClassifier for classification ###
#random_state=239 => 0.8125

clf=RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=8, min_samples_leaf=4, min_samples_split=4, random_state=239)

### Using  LogisticRegression algorithm
#clf=LogisticRegression(C=0.1, random_state=10, solver='liblinear', multi_class='ovr', penalty='l2')

clf.fit(loanTrain, loanTargets)


### MODELLING ###
xval = cross_val_score(clf, loanTrain, loanTargets, cv = 5, scoring='accuracy')
print ("Accuracy :", np.mean(xval))


### Predict on test data and generate output
loanData_test['Loan_Status']=np.vectorize(lambda s: 'Y' if s==1 else 'N')(clf.predict(loanTest))
loanData_test[['Loan_ID','Loan_Status']].to_csv('TestDataPred.csv', index=False)