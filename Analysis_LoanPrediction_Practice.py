# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 05:54:30 2020

@author: manju
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns=20
pd.options.display.max_columns=20
pd.options.display.width=100
  
def missingVals(data):
    missVals_count=data.isnull().sum()
    missVals_count=missVals_count[missVals_count>0]
    missVals = data.isnull().sum()/len(train)
    missVals = missVals[missVals > 0]
    missVals.sort_values(inplace=True)
    print (missVals)
    print (missVals_count)
    
    missDf=pd.DataFrame({'count':missVals.values, 'Name':missVals.index})
    
    sns.set(style="whitegrid", color_codes=True)
    sns.barplot(x = 'Name', y = 'count', data=missDf)
    plt.xticks(rotation = 45)
    plt.show()

def imputeMissing(data):
    #All categorival variable and Loan_Amount_Term imputed with mode()
    #Most common value for Loan_Amount_Term is 360, so used mode()
    #LoanAmount imputed with median(), because of outliers
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True) 
    data['Married'].fillna(data['Married'].mode()[0], inplace=True) 
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True) 
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True) 
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)

def analyze():
    #Since Loan_Status categroical, use count plot and check the distribution
    sns.countplot(train['Loan_Status'])
    plt.show()
    
    sns.set(rc={'figure.figsize':(15.0,4.0)})
    #use count plot and check the distribution of categorical values
    plt.figure(1) 
    plt.subplot(221) 
    train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 
    plt.subplot(222) 
    train['Married'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Married') 
    plt.subplot(223) 
    train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Self_Employed') 
    plt.subplot(224) 
    train['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Credit_History') 
    plt.show()
    
    #use count plot and check the distribution of ordinal values
    plt.figure(1) 
    plt.subplot(221) 
    sns.countplot(train['Dependents'])
    plt.subplot(222)
    sns.countplot(train['Education']) 
    plt.subplot(223)
    sns.countplot(train['Property_Area']) 
    plt.subplot(224)
    train['Loan_Amount_Term'].value_counts().plot.bar(figsize=(20,10),title='Loan_Amount_Term') 
    plt.show()
    
    #Numerical features plots
    plt.figure(1)
    plt.subplot(321)
    sns.distplot(train['ApplicantIncome'])
    plt.subplot(322) 
    train['ApplicantIncome'].plot.box(figsize=(20,10)) 
    
    plt.subplot(323)
    sns.distplot(train['CoapplicantIncome'])
    plt.subplot(324) 
    train['CoapplicantIncome'].plot.box(figsize=(20,10)) 
    
    plt.subplot(325)
    sns.distplot(train['LoanAmount'].dropna())
    plt.subplot(326) 
    train['LoanAmount'].plot.box(figsize=(20,10)) 
    plt.show()
    
    # Segregarte incomes by Education to check outliers
    train.boxplot(column='ApplicantIncome', by = 'Education');plt.suptitle("")
    train.boxplot(column='CoapplicantIncome', by = 'Education'); plt.suptitle("")
    
    
    
    #### Bi Variate Analysis ####
    
    # Categorical vs Loan_Status
    Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
    Married=pd.crosstab(train['Married'],train['Loan_Status'])
    Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
    Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
    Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
    Education=pd.crosstab(train['Education'],train['Loan_Status'])
    Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
    #Loan_Amount_Term=pd.crosstab(train['Loan_Amount_Term'],train['Loan_Status'])
        
    Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    
    Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    
    Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    
    Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    
    Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    
    Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    
    Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    
    #Loan_Amount_Term.div(Loan_Amount_Term.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    
    
    # Numerical vs Loan_Status
    #ApplicantIncome and CoapplicantIncome  vs Loan_Status. Will bin the ApplicantIncome and CoapplicantIncome data
    
    bins=[0,2500,4000,6000,90000]
    grp=['Low','Average','High','Very High']
    train['Income_Bin']=pd.cut(train['ApplicantIncome'], bins, labels=grp)
    
    Income_Bin=pd.crosstab(train['Income_Bin'],train['Loan_Status'])
    Income_Bin.div(Income_Bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    plt.xlabel('ApplicantIncome')
    plt.ylabel('%')
    
    train['CoIncome_Bin']=pd.cut(train['CoapplicantIncome'], bins, labels=grp)
    CoIncome_Bin=pd.crosstab(train['CoIncome_Bin'],train['Loan_Status'])
    CoIncome_Bin.div(CoIncome_Bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    plt.xlabel('CoapplicantIncome')
    plt.ylabel('%')
    
    train['TotIncome']=train['ApplicantIncome']+train['CoapplicantIncome']
    bins=[0,2500,8000,20000,90000]
    grp=['Low','Average','High','Very High']
    train['TotIncome_Bin']=pd.cut(train['TotIncome'], bins, labels=grp)
    
    Income_Bin=pd.crosstab(train['TotIncome_Bin'],train['Loan_Status'])
    Income_Bin.div(Income_Bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    plt.xlabel('Total Income')
    plt.ylabel('%')
    
    #LoanAmount binning
    bins=[0,100,300,700]
    grp=['Low','Average','High']
    train['LoanAmount_Bin']=pd.cut(train['LoanAmount'], bins, labels=grp)
    
    Income_Bin=pd.crosstab(train['LoanAmount_Bin'],train['Loan_Status'])
    Income_Bin.div(Income_Bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    plt.xlabel('Loan Amount')
    plt.ylabel('%')
    
#EMI
    train['EMI']=(train['LoanAmount']*1000*R*pow((1+R),train['Loan_Amount_Term']))/(pow((1+R),train['Loan_Amount_Term'])-1)
    bins=[0,5000,15000,25000,90000]
    grp=['Low','Average','High','Very High']
    train['EMI_Bin']=pd.cut(train['EMI'], bins, labels=grp)
    
    EMI_Bin=pd.crosstab(train['EMI_Bin'],train['Loan_Status'])
    EMI_Bin.div(EMI_Bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    plt.xlabel('EMI')
    plt.ylabel('%')
    
#EMI to Income ratio
    train['EIR']=train['EMI']/train['TotIncome']
    bins=[0,2,7]
    grp=['Low','High']
    train['EIR_Bin']=pd.cut(train['EIR'], bins, labels=grp)
    
    EIR_Bin=pd.crosstab(train['EIR_Bin'],train['Loan_Status'])
    EMI_Bin.div(EIR_Bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
    plt.xlabel('EIR')
    plt.ylabel('%')
    
    plt.show()
    
    #Delete newly created bin variables
    train.drop(['Income_Bin', 'CoIncome_Bin', 'TotIncome_Bin', 'LoanAmount_Bin', 'TotIncome'], axis=1, inplace=True)
    
    #Correlation mapping b/w numerical features and Loan_Status
    #Convert Loan_Status into numerical
    train['Loan_Status']=train['Loan_Status'].map({'Y':1,'N':0})
    #Dependents have some string data like '3+', convert them to numerical
    train['Dependents'].replace('3+',3, inplace=True)
    test['Dependents'].replace('3+',3, inplace=True)
    
    train_corr=train.corr()
    
    sns.heatmap(train_corr, square=True)
    print (train_corr['Loan_Status'].sort_values(ascending=False))
    print (train_corr['LoanAmount'].sort_values(ascending=False))
    #Convert Loan_Status into categorical - we will do the conversion during modelling again
    train['Loan_Status']=train['Loan_Status'].map({1:'Y',0:'N'})

#### MAIN #####
train=pd.read_csv('train_ctrUa4K.csv')
test=pd.read_csv('test_lAUu6dG.csv')

print ("Train set has {0} Rows and {1} Columns".format(train.shape[0], train.shape[1]))
print ("Test set has {0} Rows and {1} Columns".format(test.shape[0], test.shape[1]))

#Drop the loan_iD feature that doesn't add much information for modelling
train.drop(['Loan_ID'], axis=1, inplace=True)
testLoan_ID=test['Loan_ID']
test.drop(['Loan_ID'], axis=1, inplace=True)

#Analysis
analyze()

#MISSING VALUES TREATMENT
#Find which are the fetures having missing values
print ("TRAIN:")
missingVals(train)
print ("TEST:")
missingVals(test)

#Impute missing values
print ("IMPUTING TRAIN:")
imputeMissing(train)
print ("IMPUTING TEST:")
imputeMissing(test)

print ("TRAIN AFTER IMPUTING:\n", train.isnull().sum())
print ("TEST  AFTER IMPUTING:\n", test.isnull().sum())
