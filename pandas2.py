# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 23:21:44 2015

@author: DOHA
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import re
import numpy as np
# read in the training and testing data into Pandas.DataFrame objects
input_df = pd.read_csv('train.csv', header=0)
submit_df  = pd.read_csv('test.csv',  header=0)

# merge the two DataFrames into one
df = pd.concat([input_df, submit_df])

# re-number the combined data set so there aren't duplicate indexes
df.reset_index(inplace=True)

# reset_index() generates a new column that we don't want, so let's get rid of it
df.drop('index', axis=1, inplace=True)

# the remaining columns need to be reindexed so we can access the first column at '0' instead of '1'
df = df.reindex_axis(input_df.columns, axis=1)

#1>FILLING MISSING VALUES
#from the info we get to know that out of the 12 col
# [Age,Fare,Cabin,Embarked] columns are having missing vals
#For the fare -->Assign a value that indicates a missing value
df.loc[df.Fare.isnull(),('Fare')]=-1
#replace the null values in Embarked with the most common values
df.loc[df.Embarked.isnull(),('Embarked')]=df.Embarked.dropna().mode().values
#Since Random Forests doesn't accept strings
#Change Gender into values of 0 for female where 1 for male
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#Change Gender into values of 0 for 'c' where 1 for 'q' and 2 for 'S'
df['Emabrked-Num'] = df['Embarked'].map( {'C': 0, 'Q': 1 ,'S':2} ).astype(int)
print "HERE"
#Filling the missing values in Age using Random Forest
#1.The features used in the regressor
age_df = df[['Age','Gender','Fare', 'Parch', 'SibSp', 'Pclass']]

 #2. Split into sets with known and unknown Age values
knownAge = age_df.loc[ (df.Age.notnull()) ]
unknownAge = age_df.loc[ (df.Age.isnull()) ]

 # All age values are stored in a target array
y = knownAge.values[:, 0]
    
    # All the other values are stored in the feature array
X = knownAge.values[:, 1::]

rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
rtr.fit(X, y)
# Use the fitted model to predict the missing values
predictedAges = rtr.predict(unknownAge.values[:, 1::])

df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

# Filling the Cabin 

#2.fEATURE SELECTION AND GENERATION
# The Name could imply the gender and the family which may have an implication to the social status
#which accordingly the person will survive or not
df['Family'] = df['Name'].map(lambda x: (re.split(' ', x)[0]))

#print df[['Family','Gender']]
###############################
# What is each person's title? 
df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

# Group low-occuring, related titles together
df['Title'][df.Title == 'Jonkheer'] = 'Master'
df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
df['Title'][df.Title == 'Mme'] = 'Mrs'
df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'

# Build binary features
df = pd.concat([df, pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)
print df[['Family','Gender','Title']]
###############################
#Calc. Correlation between features

df_corr = df.drop(['Survived', 'PassengerId'],axis=1).corr(method='spearman')
# create a mask to ignore self-

mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)

#print df_corr
df_corr = mask * df_corr
drops = []
# loop through each variable
for col in df_corr.columns.values:
    # if we've already determined to drop the current variable, continue
    if np.in1d([col],drops):
        continue
    
    # find all the variables that are highly correlated with the current variable 
    # and add them to the drop list 
    corr = df_corr[abs(df_corr[col]) > 0.69].index
    drops = np.union1d(drops, corr)
 
print "\nDropping", drops.shape[0], "highly correlated features...\n", drops
df.drop(drops, axis=1, inplace=True)