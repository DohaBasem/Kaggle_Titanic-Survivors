# -*- coding: utf-8 -*-
"""
Created on Sat Jul 04 05:52:10 2015

@author: DOHA
"""

import pandas as pd
import numpy as np
import pylab as P
import sklearn
# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('train.csv', header=0)
#print df[['Age','Name']][0:10]
#print df.describe()
#print df[df['Age']<20][['Name','Pclass','Sex']]
bf=df[df['Age']<20][df['Pclass']==2][df['Sex']=="female"][['Name','Pclass','Sex']]
#print bf
Af=df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]
#print df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]
#for i in range(1,4):
 #   print i, len(Af[ (Af['Sex'] == 'male') & (Af['Pclass'] == i) ])
#df[df['Survived']==0]['Pclass'].hist()
#P.plot()
'''
print df[df['Survived']==0][['Pclass','Survived']]
df[df['Survived']==1][['Pclass','Age','Sex']].hist()
P.plot()
'''
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#print df.head()
print Af