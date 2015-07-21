# -*- coding: utf-8 -*-
"""
Created on Fri Jul 03 04:45:14 2015

@author: DOHA
"""

import numpy as np
import csv as csv
# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('train.csv', 'rb')) 
header=csv_file_object.next()
data=[]
for row in csv_file_object:
    data.append(row)
data = np.array(data) #data is an array of lists
tot_passengers=np.size(data[0::,0].astype(np.float)) #Total number of passengers
survived=np.sum(data[0::,1].astype(np.float))  #Total number of those who actually survived
fraction_survival= survived/tot_passengers     #fraction for survival from all the passengers
women_only_stats = data[0::,4] == "female"
men_only_stats=data[0::,4] !="female"

tot_women=np.size(data[women_only_stats,1].astype(np.float)) #Total number of women on board
tot_men=np.size(data[men_only_stats,1].astype(np.float)) #Total number of men on board
# Then we finds the proportions of them that survived
proportion_women_survived = \
                      np.sum(data[women_only_stats,1].astype(np.float)) / tot_women  
proportion_men_survived = \
                       np.sum(data[men_only_stats,1].astype(np.float)) / tot_men

# and then print it out
print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived

ages_onboard = data[0::,5].astype(np.float) 
print "AGES"
print ages_onboard

#The ratio of women who survived is > than that of men
#so our prediction will be based on this fact
#using the test file
test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

prediction_file = open("genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close()
prediction_file.close()
