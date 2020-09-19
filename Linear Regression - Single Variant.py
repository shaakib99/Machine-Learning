#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:55:49 2020

@author: wahid
"""
# Load Necessary library
import matplotlib.pyplot as plt
import pandas as pd

data = {
         'Size': [6,8,12,14,18],
         'Price': [350,775,1150,1395,1675]
     }

data = pd.DataFrame(data, columns=['Size', 'Price'])

# Data Visualisation
#plt.plot(data['Size'], data['Price'])

# Linear Regression

# Initiate Weight
weight = 0
# epoch
epoch = 7000
c = 1
learning_rate = 0.01
data_length = len(data['Size'])

for j in range(epoch):
    cost = 0
    average_prediction = 0
    hold_predicted_value = []
    for i in range(len(data['Size'])):
        # Calculate Hypothesis 
        y = weight * data['Size'][i] + c
        
        hold_predicted_value.append(y)
        # Saving value of y to calculate average output value
        average_prediction += y 
        
        cost += (1/data_length) * (y - data['Price'][i])**2
        '''
            cost = (1/m) * {h(x) - y}^2
        '''
        # Update weight
        weight = weight - ( learning_rate * (1/(2*data_length)) * (y - data['Price'][i]) * data['Size'][i])
        c = c -  learning_rate * (1/(2*data_length)) * (y - data['Price'][i])

    # Calculating R-Squared Value
    average_prediction = average_prediction / data_length
    predicted_value = 0
    real_value = 0
    for i in range(len(hold_predicted_value)):
        predicted_value += (hold_predicted_value[i] - average_prediction)**2
        real_value +=  (data['Price'][i] - average_prediction) **2
    
    print("Accuracy after ",j," iteration ", (predicted_value / real_value) * 100)

predicted_output = []
for each_value in data['Size']:
    predicted_output.append(each_value * weight + c) 
    
plt.plot(data['Size'],data['Price'], color='black')
plt.plot(data['Size'],predicted_output,color='red')
plt.legend(['Original', 'Predicted'])
plt.show()

