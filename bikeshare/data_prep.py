import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#load and prep data

data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
print(rides.head())


#plot of number of bike riders over the first 10 days in the data set
#rides[:24*10].plot(x='dteday', y='cnt')
#plt.show()

# To include some categorical variables like season, weather, month in my model, they will be  made binary dummy variables

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
print(data.head())