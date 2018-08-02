import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#load and prep data

data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
print(rides.head())


