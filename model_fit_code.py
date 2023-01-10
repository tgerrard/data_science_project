import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
import geopandas as gpd
import seaborn as sns

#If required to install geopython, uncomment line below
#pip install geopy 

#import data and have a look at it
data = pd.read_csv('covid_data.csv')
# to explicitly convert the date column to type DATETIME(DD-MM-YYYY)
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)

print(data)

#have a look at a plot of the data
#data.plot()

#convert dates to ordinals as linear regression cannot deal with dates
#Ordinals are number codes for dates. first day of the first year = 1 then each day increases by 1 from that point 
dates_to_ordinals = data["Date"].map(dt.datetime.toordinal)

#print these to check they have been changed correctly
print(dates_to_ordinals)

#Training a linear model to try and predict future covid stats
#set our X and y as our target variables to train the model
#in this case dates and New cases 
X = dates_to_ordinals
y = data["New cases"]

#Split the dataset into training and test sets 
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

#Plots the data with "o" being datapoints used for training the dataset and "x" used for testing the dataset
#Check visually that the split is fairly even
fig, ax = plt.subplots()

ax.scatter(train_X, train_y, color="red", marker="o", label="train")
ax.scatter(test_X, test_y, color="blue", marker="x", label="test")
ax.legend()
plt.show()

#reshape our dates collumn so it is 2 dimensional as required by sklearn
train_X = train_X.values.reshape(-1, 1)
test_X = test_X.values.reshape(-1, 1)

#fit the model with our training sets 
model = LinearRegression(fit_intercept=True)
model.fit(train_X, train_y)

#get a score for the model
model_score = model.score(test_X, test_y)
print(f'Model score =')
print(model_score)


#Now that we have trained and tested our model on the data we already have, we can use it to predict the future of covid new cases
#we can enter any ordinal date and the model will predict the number of new cases on that date 
pred = 737733
y_prediction = model.predict([[pred]])
print(f'predicted number of covid cases for day 737733 is:')
print(y_prediction.astype('int'))



#we can also make predictions for a range of future dates
#use a for loop to predict new covid cases for the 100 dates following on from this dataset
predictions = []
final_predictions = []

for i in range(737634, 737734):
    t=(model.predict([[i]])).astype(int)
    predictions.append(t)
    final_predictions = str(predictions[-100:])
    

print(f'The predicted number of new cases for the following 100 days are:')
print(final_predictions)
