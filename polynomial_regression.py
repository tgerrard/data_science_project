import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('covid_data.csv')

# to explicitly convert the date column to type DATETIME(DD-MM-YYYY)
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)

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

print(X)


#Split the dataset into training and test sets 
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

train_X = train_X.values.reshape(-1, 1)

#convert into polynomial
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(train_X)


#print(x_poly)



#reshape our dates collumn so it is 2 dimensional as required by sklearn
#X_poly_reshape_2 = x_poly.reshape(-1, 1)
test_X_reshape = test_X.values.reshape(-1, 1)

#linear regression
model = LinearRegression()

#fit the parameters to the model
#poly.fit(x_poly, train_y)
model.fit(x_poly, train_y) 

#predict the value of corona confirm cases at 220th day
pred = model.predict(poly.fit_transform([[737733]]))
print("The confirm cases on 737733rd is: " + str(int(pred)))


poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(train_X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,train_y)
 
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1) 
plt.scatter(X,y, color='red') 
 
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue') 
 
plt.title("polynomial regression fit")
plt.xlabel('Date')
plt.ylabel('New cases')
plt.show()

