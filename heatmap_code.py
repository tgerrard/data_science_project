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


###Now we can take a second dataset from the same source to look at this data related back to geographical data###
df = pd.read_csv("country_wise_latest.csv")
print(df)

# Define a dictionary containing data
country_data = df['Country/Region']
# Observe the result
print(country_data)

#use geopython to get longitude and lattitude values for the list of countries/regions
# declare an empty list to store latitude and longitude of values of country/region collumn
longitude = []
latitude = []

# function to find the coordinate of a given country
def findGeocode(country):

# try and catch is used to overcome the exception thrown by geolocator using geocodertimedout
    try:
# Specify the user_agent as your app name it should not be none
        geolocator = Nominatim(user_agent="your_app_name")

        return geolocator.geocode(country)

    except GeocoderTimedOut:

        return findGeocode(country)	

# each value from country column will be fetched and sent to function find_geocode
for i in country_data:

    if findGeocode(i) != None:

        loc = findGeocode(i)

# coordinates returned from function is stored into two separate list
        latitude.append(loc.latitude)
        longitude.append(loc.longitude)

# if coordinate for a city not found, insert "NaN" indicating missing value
    else:
        latitude.append(np.nan)
        longitude.append(np.nan)

#print values to see if they have benn found and transcribed correctly
print(longitude)


df["Longitude"] = longitude
df["Latitude"] = latitude

#scatterplot of logitude against deaths, this doesnt show us a lot!
sns.scatterplot(data=df, x='Longitude', y='Deaths')

#If we plot logitude against latitude coloured by Deaths we start to see the world map but it could be clearer (figure 3)
df.plot(x="Longitude", y="Latitude", kind="scatter", c="Deaths",
        colormap="YlOrRd")

#create the blank world map (figure4)
countries = gpd.read_file(
               gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color="lightgrey")

##plot our data onto this map
# initialize an axis
fig, ax = plt.subplots(figsize=(8,6))

# plot map on axis
countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color="lightgrey", ax=ax)

# plot points
df.plot(x="Longitude", y="Latitude", kind="scatter", c="Deaths", colormap="YlOrRd", 
        title="COVID deaths", ax=ax)

# add grid
ax.grid(visible=True, alpha=0.5)

#figure5
plt.show()
