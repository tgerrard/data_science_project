The script COVID_data_analysis_heatmap.py requires two .csv files to be present in the same folder; 
1) covid_data.csv
    This file contains covid data for a series of dates from January to July 2020
2) country_wise_latest.csv
    This file contains covid data related to countries and regions aorund the world 
    
To run the script call python COVID_data_analysis_heatmap.py in the terminal and the script will run. Sometimes there have been problems running this as one script so i have also provided the script split inot its three main parts: model_fit_code.py, polynomial_regression.py to train and test the linear and polynomial models and heatmap_code.py to ceate the heatmap. Code will not continue to run until the output figure has been viewed and window closed.

In order to run heatmap_code.py geopython must be installed 

The following will be returned in the terminal window:
model_fit_code.py:
1) A summary of the data from covid_data.csv
2) A list of ordinal dates once they have been converted from date-time format
3) The model score for the linear model which has been trained to fit the data
4) The predicted number of new cases for the ordinal date 100 days in the 'future' past this data set
5) A list of predictions for new cases for the 100 days follwoing on from this dataset

polynomial_regression.py
6) A prediction for the new cases 100 days into the 'future'

The follwoing plots will be produced in a separate window:
model_fit_code.py
1) Plot of the split of training and test data split for new cases over time

polynomial_regression.py
2) scatter plot with the polynomial curve fit to the data of new cases vs date

heatmap_code.py:
3) Heatmap of logitude vs lattitude coloured by covid deaths
4) Blank world map
5) Heatmap of covid deaths overlayed onto world map


NB a series of error messages are produced after plotting of the second figure. It is unclear what they refer to as the line numbers and functions do not correspond to the script produced. The outputs are still generated as expected after these messages. 

    
    