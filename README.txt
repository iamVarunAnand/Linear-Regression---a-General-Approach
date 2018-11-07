This implementation of Linear Regression relies on the following python packages:
1) NumPy
2) Pandas
3) Matplotlib
4) Scikit-learn (metrics)
---------------------------------------------------------------------------------------------------------------------------
Installation of the above mentioned libraries can be done through a terminal / command line using the following commands. 
( This assumes you have a version of python installed in your computer )

1) NumPy - pip install numpy
2) Pandas - pip install pandas
3) Matplotlib - pip install matplotlib
4) Scikit-learn - pip install scikit-learn
---------------------------------------------------------------------------------------------------------------------------
There are 6 major steps in the execution:
1) Importing the dataset into python using pandas
2) Extracting the X and Y arrays from the imported dataset 
3) Normalizing the input space ( incase of multiple linear regression ) and splitting the dataset into train and test data
4) Training the model using Gradient Descent
5) Prediction and Analysis
---------------------------------------------------------------------------------------------------------------------------
EXECUTION:

The only input the library expects is that of the path to the CSV data file. This path needs to be entered as input into the trainFileName variable. The path needs to be enclosed in double quotes (""), and if you are on a windows environment, the backslashes in the path are to be doubled i.e "D:\\Datasets\\housingPrices.csv"

Another key aspect that needs to be set, is the learning rate which is by default set to 0.67. This needs to be adjusted according to the specific dataset you are working on. The number of iterations can also be adjusted.
---------------------------------------------------------------------------------------------------------------------------
OUTPUT:

The program prints out the Final Cost of the algorithm, the trained parameters and the R squared coefficient. It also displays a graphical representation of the changes to the cost over all iterations.
---------------------------------------------------------------------------------------------------------------------------