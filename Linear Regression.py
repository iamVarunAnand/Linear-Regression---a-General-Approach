import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import metrics

trainFileName = ## Enter path to CSV data file - eg: "D:\\Datasets\\housingPrices.csv"
trainData = pd.read_csv(trainFileName) 

numberIterations = 100

def setup(): ## function to setup dependent and independent variables - refer to README.txt
    
    ## Initialising ndarray with input values, and adding bias column
    X = np.vstack(trainData.values[:,:-1])    
    X = normalize(X)  # Normalizing Inputs - comment out for Simple Linear Regression
    X = np.append(np.ones(shape=(X.shape[0], 1), dtype=np.float64), values=X, axis=1)

    ## Initialising ndarray with output values
    Y = np.vstack(trainData.values[:,-1])

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=100)

    return X_train, Y_train, X_test, Y_test

def normalize(X): ## function to normlize the inputs

    xMean = np.mean(X, axis=0)
    xStd = np.std(X, axis=0)

    X = (X - xMean) / xStd

    return X

def calcGradient(H_theta, X, Y, M, N):  ## function to calculate gradient at given theta
    gradient = np.zeros(shape = (N, 1), dtype = np.float64)

    E_theta = H_theta - Y
    gradient = np.dot(X.transpose(), E_theta)
    gradient = (1 / M) * gradient   

    return gradient

def calcCost(H_theta, Y, M):  ## function to calculate cost of given hypothesis

    E_theta = H_theta - Y  ## Error in hypothesis
    J_theta = np.nansum(pow(E_theta, 2))  ## Squared Error
    J_theta = (1 / (2 * M)) * J_theta  ## Averaged Square Error
    
    return J_theta

def linearRegression(X, Y, theta, alpha, M, N):  ## function to perform Linear Regression

    costHistory = np.zeros(shape=(numberIterations, 1), dtype=np.float64)
    
    for i in range(numberIterations):
        H_theta = np.dot(X, theta)  ## Hypothesis: H(theta)        
        gradient_theta = calcGradient(H_theta, X, Y, M, N)  ## Calculates gradient of cost function at given theta
        theta = theta - alpha * gradient_theta        
        costHistory[i] = calcCost(H_theta, Y, M) ## Calculates cost for given hypothesis

    H_theta = np.dot(X, theta)  # Final hypothesis: H(theta)
    return theta, costHistory

def main():  ## Driver function
    
    X_train, Y_train, X_test, Y_test = setup()
    
    M = X_train.shape[0]  # Number of training examples
    N = X_train.shape[1]  ## Number of attributes    
    alpha = 0.67  ## Learning Rate - Adjust according to data set
    theta = np.random.rand(N, 1)  ## Weight matrix  

    theta, costHistory = linearRegression(X_train, Y_train, theta, alpha, M, N)

    Y_pred   = np.dot(X_test, theta)
    rSquared = metrics.r2_score(Y_test, Y_pred)

    print(f"Final Cost = {costHistory[numberIterations-1]}")
    print(f"Theta = {theta.transpose()}", end='\t')
    print(f"rSquared = {rSquared}")    

    ## ---------------------------------------------------------------------------
    ## Cost Function mapping
    x = np.linspace(1, numberIterations, numberIterations)
    plt.plot(x, costHistory)
    plt.show()
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    ## ---------------------------------------------------------------------------   
             
main()

