import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import where
from pylab import scatter, show, legend, xlabel, ylabel

#columns that is going to readed into csv
columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
data = pd.read_csv('iris.csv', usecols=[1, 2, 3, 4, 5])

# the data comes from csv file stored in xTemp
xTemp = data[columns]# read the related columns
xTemp = np.array(xTemp)
X = []
# To append 1 to the each element in the xTemp to fit the hypothesis
for i in range(len(xTemp)):
    tmp = np.array([1])
    tmpArray = np.concatenate(([tmp, xTemp[i]]), axis=0)
    X.append(tmpArray)

X = np.array(X) #X is converted into numpy array which will be helpful for calculating.

target_map = {'Iris-setosa':0,
              'Iris-versicolor':1}

data[["Species"]] = data['Species'].apply(lambda x: target_map[x])
Y = np.array(data[["Species"]]) # Iris-setosa converted to 0, Iris-versicolor converted to 1

# test data is reading
testData = pd.read_csv('iristest2.csv', usecols=[1, 2, 3, 4, 5])
xTestTemp = testData[columns]# read the related columns
xTestTemp = np.array(xTestTemp)
X_test = []

# To append 1 to the each element in the to fit the hypothesis
for i in range(len(xTestTemp)):
    tmp = np.array([1])
    tmpArray = np.concatenate(([tmp, xTestTemp[i]]), axis=0)
    X_test.append(tmpArray)

X_test = np.array(X_test) #X_test is converted into numpy array.

testData[["Species"]] = testData['Species'].apply(lambda x: target_map[x])
Y_test = np.array(testData[["Species"]])# test data is inserted into Y_test


 
theta = [0, 0, 0, 0, 0] # declare the initial theta
iterations = 1000 # iteration
alpha = 0.3 #learning rate


# @param z : applies sigmoid function to given parameter
# returns : a number between 0-1
def Sigmoid(z):
    return float(1.0 / float((1.0 + np.exp(-1.0 * z))))

# @param coefficients @type ndarray 
# @returns value that applied sigmoid function in given feature vector.
def Hypothesis(coefficients):
    z = 0
    for i in range(len(theta)):
        z += coefficients[i] * theta[i]
    return Sigmoid(z.astype(np.float128))

# cost function calculated for n observations which is length of Y
# function {i=1:n}Σ -yilog(h(xi))- (1-yi)log(1-h(xi))
def CostFunction():
    sumOfErrors = 0
    for i in range(len(Y)):
        if Y[i] == 1:
            error = Y[i] * np.log(Hypothesis(X[i]))
        elif Y[i] == 0:
            error = (1 - Y[i]) * np.log(1 - Hypothesis(X[i]))
        sumOfErrors += error
    return (-1 / len(Y)) * sumOfErrors


# @param j : theta index that will be calculated
# @returns the new value for theta[j]
# function βj = βj- a{i=1:n}Σ(h(xi)-yi)xij
# where β is the theta array , a = learning rate,  xi = feature vector, xij = β value for xi
# yi = label of the feature vector
def CostDerivative(j):
    sumErrors = 0
    n = len(Y)  
    for i in range(n):
        xi = X[i]
        xij = xi[j]
        hi = Hypothesis(xi)
        error = (hi - Y[i]) * xij
        sumErrors += error
    c = float(alpha) / float(n)    
    theta[j] = theta[j] - (sumErrors *c)
    return theta[j]


# With GradientDescent, iterates all of the theta values, calls @func CostDerivative for each element in theta
# updates the new value.
def GradientDescent():
    new_theta = []
    for i in range(len(theta)):
        new_theta.append(CostDerivative(i))
    return new_theta
    
# calculates the regression coefficients in given feature vectors.
def estimate_coef(x,y):
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1)
    
# plots the regression line with given feature vectors x and y     
def plot_regression_line(x, y, b): 
    versicolor = np.where(Y == 1)
    setosa = np.where(Y == 0)

    plt.figure()
    # plotting the actual points as scatter plot 
    plt.scatter(X[versicolor[0], 1], X[versicolor[0], 2], color= 'r', label="Versicolor") # versicolor
    plt.scatter(X[setosa[0], 3], X[setosa[0], 4], color = 'b',label="Setosa") # setosa
    
    #plt.scatter(x, y, color=['red'],  label="Versicolor") # versicolor    
    #plt.plot(x,color='red', label= 'Sepal Area')
    #plt.plot(y,color='blue', label='Petal Area')
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "k", label= "Regression Line") 
  
    # putting labels 
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    # function to show plot 
    plt.show() 


# Predicts a number with trained theta coefficients
def Predict():
    score = 0
    length = len(X_test)
    for i in range(length):
        prediction = round(Hypothesis(X_test[i]))
        answer = Y_test[i]
        print("prediction was " + str(prediction) + " answer was " + str(answer))
        if prediction == answer:
            score += 1

    accuracy = ((score / length) * 100)
    print("score is = " +str(score) + " out of " + str(length) + " samples")
    print('accuracy is =  {} %.'.format(accuracy))


def LogisticRegression():

    for i in range(iterations):
        new_theta = GradientDescent()
        theta = new_theta
        if i % 100 == 0:
            print('Cost function : {:.10f}'.format(CostFunction()[0]))
    print("theta: ", theta)
    Predict()

    #used for estimating the regression lines between the features.
    
    # b = estimate_coef(X[:,1], X[:,2])
    # plot_regression_line(X[:,1], X[:,2], b)
    # b = estimate_coef(X[:,1], X[:,3])
    # plot_regression_line(X[:,1], X[:,3], b)
    # b = estimate_coef(X[:,1], X[:,4])
    # plot_regression_line(X[:,1], X[:,4], b)
    # b = estimate_coef(X[:,2], X[:,3])
    # plot_regression_line(X[:,2], X[:,3], b)
    # b = estimate_coef(X[:,2], X[:,4])
    # plot_regression_line(X[:,2], X[:,4], b)


LogisticRegression()
