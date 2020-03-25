import math

import pandas as pd
import numpy as np
from numpy import loadtxt, where
from matplotlib import pylab
from pylab import scatter, show, legend, xlabel, ylabel


def loadData(filename, columns):
    arr = []
    data = pd.read_csv(filename, )
    arr = data[columns]
    return np.array(arr)


columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

data = pd.read_csv('iris.csv', usecols=[1, 2, 3, 4, 5])
X = data[columns]
X = np.array(X)
#print(X)

#print(len(X))
Y = data[["Species"]]
Y = np.array(Y)
for i in range(len(Y)):
    if Y[i] == "Iris-setosa":
        Y[i] = 1
    else:
        Y[i] = 0

pos = where(Y == 1)
neg = where(Y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not Admitted', 'Admitted'])
show()

testdata = pd.read_csv('iristest.csv', usecols=[1, 2, 3, 4, 5])
X_test = testdata[columns]
X_test = np.array(X_test)
print(X_test)
Y_test = testdata[["Species"]]
Y_test = np.array(Y_test)
for i in range(len(Y_test)):
    if Y_test[i] == "Iris-setosa":
        Y_test[i] = 1
    else:
        Y_test[i] = 0

print(Y_test)


def Sigmoid(z):
    return float(1.0 / float((1.0 + np.exp(-1.0 * z))))


def Hypothesis(theta, x):
    z = 0
    for i in range(len(theta)):
        z += x[i] * theta[i]
    return Sigmoid(z)


def Cost_Function_Derivative(X, Y, theta, j, m, alpha):
    sumErrors = 0
    for i in range(m):
        xi = X[i]
        xij = xi[j]
        hi = Hypothesis(theta, X[i])
        error = (hi - Y[i] * xij)
        sumErrors += error
    m = len(Y)
    cons = float(alpha) / float(m)
    J = cons * sumErrors
    return j


def Cost_Function(X, Y, theta, m):
    sumErrors = 0
    for i in range(m):
        xi = X[i]
        hi = Hypothesis(theta, xi)
        if Y[i] == 1:
            error = Y[i] * math.log1p(hi)
        elif Y[i] == 0:
            error = (1 - Y[i]) * math.log1p(1 - hi)
        sumErrors += error
    const = -1 / m
    J = const * sumErrors
    print('const func cost is ', J)
    return J


def Gradient_Descent(X, Y, theta, m, alpha):
    new_theta = []
    for j in range(len(theta)):
        CostDerivative = Cost_Function_Derivative(X, Y, theta, j, m, alpha)
        new_theta_value = theta[j] - CostDerivative
        new_theta.append(new_theta_value)
    return new_theta


def Predict(theta):
    score = 0
    length = len(X_test)
    for i in range(length):
        prediction = round(Hypothesis(X_test[i], theta))
        answer = Y_test[i]
        print("prediction was " + str(prediction) + " answer was " + str(answer))
        if prediction == answer:
            score += 1
    print("score is = ", score)


def Logistic_Regression(X, Y, alpha, theta, iterations):
    m = len(Y)
    for x in range(iterations):
        new_theta = Gradient_Descent(X, Y, theta, m, alpha)
        theta = new_theta
        if x % 100 == 0:
            Cost_Function(X, Y, theta, m)
            print('theta', theta)
            print('cost is ', Cost_Function(X, Y, theta, m))
            print("----------")
    Predict(theta)


initial_theta = [0, 0, 0, 0]
alpha = 0.1
iterations = 1000

Logistic_Regression(X, Y, alpha, initial_theta, iterations)
