import pandas as pd
import numpy as np
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

data = pd.read_csv('iris.csv', usecols=[1, 2, 3, 4, 5])

# the data comes from csv file stored in a
a = data[columns]
a = np.array(a)
X = []

# To append 1 to the each element in the a
for i in range(len(a)):
    tmp = np.array([1])
    tmpArray = np.concatenate(([tmp, a[i]]), axis=0)
    X.append(tmpArray)

X = np.array(X)
# print(X)

Y = data[["Species"]]
Y = np.array(Y)
for i in range(len(Y)):
    if Y[i] == "Iris-setosa":
        Y[i] = 1
    else:
        Y[i] = 0
'''
pos = where(Y == 1)
neg = where(Y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
legend(['Iris-Setosa', 'Iris-Verticolor'])
show()

'''

testData = pd.read_csv('iristest.csv', usecols=[1, 2, 3, 4, 5])
b = testData[columns]
b = np.array(b)
X_test = []

# To append 1 to the each element in the a
for i in range(len(b)):
    tmp = np.array([1])
    tmpArray = np.concatenate(([tmp, b[i]]), axis=0)
    X_test.append(tmpArray)

X_test = np.array(X_test)

# print(X_test)
Y_test = testData[["Species"]]
Y_test = np.array(Y_test)
for i in range(len(Y_test)):
    if Y_test[i] == "Iris-setosa":
        Y_test[i] = 1
    else:
        Y_test[i] = 0

# print(Y_test)

theta = [0, 0, 0, 0, 0]
iterations = 1000
alpha = 0.1


def Sigmoid(z):
    return float(1.0 / float((1.0 + np.exp(-1.0 * z))))


def Hypothesis(coefficients):
    z = 0
    for i in range(len(theta)):
        z += coefficients[i] * theta[i]
    return Sigmoid(z.astype(np.float128))


def CostFunction(prediction):
    sumOfErrors = 0
    # cost function calculated for n observations which is length of Y
    for i in range(len(Y)):
        if Y[i] == 1:
            error = Y[i] * np.log(prediction)
        elif Y[i] == 0:
            error = (1 - Y[i]) * np.log(1 - prediction)
        sumOfErrors += error
    return (-1 / len(Y)) * sumOfErrors


# @param j : theta index
def CostDerivative(j):
    sumErrors = 0
    n = len(Y)
    for i in range(n):
        xi = X[i]
        xij = xi[j]
        hi = Hypothesis(xi)
        error = (hi - Y[i]) * xij
        sumErrors += error
    theta[j] = theta[j] - sumErrors
    return theta[j]


def GradientDescent():
    new_theta = []
    for i in range(len(theta)):
        new_theta.append(CostDerivative(i))
    return new_theta


def Predict():
    score = 0
    length = len(X_test)
    for i in range(length):
        # floatPrediction = Hypothesis(X_test[i])
        # print("Float prediction ", floatPrediction)
        prediction = round(Hypothesis(X_test[i]))
        answer = Y_test[i]
        print("prediction was " + str(prediction) + " answer was " + str(answer))
        if prediction == answer:
            score += 1

    accuracy = ((score / len(X_test)) * 100)
    print("score is = ", score)
    print("accuracy is = ", str(accuracy) + " %")


def LogisticRegression(X, Y):
    for i in range(iterations):
        new_theta = GradientDescent()
        theta = new_theta
        if i % 100 == 0:
            print("theta: ", theta)
    Predict()
    # prediction = Hypothesis(X[i])
    # print(CostFunction(prediction))


LogisticRegression(X, Y)
