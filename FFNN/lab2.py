import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Load the data
name_file = './data_ffnn_3classes.txt'

#load dataset from file starting from the second line
data = np.loadtxt(name_file, skiprows=1)

x = data[:, :-1]
y = data[:, -1]

Y = np.zeros((y.shape[0],3))
for i in range(y.shape[0]):
    Y[i,int(y[i])] = 1

k = 4

#implement the function initializeRandomWeights with bias
def initializeRandomWeights(inputSize, outputSize, k):
    v = np.random.rand(inputSize + 1, k)
    w = np.random.rand(k + 1, outputSize)
    return v, w

# initialize empty list to store errors
all_errors = [[] for _ in range(10)]

# repeat the while loop 10 times
for i in range(10):

    v, w = initializeRandomWeights(x.shape[1], Y.shape[1], k)
    alpha1 = 0.001
    alpha2 = 0.001

    itera = 0

    epsilon = 0.001
    E = 10000
    past_E = 10000
    delta_E = 10000

    errors = []

    #add a bais column to x intto a new matrix named "X"
    X_bar = np.ones((x.shape[0], x.shape[1] + 1))
    X_bar[:, 1:] = x

    #print(w)
    #print(v)


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def FWP(X_bar,Y,W,V):
        X_bar_bar = np.matmul(X_bar,V)
        F = sigmoid(X_bar_bar)
        F_bar = np.ones((F.shape[0], F.shape[1] + 1))
        F_bar[:, 1:] = F
        F_bar_bar = np.matmul(F_bar,W)
        G = sigmoid(F_bar_bar)
        E = (1/2) * np.sum((G - Y)**2)
        return E, G, F_bar, F

    def BWP(X_bar,Y,W,V,G,F_bar, F):
        #use alpha1 and alpha2
        Dg = (G - Y)*G*(1 - G)
        delta_W = np.matmul(F_bar.T,Dg)
        #exlude the bias term
        Df = np.matmul(Dg,W[1:,:].T)*F*(1 - F)
        delta_V = np.matmul(X_bar.T,Df)
        W = W - alpha1 * delta_W
        V = V - alpha2 * delta_V
        return W, V


    while abs(delta_E) > epsilon:
        itera += 1
        E, G, F_bar, F = FWP(X_bar,Y,w,v)
        w, v = BWP(X_bar,Y,w,v,G,F_bar, F)
        delta_E = E - past_E
        past_E = E
        #print(abs(delta_E))
        all_errors[i].append(E)
        #print(f"Iteration: {itera}, Cost: {E}, Delta: {delta_E}, w: {w}, v: {v}")

    #print(itera)
    # print(w)
    # print(v)
    #all_errors.append(errors)

    #compute the function to determine Y hat
    def Y_hat(X_bar,W,V):
        X_bar_bar = np.matmul(X_bar,V)
        F = sigmoid(X_bar_bar)
        F_bar = np.ones((F.shape[0], F.shape[1] + 1))
        F_bar[:, 1:] = F
        F_bar_bar = np.matmul(F_bar,W)
        G = sigmoid(F_bar_bar)
        Y_hat = np.argmax(G, axis=1)
        return Y_hat, G
    
    Y_hat, G = Y_hat(X_bar,w,v)

    print(G)

# Find the array with the most iterations
max_iter = max(all_errors, key=len)

# Plot the error with the most iterations
plt.plot(range(1, len(max_iter)+1), max_iter)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'Error with {len(max_iter)} iterations')
plt.show()