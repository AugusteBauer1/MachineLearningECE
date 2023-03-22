import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

name_file = './data_lab1.txt' 

columns = ['x','y']
data_in = pd.read_csv(name_file, 
                      names=columns,
                      sep=' ')

data_in.plot(kind='scatter',x='x',y='y',color='red')

x = np.asarray(data_in['x'])
y = np.asarray(data_in['y'])

# 2. Divide the data into training and test data
split_idx = int(0.7 * data_in.shape[0])
X_train, Y_train = data_in.iloc[:split_idx, :-1], data_in.iloc[:split_idx, -1]
X_test, Y_test = data_in.iloc[split_idx:, :-1], data_in.iloc[split_idx:, -1]

N = 1
theta = np.zeros(X_train.shape[1])
print(theta)
E = 10**3
dE = 10**3
eps = 10**-10 # Tu commenceras avec une valeur plus élevé (10 il me semble). Cpt plus tu mets un epsilon faible, plus ton apprentissage est précis
alpha = 10**-2 # or 10**-3
iter = 0


#You'll get algo during the class

while abs(dE) > eps :

    iter += 1
    past_E = E
    #BGD
    for n in range (N+1) :
        h = X_train @ theta
        error = h - Y_train
        delta = (1/len(Y_train)) * (X_train.T @ error)
        theta = theta - alpha * delta

    E = (1/(2*len(Y_train))) * np.sum((X_train @ theta - Y_train)**2)
    dE = E - past_E

# Output the optimal values of the parameters
print(f'Optimal values of the parameters: {theta}')

# Predict on test data
Y_pred = np.dot(X_test, theta)

# Calculate Mean Squared Error (MSE) for the model
mse = np.mean((Y_pred - Y_test) ** 2)

print("Mean Squared Error (MSE) for test data: ", mse)
print(theta)


# Plot the training data and regression line
plt.figure(2)
plt.plot(X_train.values, Y_train.values, 'ro')
plt.plot(X_train.values, np.dot(X_train, theta), '-b')

# Plot the test data and regression line
plt.figure(3)
plt.plot(X_test.values, Y_test.values, 'ro')
plt.plot(X_test.values, np.dot(X_test, theta), '-b')

plt.show()