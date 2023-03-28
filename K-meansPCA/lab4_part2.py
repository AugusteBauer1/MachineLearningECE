import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.loadtxt(filename)
    return data


def PCA(X, P_largest):
    # Compute the mean value
    mean = np.mean(X, axis=1)
    
    # Create x_tilde by subtracting the mean from each column
    X_tilda = X - mean[:, np.newaxis]

    # Compute the covariance matrix with a for loop
    sigma = np.zeros((X_tilda.shape[0], X_tilda.shape[0]))
    for i in range(X_tilda.shape[0]):
        for j in range(X_tilda.shape[0]):
            sigma[i, j] = np.dot(X_tilda[i], X_tilda[j]) / X_tilda.shape[1]

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    sigmaEigenvalues, sigmaEigenvectors = np.linalg.eig(sigma)

    #select the P_largest eigenvectors without sorting
    sigmaEigenvectors = sigmaEigenvectors[:, :P_largest]

    # project X_tilda onto the eigenvectors
    Y_tilda = np.dot(X_tilda.T, sigmaEigenvectors)

    return Y_tilda, sigmaEigenvectors.flatten(), mean
    

data = load_data('./data_pca.txt')

X = data.T

Y_tilda, principalAxes, mu = PCA(X,1)

data = np.array([[0.0, 0.0],
                 [1.0, 1.5],
                 [2.1, 2.0],
                 [2.7, 3.1],
                 [4.0, 4.0],
                 [5.0, 5.0],
                 [6.0, 6.0],
                 [7.0, 7.0],
                 [8.0, 8.0],
                 [9.0, 9.0]])

test_data = data.T
Y_test_tilda, testPrincipalAxes, mean = PCA(test_data, 1)


plt.figure("training data")
# Plot training data
plt.scatter(X[0], X[1], color='blue', label='Training data')

# Add mean to plot
plt.scatter(mu[0], mu[1], color='black', label='Mean')

# Plot principal axis
plt.arrow(mu[0], mu[1], principalAxes[0], principalAxes[1], color='green', width=0.05, head_width=0.3, length_includes_head=True)

# Plot projected data
plt.scatter(mu[0] + Y_tilda[:, 0] * principalAxes[0], mu[1] + Y_tilda[:, 0] * principalAxes[1], color='red', label='Projected data')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.figure("test data")
# Plot training data
plt.scatter(test_data[0], test_data[1], color='blue', label='Testing data')

# Add mean to plot
plt.scatter(mean[0], mean[1], color='black', label='Mean')

# Plot principal axis
plt.arrow(mean[0], mean[1], principalAxes[0], principalAxes[1], color='green', width=0.05, head_width=0.3, length_includes_head=True)

# Plot projected data
plt.scatter(mean[0] + Y_test_tilda[:, 0] * testPrincipalAxes[0], mean[1] + Y_test_tilda[:, 0] * testPrincipalAxes[1], color='red', label='Projected data')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
    
    


