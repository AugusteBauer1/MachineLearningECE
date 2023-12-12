import numpy as np
import matplotlib.pyplot as plt

def multivariate_normal_pdf(x, mu, sigma):
    """
    Compute the multivariate normal probability density function.
    """
    d = x.shape[0]
    sqrt_det_sigma = np.sqrt(np.linalg.det(sigma))
    inv_sigma = np.linalg.inv(sigma)
    norm_const = 1.0 / ( np.power((2*np.pi), float(d)/2) * sqrt_det_sigma )
    x_mu = np.matrix(x - mu)
    result = np.exp(-0.5 * (x_mu * inv_sigma * x_mu.T))
    return norm_const * result

def E_step(X, phi, mu, sigma):
    """
    Estimate the membership of each data point to each cluster.
    """
    N = X.shape[0]  
    K = mu.shape[0]  
    W = np.zeros((N, K))

    for i in range(N):
        total_prob_X = 0 

        for k in range(K):
            total_prob_X += phi[k] * multivariate_normal_pdf(X[i], mu[k], sigma[k])

        for k in range(K):
            prior = phi[k]
            
            likelihood = multivariate_normal_pdf(X[i], mu[k], sigma[k])

            W[i, k] = prior * likelihood / total_prob_X

    return W

def M_step(X, W):
    """
    Find tetha that maximizes the expected log-likelihood function.
    """
    N, D = X.shape  
    K = 2  
    phi = np.zeros(K)  
    mu = np.zeros((K, D))  
    sigma = np.zeros((K, D, D)) 


    # First, update mixture coefficients (phi)
    N_k = np.sum(W, axis=0)  
    phi = N_k / N

    # Next, update means (mu)
    for k in range(K):
        mu[k] = np.sum(W[:, k].reshape(N, 1) * X, axis=0) / N_k[k]

    # Finally, update covariance matrices (sigma)
    for k in range(K):
        for i in range(N):
            X_centered = (X[i] - mu[k]).reshape(D, 1)
            sigma[k] += W[i, k] * np.dot(X_centered, X_centered.T)
        sigma[k] /= N_k[k]

    return phi, mu, sigma

def fit_GMM(X, K, max_iter):
    """
    Implement the Gaussian Mixtures algorithm from scratch.

    params:
    X: input data
    K: number of clusters
    max_iter: maximum number of iterations
    """

    # for all k = 1, ..., K, initialize phi_k
    phi = np.ones(K) / K

    # initialize mu_k for randomly from the data points
    mu = X[np.random.choice(X.shape[0], K, replace=False)]

    # initialize sigma_k
    sigma = np.array([np.eye(X.shape[1])] * K)

    # repeat until convergence
    for _ in range(max_iter):
        # E-step
        W = E_step(X, phi, mu, sigma)
        # M-step
        phi, mu, sigma = M_step(X, W)


    return phi, mu, sigma, W

def predict_GMM(W):
    # find the cluster assignment for each data point
    y_pred = np.argmax(W, axis=1)

    return y_pred



if __name__ == "__main__":
    n_samples = 300
    # generate random sample, two components
    np.random.seed(0)
    # generate spherical data centered on (20, 20)
    shifted_gaussian = np.random.randn(n_samples, 2) + np.array([4, 4])
    # generate zero centered stretched Gaussian data
    C = np.array([[0., -0.7], [3.5, .7]])
    stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
    # concatenate the two datasets into the final training set
    X_train = np.vstack([shifted_gaussian, stretched_gaussian])

    # fit the data to the model
    phi, mu, sigma, w = fit_GMM(X_train, 2, 100)

    # predict the cluster assignment
    y_pred = predict_GMM(w)
    print("optimal weights: ", w)
    # plot the points in 2D 
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred, s=40, cmap='viridis')
    plt.title("Gaussian Mixture Model Clustering")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.show()
