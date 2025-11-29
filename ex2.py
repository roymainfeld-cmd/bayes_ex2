import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        N = x.shape[0]
        H = np.zeros((N, degree + 1))
        for i in range(degree + 1):
            if i == 0:
                H[:, i] = 1
            else:
                H[:, i] = (x / degree) ** i
        return H
    return pbf


def fourier_basis_functions(num_freqs: int) -> Callable:
    """
    Create a function that calculates the fourier basis functions up to a certain frequency
    :param num_freqs: the number of frequencies to use
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Fourier basis functions, a numpy array of shape [N, 2*num_freqs + 1]
    """
    def fbf(x: np.ndarray):
        N = x.shape[0]
        H = np.zeros((N, 2 * num_freqs + 1))
        
        H[:, 0] = 1
        
        period = 24.0
        
        for k in range(1, num_freqs + 1):
            H[:, 2*k - 1] = np.sin(2 * np.pi * k * x / period)
            H[:, 2*k] = np.cos(2 * np.pi * k * x / period)
        
        return H
    return fbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """
    def csbf(x: np.ndarray):
        N = x.shape[0]
        k = len(knots)
        H = np.zeros((N, k + 4))
        
        H[:, 0] = 1
        H[:, 1] = x
        H[:, 2] = x ** 2
        H[:, 3] = x ** 3
        
        for i, knot in enumerate(knots):
            truncated = np.maximum(x - knot, 0)
            H[:, 4 + i] = truncated ** 3
        
        return H
    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior usTing historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        thetas.append(ln.theta)

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig_sq: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig_sq:              the signal noise squared to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.prior_mean = theta_mean
        self.prior_cov = theta_cov
        self.sigma_sq = sig_sq
        self.basis_functions = basis_functions
        self.posterior_mean = theta_mean
        self.posterior_cov = theta_cov

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.basis_functions(X)
        prior_cov_inv = np.linalg.inv(self.prior_cov)
        
        self.posterior_cov = np.linalg.inv(prior_cov_inv + (1 / self.sigma_sq) * H.T @ H)
        self.posterior_mean = self.posterior_cov @ (prior_cov_inv @ self.prior_mean + (1 / self.sigma_sq) * H.T @ y)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.basis_functions(X)
        theta_map = self.posterior_mean
        return H @ theta_map

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        H = self.basis_functions(X)
        variance = np.diagonal(H @ self.posterior_cov @ H.T) + self.sigma_sq
        return np.sqrt(variance)

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.basis_functions(X)
        theta_sample = np.random.multivariate_normal(self.posterior_mean, self.posterior_cov)
        return H @ theta_sample


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.basis_functions = basis_functions
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.basis_functions(X)
        self.theta = np.linalg.lstsq(H, y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.basis_functions(X)
        return H @ self.theta

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)


def main():
    # load the data for November 16 2024
    nov16 = np.load('nov162024.npy')
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16)//2]
    train_hours = nov16_hours[:len(nov16)//2]
    test = nov16[len(nov16)//2:]
    test_hours = nov16_hours[len(nov16)//2:]

    # setup the model parameters
    degrees = [3, 7]

    # ----------------------------------------- Classical Linear Regression
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)

        print(f'Average squared error with LR and d={d} is {np.mean((test - ln.predict(test_hours))**2):.2f}')

        plt.figure(figsize=(10, 6))
        plt.scatter(train_hours, train, color='green', label='train data', alpha=0.6, s=30)
        plt.scatter(test_hours, test, color='blue', label='test data', alpha=0.6, s=30)
        
        x_plot = np.arange(0, 24, 0.1)
        y_plot = ln.predict(x_plot)
        plt.plot(x_plot, y_plot, color='red', linewidth=2, label=f'prediction')
        
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.title(f'Linear Regression (degree {d})')
        plt.legend()
        plt.show()

    # ----------------------------------------- Bayesian Linear Regression

    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma_sq = 0.25
    degrees = [3, 7]  # polynomial basis functions degrees

    # frequencies for Fourier basis
    freqs = [1, 2, 3]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)

        blr = BayesianLinearRegression(mu, cov, sigma_sq, pbf)

        # plot prior graphs
        plt.figure()
        H = pbf(x)
        mean_prior = H @ mu
        std_prior = np.sqrt(np.diagonal(H @ cov @ H.T))
        
        plt.fill_between(x, mean_prior - std_prior, mean_prior + std_prior, alpha=.5, label='confidence interval')
        for i in range(5):
            theta_sample = np.random.multivariate_normal(mu, cov)
            y_sample = H @ theta_sample
            plt.plot(x, y_sample)
        plt.plot(x, mean_prior, 'k', lw=2, label='mean')
        plt.legend()
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.title(f'Prior (degree {deg})')
        plt.show()

        # plot posterior graphs
        blr.fit(train_hours, train)
        
        test_pred = blr.predict(test_hours)
        
        mse = np.mean((test - test_pred)**2)
        print(f'Average squared error with BLR and d={deg} is {mse:.2f}')
        
        plt.figure(figsize=(10, 6))
        
        plt.scatter(train_hours, train, color='green', label='train data', alpha=0.6, s=30)
        plt.scatter(test_hours, test, color='blue', label='test data', alpha=0.6, s=30)
        
        mmse_pred = blr.predict(x)
        
        std_posterior = blr.predict_std(x)
        
        plt.fill_between(x, mmse_pred - std_posterior, mmse_pred + std_posterior, 
                         alpha=0.3, color='orange', label='posterior std')
        
        for i in range(5):
            y_sample = blr.posterior_sample(x)
            plt.plot(x, y_sample, alpha=0.5, linewidth=1)
        
        plt.plot(x, mmse_pred, 'r', linewidth=2, label='MMSE prediction')
        
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.title(f'Posterior (degree {deg}) - MSE: {mse:.2f}')
        plt.legend()
        plt.show()

    # ---------------------- Fourier basis functions
    for ind, K in enumerate(freqs):
        fbf = fourier_basis_functions(K)
        mu, cov = learn_prior(hours, temps, fbf)

        blr = BayesianLinearRegression(mu, cov, sigma_sq, fbf)

        # plot prior graphs
        plt.figure()
        H = fbf(x)
        mean_prior = H @ mu
        std_prior = np.sqrt(np.diagonal(H @ cov @ H.T))
        
        plt.fill_between(x, mean_prior - std_prior, mean_prior + std_prior, alpha=.5, label='confidence interval')
        for i in range(5):
            theta_sample = np.random.multivariate_normal(mu, cov)
            y_sample = H @ theta_sample
            plt.plot(x, y_sample)
        plt.plot(x, mean_prior, 'k', lw=2, label='mean')
        plt.legend()
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.title(f'Prior (Fourier K={K})')
        plt.show()

        # plot posterior graphs
        blr.fit(train_hours, train)
        
        test_pred = blr.predict(test_hours)
        
        mse = np.mean((test - test_pred)**2)
        print(f'Average squared error with BLR and Fourier K={K} is {mse:.2f}')
        
        plt.figure(figsize=(10, 6))
        
        plt.scatter(train_hours, train, color='green', label='train data', alpha=0.6, s=30)
        plt.scatter(test_hours, test, color='blue', label='test data', alpha=0.6, s=30)
        
        mmse_pred = blr.predict(x)
        
        std_posterior = blr.predict_std(x)
        
        plt.fill_between(x, mmse_pred - std_posterior, mmse_pred + std_posterior, 
                         alpha=0.3, color='orange', label='posterior std')
        
        for i in range(5):
            y_sample = blr.posterior_sample(x)
            plt.plot(x, y_sample, alpha=0.5, linewidth=1)
        
        plt.plot(x, mmse_pred, 'r', linewidth=2, label='MMSE prediction')
        
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.title(f'Posterior (Fourier K={K}) - MSE: {mse:.2f}')
        plt.legend()
        plt.show()

    # ---------------------- cubic regression splines
    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        blr = BayesianLinearRegression(mu, cov, sigma_sq, spline)

        # plot prior graphs
        plt.figure()
        H = spline(x)
        mean_prior = H @ mu
        std_prior = np.sqrt(np.diagonal(H @ cov @ H.T))
        
        plt.fill_between(x, mean_prior - std_prior, mean_prior + std_prior, alpha=.5, label='confidence interval')
        for i in range(5):
            theta_sample = np.random.multivariate_normal(mu, cov)
            y_sample = H @ theta_sample
            plt.plot(x, y_sample)
        plt.plot(x, mean_prior, 'k', lw=2, label='mean')
        plt.legend()
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        knots_str = ','.join(map(str, k))
        plt.title(f'Prior (Spline K_{ind+1}, knots=[{knots_str}])')
        plt.show()

        # plot posterior graphs
        blr.fit(train_hours, train)
        
        test_pred = blr.predict(test_hours)
        
        mse = np.mean((test - test_pred)**2)
        print(f'Average squared error with BLR and Spline K_{ind+1} (knots=[{knots_str}]) is {mse:.2f}')
        
        plt.figure(figsize=(10, 6))
        
        plt.scatter(train_hours, train, color='green', label='train data', alpha=0.6, s=30)
        plt.scatter(test_hours, test, color='blue', label='test data', alpha=0.6, s=30)
        
        mmse_pred = blr.predict(x)
        
        std_posterior = blr.predict_std(x)
        
        plt.fill_between(x, mmse_pred - std_posterior, mmse_pred + std_posterior, 
                         alpha=0.3, color='orange', label='posterior std')
        
        for i in range(5):
            y_sample = blr.posterior_sample(x)
            plt.plot(x, y_sample, alpha=0.5, linewidth=1)
        
        plt.plot(x, mmse_pred, 'r', linewidth=2, label='MMSE prediction')
        
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.title(f'Posterior (Spline K_{ind+1}, knots=[{knots_str}]) - MSE: {mse:.2f}')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
