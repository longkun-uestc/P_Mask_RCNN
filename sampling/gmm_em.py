import numpy as np
import math
import copy
from sampling import config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class GMM:
    """
    Encapsulates the Gaussian mixture model functionality.
    """
    def __init__(self, X):
        """
        :param X: the dataset to be used to build GMM
        The initial parameters of the model are provided by config
        """
        self.K = config.K
        self.X = X
        self.sample_num = X.shape[0]
        self.mu = config.mu_initial
        self.sigma = config.sigma_initial
        self.alpha = config.alpha_initial
        self.E = np.zeros((self.sample_num, config.K))

    def e_step(self):
        """
        the E step for EM algorithm
        """
        print("e_step")
        for i in range(self.sample_num):
            q_total = 0
            for j in range(self.K):
                q_total += self.alpha[j] * math.exp(
                    -0.5 * (self.X[i] - self.mu[j]) * np.linalg.inv(self.sigma[j]) * (self.X[i] - self.mu[j]).T) \
                           / np.sqrt(np.linalg.det(self.sigma[j]))
            for j in range(self.K):
                q_j = self.alpha[j] * math.exp(
                    -0.5 * (self.X[i] - self.mu[j]) * np.linalg.inv(self.sigma[j]) * (self.X[i] - self.mu[j]).T) \
                      / np.sqrt(np.linalg.det(self.sigma[j]))
                self.E[i, j] = q_j / q_total

    def m_step(self):
        """
        the M step for EM algorithm
        """
        print("m_step")
        # self.alpha = np.sum(self.E, axis=0)/self.sample_num
        for j in range(self.K):
            deno = 0
            member = 0
            for i in range(self.sample_num):
                member += self.E[i, j] * self.X[i, :]
                deno += self.E[i, j]
            self.alpha[j] = deno / self.sample_num
            self.mu[j, :] = member / deno
            member = 0
            for i in range(self.sample_num):
                member += (self.X[i] - self.mu[j]).T * (self.X[i] - self.mu[j]) * self.E[i, j]
                self.sigma[j] = np.asmatrix(member / deno)

    def em_algorithm(self, iter_num, threshold):
        """
        :param iter_num: the maximum number of iterations
        :param threshold: the termination condition of EM algorithm.When the error is less than the threshold, the EM algorithm stops.
        """
        for i in range(iter_num):
            err_mu = 0
            err_sigma = 0
            err_alpha = 0
            old_mu = copy.deepcopy(self.mu)
            old_sigma = copy.deepcopy(self.sigma)
            old_alpha = copy.deepcopy(self.alpha)
            self.e_step()
            self.m_step()
            for j in range(self.K):
                err_mu += np.sum(old_mu[j] - self.mu[j])
                err_sigma += np.sum(np.sum(old_sigma[j] - self.sigma[j], axis=0), axis=1)
                err_alpha += abs(old_alpha[j] - self.alpha[j])
            if (err_mu <= threshold) and (err_sigma <= threshold) and (err_alpha < threshold):
                print("err_mu: %.2f  err_sigma: %.2f  err_alpha: %.2f" % (err_mu, err_sigma, err_alpha))
                break


class Distribution():
    def __init__(self,mu, Sigma):
        self.mu = mu
        self.sigma = Sigma

    def tow_d_gaussian(self, x):
        mu = self.mu
        Sigma =self.sigma
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n*Sigma_det)
        print(N)
        fac = np.einsum('...k,kl,...l->...',x-mu,Sigma_inv,x-mu)

        return np.exp(-fac/2)/N

    def one_d_gaussian(self,x):
        mu = self.mu
        sigma = self.sigma

        N = np.sqrt(2*np.pi*np.power(sigma,2))
        fac = np.power(x-mu,2)/np.power(sigma,2)
        return np.exp(-fac/2)/N


def plot_gmm(gmm):
    """
    plot the GMM in a 3D coordinate system
    :param gmm: GMM class
    """
    gauss1 = Distribution(mu=np.array(gmm.mu[0]), Sigma=np.asarray(gmm.sigma[0]))
    gauss2 = Distribution(mu=np.array(gmm.mu[1]), Sigma=np.asarray(gmm.sigma[1]))
    N = 256
    X = np.linspace(0, 512, N)
    Y = np.linspace(0, 512, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z1 = gauss1.tow_d_gaussian(pos)
    Z1 = Z1 * 100000
    Z2 = gauss2.tow_d_gaussian(pos)
    Z2 = Z2 * 100000
    Z = gmm.alpha[0] * Z1 + gmm.alpha[1] * Z2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=1, antialiased=True)
    plt.show()