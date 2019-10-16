import numpy as np
from numpy.linalg import cholesky
import json
from sampling import gmm_em
import matplotlib.pyplot as plt


def sample_from_gaussian(mu, sigma, sample_No=1000):
    """
    Sampling from a Gaussian model
    :param mu: the mean of gaussian model
    :param sigma: the variance of gaussian model
    :param sample_No: the number of points that will sample from gaussian model
    :return: the sampled points
    """
    R = cholesky(sigma)
    s = np.dot(np.random.randn(sample_No, 2), R) + mu
    return s


def sample_from_uniform(alpha_vec):
    """
    sampling from a uniform
    :param alpha_vec: the probability each gaussian model being selected
    :return: the index of selected gaussian model
    """
    x = np.random.rand()
    total = 0
    index = -1
    for ids, a in enumerate(alpha_vec):
        if x >= total and (x < (total + a)):
            index = ids
            break
        else:
            total += a
    return index


def sample_from_mixture_gaussian(alpha, mu, sigma, sample_No=1000):
    """
    sampling from gaussian mixture model
    :param alpha: the weights of each gaussian model
    :param mu: the mean of each gaussian model
    :param sigma: the variance of each gaussian model
    :param sample_No: the number of points that will sample from gaussian mixture model
    :return: the sampled points
    """
    points = []
    for ids in range(sample_No):
        gaussian_ids = sample_from_uniform(alpha)
        p = sample_from_gaussian(mu[gaussian_ids], sigma[gaussian_ids], sample_No=1)
        points.append(p[0])
    points = np.round(points)
    # If the coordinate is repeatedly selected, then it is rejected.
    points = list(set([tuple(t) for t in points]))
    return np.array(points)


def plot_object_distribution(points, image):
    """
    plot the location distribution of objects on the image
    :param points: objects' center coordinates
    :param image: background image
    :return:
    """
    plt.scatter(points[:, 0], points[:, 1])
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    f = open("../annotations/instances_all.json")
    data = json.load(f)
    annotations = data["annotations"]
    centers = []
    for ann in annotations:
        bbox = ann["bbox"]
        y = bbox[0] + bbox[2] / 2
        x = bbox[1] + bbox[3] / 2
        centers.append([x, y])
    print(centers)
    centers = np.asarray(centers)
    image = plt.imread("../images/01123.jpg")
    plot_object_distribution(centers, image)
    gmm = gmm_em.GMM(centers)
    gmm.em_algorithm(100, 0.0001)
    print(gmm.mu)
    print(gmm.sigma)
    print(gmm.alpha)
    gmm_em.plot_gmm(gmm)
    alpha = np.array(gmm.alpha)
    mu = np.array(gmm.mu)
    sigma = np.array(gmm.sigma)
    points = sample_from_mixture_gaussian(alpha, mu, sigma, sample_No=100)
    np.savetxt("../locations/sample_points.txt", points, fmt="%d")
    print(points)


