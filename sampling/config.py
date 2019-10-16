import numpy as np
"""
the initial value to run the EM algorithm to build GMM
K: the number of gaussian models, default 2
mu_initial: the initial mean for each Gaussian model. 
sigma_initial: the initial variance for each Gaussian model. 
alpha_initial: the initial weight for each Gaussian model. 
The initial value should be set appropriately, otherwise it may cause an error.
"""
K = 2
mu_initial = np.matrix([[183, 303], [343, 303]], dtype=np.float32)
sigma_initial = [np.matrix([[100, 0], [0, 100]], dtype=np.float32),
                 np.matrix([[100, 0], [0, 100]], dtype=np.float32)]
alpha_initial = [0.5, 0.5]
