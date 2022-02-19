"""
Step 1:
Generate simulation data
"""

import os
import sys
import numpy as np
from scipy.stats import norm, uniform, bernoulli, invgamma
from sklearn.preprocessing import scale
from scipy.io import savemat

"""
installed all the libraries above
"""

""" initialization """
simu = 200  # number of simulation datasets
l = 5000   # size for coordinates
n = 50  # sample size
m = 2  # number of measurements
p = 3  # number of observed effects
q = 1  # number of unobserved effects
phase = "setting_%s" % sys.argv[1]  # case setting up
np.random.seed(123)

# generate coordinate data
coord_data = uniform.rvs(size=l)
coord_data.sort()

# set up the observed effect size
beta11 = np.reshape(3*coord_data**2, newshape=(1, l))
beta12 = np.reshape(3*(1-coord_data)**2, newshape=(1, l))
beta13 = np.reshape(6*coord_data*(1-coord_data), newshape=(1, l))
beta14 = np.reshape(-coord_data**2, newshape=(1, l))
beta1 = np.vstack((beta11, beta12, beta13, beta14))

beta21 = np.reshape(12*(coord_data-0.5)**2, newshape=(1, l))
beta22 = np.reshape(1.5*coord_data**0.5, newshape=(1, l))
beta23 = np.reshape(3*coord_data**2, newshape=(1, l))
beta24 = np.reshape(-2/3*coord_data, newshape=(1, l))
beta2 = np.vstack((beta21, beta22, beta23, beta24))

beta0 = np.zeros(shape=(p+q, l, 2))
beta0[:, :, 0] = beta1
beta0[:, :, 1] = beta2

# set up the unobserved effect size
gamma1 = -np.sqrt(2)*np.reshape(np.sin(np.pi*coord_data), newshape=(1, l))
gamma2 = np.sqrt(2)*np.reshape(np.cos(2*np.pi*coord_data), newshape=(1, l))
gamma0 = np.vstack((gamma1, gamma2))


for k in range(simu):
    # make folder
    idx = k+1
    data_folder = "./data/simu_1/%s/%d" % (phase, idx)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    coord_file_name = '%s/coord_data.mat' % data_folder
    savemat(coord_file_name, mdict={'coord_data': coord_data})
    # generate covariate data
    x_c = scale(norm.rvs(size=(n, p-1)))
    x_d = bernoulli.rvs(p=0.5, size=(n, 1))
    x = np.hstack((np.ones(shape=(n, 1)), x_d, x_c))
    x_file_name = '%s/x.mat' % data_folder
    savemat(x_file_name, mdict={'x': x})
    w = 0.1 * norm.rvs(size=(n, q))
    # set up the correlation between x and z
    if int(sys.argv[1]) == 1:
        para1, para2 = 0, 0
    elif int(sys.argv[1]) == 2:
        para1, para2 = 0, 0.2
    elif int(sys.argv[1]) == 3:
        para1, para2 = 0.2, 0.5
    else:
        para1, para2 = 0.5, 1
    a = uniform.rvs(loc=para1, scale=para2, size=(p + 1, q)) * (bernoulli.rvs(p=0.5, size=(p + 1, q)) * 2 - 1)
    alpha = np.array(a)
    z = np.dot(x, alpha) + w
    z_file_name = '%s/z.mat' % data_folder
    savemat(z_file_name, mdict={'z': z})
    y = np.zeros(shape=(n, l, m))
    sigma_1 = invgamma.rvs(a=3, scale=2, size=1)
    eta_1 = 0.1*sigma_1*norm.rvs(size=(n, l))
    err_1 = 0.1*sigma_1*norm.rvs(size=(n, l))
    y[:, :, 0] = np.dot(x, beta1) + np.dot(z, gamma1) + eta_1 + err_1
    sigma_2 = invgamma.rvs(a=3, scale=2, size=1)
    eta_2 = 0.1*sigma_2*norm.rvs(size=(n, l))
    err_2 = 0.1*sigma_2*norm.rvs(size=(n, l))
    y[:, :, 1] = np.dot(x, beta2) + np.dot(z, gamma2) + eta_2 + err_2
    y_file_name = '%s/y.mat' % data_folder
    savemat(y_file_name, mdict={'y': y})
    beta_file_name = '%s/beta0.mat' % data_folder
    savemat(beta_file_name, mdict={'beta0': beta0})
    gamma_file_name = '%s/gamma0.mat' % data_folder
    savemat(gamma_file_name, mdict={'gamma0': gamma0})
