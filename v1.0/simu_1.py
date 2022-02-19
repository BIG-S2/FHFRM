"""
Run simulation script: MVCM-SVA pipeline
Usage: python ./simu_1.py ./data/simu_2/setting_2/1/ ./result/simu_2/setting_2/1/

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-12-28
"""

import sys
import os
from scipy.io import loadmat, savemat
import numpy as np
from numpy.linalg import inv
from mvcm import mvcm
from mvcm_sva import mvcm_sva

"""
installed all the libraries above
"""

if __name__ == '__main__':
    input_dir0 = sys.argv[1]
    output_dir0 = sys.argv[2]
    if not os.path.exists(output_dir0):
        os.mkdir(output_dir0)

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Step 0. load dataset\n """)
    print("+++++++Read the response data+++++++")
    y_file_name = input_dir0 + "y.mat"
    mat = loadmat(y_file_name)
    y_data = mat['y']
    if len(y_data.shape) == 2:
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
    n, l, m = y_data.shape
    print("The matrix dimension of response data is " + str(y_data.shape))
    print("+++++++Read the imaging coordinate data+++++++")
    coord_file_name = input_dir0 + "coord_data.mat"
    mat = loadmat(coord_file_name)
    coord_data = mat['coord_data']
    if len(coord_data.shape) == 1:
        coord_data = coord_data.reshape(coord_data.shape[0], 1)
    elif coord_data.shape[0] == 1:
        coord_data = coord_data.T
    # d = coord_data.shape[1]
    print("The matrix dimension of coordinate data is " + str(coord_data.shape))
    print("+++++++Read the covariate data+++++++")
    x_file_name = input_dir0 + "x.mat"
    mat = loadmat(x_file_name)
    x_data = mat['x']
    print("The matrix dimension of covariate data is " + str(x_data.shape))

    p = x_data.shape[1]
    c_mat = np.dot(inv(np.dot(x_data.T, x_data) + np.eye(p) * 0.00001), x_data.T)
    sm_y, bw_beta, _ = mvcm(coord_data, y_data)
    res_y = y_data * 0
    b_s = np.zeros(shape=(p, l, m))
    for mii in range(m):
        b_s[:, :, mii] = np.dot(c_mat, sm_y[:, :, mii])
        res_y[:, :, mii] = y_data[:, :, mii] - np.dot(x_data, b_s[:, :, mii])
    print("+++++++ permutation approach for q +++++++")
    i_res_y = np.dot(np.mean(res_y, axis=1).T, np.mean(res_y, axis=1))
    eigval = np.linalg.eigvalsh(i_res_y)
    n_per = 99  # number of permutations
    pre_eigval = np.zeros(shape=(n_per, m))
    nge = np.zeros(shape=(m, 1))
    for k_ii in range(n_per):
        per_mat = np.mean(res_y, axis=1)
        for mii in range(m-1):
            per_mat[:, mii+1] = per_mat[np.random.permutation(n), mii+1]
        i_per_mat = np.dot(per_mat.T, per_mat)
        pre_eigval[k_ii, :] = np.linalg.eigvalsh(i_per_mat)
    for mii in range(m):
        nge[mii] = (np.sum(pre_eigval[:, mii] >= eigval[mii])+1)/(n_per+1)
    q = int(sum(nge <= 0.05)[0])
    print("The selected number of unobserved covariates is " + str(q))

    _, b_f, gamma_new, g_mat, _ = mvcm_sva(x_data, y_data, sm_y, q)

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Evaluate the estimation accuracy\n """)
    # true value: the observed effect size
    beta0_file_name = input_dir0 + "beta0.mat"
    mat = loadmat(beta0_file_name)
    beta0 = mat['beta0']

    b0_err = np.sqrt(np.sum(np.mean((b_s - beta0)**2, axis=1)))
    b_err = np.sqrt(np.sum(np.mean((b_f - beta0)**2, axis=1)))

    print("The integrated square error (ISE) of main effect is " + str(b0_err))
    print("The integrated square error (ISE) of main effect is " + str(b_err))

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Save results\n """)
    coord_file_name = output_dir0 + "coord_data.mat"
    savemat(coord_file_name, mdict={'coord_data': coord_data})
    bs_file_name = output_dir0 + "b_s.mat"
    savemat(bs_file_name, mdict={'b_s': b_s})
    bf_file_name = output_dir0 + "b_f.mat"
    savemat(bf_file_name, mdict={'b_f': b_f})
    q_file_name = output_dir0 + "q.mat"
    savemat(q_file_name, mdict={'q': q})
    gamma_file_name = output_dir0 + "gamma_new.mat"
    savemat(gamma_file_name, mdict={'gamma_new': gamma_new})
    gmat_file_name = output_dir0 + "g_mat.mat"
    savemat(gmat_file_name, mdict={'g_mat': g_mat})
