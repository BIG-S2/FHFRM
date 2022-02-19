"""
main script for MVCM-SVA pipeline
Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-12-28
"""

import time
import numpy as np
from numpy.linalg import inv

"""
installed all the libraries above
"""


def mvcm_sva(x, y, sm_y, q):
    """
        Run the simulation script for mvcm_sva

        :param
            x (matrix): covariate matrix (n*p)
            y (matrix): imaging response data (response matrix, n*l*m)
            sm_y (matrix): smoothed imaging response data (response matrix, n*l*m)
            q (scalar): number of confounders
        :return
            b_0: estimate of beta via mvcm ignoring confounders
            b: estimate of beta via mvcm_sva
            gamma: estimate of confounding effects
            g_mat: estimate of confounders

    """

    n, l, m = y.shape
    res_y_new = y * 0
    p = x.shape[1]

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Step 1. fit the multivariate varying coefficient model (MVCM) without confounding effects\n """)
    start_0 = time.time()

    """ find the optimal bandwidth & fit MVCM & get the residual terms """
    """ calculate the hat matrix """
    c_mat = np.dot(inv(np.dot(x.T, x) + np.eye(p) * 0.00001), x.T)
    res_y = y * 0
    b_star = np.zeros(shape=(p, l, m))
    for mii in range(m):
        b_star[:, :, mii] = np.dot(c_mat, sm_y[:, :, mii])
        res_y[:, :, mii] = y[:, :, mii] - np.dot(x, b_star[:, :, mii])

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Step 2. recover confounding effects via SVD on integrated residual terms\n """)

    """ integration of residual terms & SVD """
    gamma = np.zeros(shape=(q, l, m))
    ires_y = np.mean(res_y, axis=1)
    u = np.linalg.svd(ires_y, full_matrices=True)[0]
    # q = 2   # q is the estimated number of confounders, which needs to be determined. Here fixed.
    u_q = np.reshape(u[:, 0:q], newshape=(n, q))
    for mii in range(m):
        gamma[:, :, mii] = np.dot(inv(np.dot(u_q.T, u_q)), np.dot(u_q.T, res_y[:, :, mii]))

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Step 3. refine the varying coefficients of observed effects\n """)

    b = np.zeros(shape=(p, l, m))
    center_mat = np.eye(m)-np.dot(np.ones(shape=(m, 1)), np.ones(shape=(m, 1)).T)/m
    b_c = np.dot(np.mean(b_star, axis=1), center_mat)
    gamma_c = np.dot(np.mean(gamma, axis=1), center_mat)
    b_c_gamma = np.dot(b_c, gamma_c.T)
    gamma_c_gamma = np.dot(gamma_c, gamma_c.T)
    g_mat = u_q + np.dot(np.dot(x, b_c_gamma), inv(gamma_c_gamma+0.000001*np.ones(q)))

    for mii in range(m):
        b[:, :, mii] = b_star[:, :, mii] - np.dot(c_mat, np.dot(g_mat, gamma[:, :, mii]))
        res_y_new[:, :, mii] = y[:, :, mii] - np.dot(x, b[:, :, mii]) - np.dot(g_mat, gamma[:, :, mii])

    end_0 = time.time()

    print("------------------------------- \n MVCM_SVA pipeline is finished !\n ---------------------------")

    print("Elapsed time is " + str(end_0 - start_0))

    return b_star, b, gamma, g_mat, res_y_new
