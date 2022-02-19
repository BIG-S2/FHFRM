"""
Local linear kernel smoothing on beta in MVCM.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-09-18
"""

import numpy as np
import statsmodels.nonparametric.api as nparam

"""
installed all the libraries above
"""


def mvcm(coord_data, y, bw0='cv_ls', res_idx=0):
    """
        Local linear kernel smoothing on beta in MVCM.

        :param
            coord_data (matrix): common coordinate matrix (l*d)
            y (matrix): imaging response data (response matrix, n*l*m)
            bw0 (vector): pre-defined optimal bandwidth
            res_idx (scalar): indicator of calculating the residual matrix
        :return
            sm_y (matrix): smoothed response matrix (n*l*m)
            bw_o (matrix): optimal bandwidth (d*m)

    """

    # Set up
    n, l, m = y.shape
    d = coord_data.shape[1]
    sm_y = y * 0
    res_y = y * 0
    bw_o = np.zeros(shape=(d, m))

    if d == 1:
        var_tp = 'c'
    elif d == 2:
        var_tp = 'cc'
    else:
        var_tp = 'ccc'

    for mii in range(m):
        y_avg = np.mean(y[:, :, mii], axis=0)
        if bw0 is 'cv_ls':
            model_bw = nparam.KernelReg(endog=[y_avg], exog=[coord_data], var_type=var_tp, bw='cv_ls')
            bw_opt = model_bw.bw
            print("The optimal bandwidth for the " + str(mii+1) + "-th image measurement is")
            print(bw_opt)
            bw_o[:, mii] = bw_opt
        else:
            bw_opt = bw0[:, mii]
        for nii in range(n):
            y_ii = np.reshape(y[nii, :, mii], newshape=y_avg.shape)
            model_y = nparam.KernelReg(endog=[y_ii], exog=[coord_data], var_type=var_tp, bw=bw_opt)
            sm_y[nii, :, mii] = model_y.fit()[0]
        if res_idx == 1:
            res_y[:, :, mii] = y[:, :, mii] - sm_y[:, :, mii]
            print("The bound of the residual is ["
                  + str(np.min(res_y[:, :, mii])) + ", " + str(np.max(res_y[:, :, mii])) + "]")

    return sm_y, bw_o, res_y
