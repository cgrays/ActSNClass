import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern
import scipy.integrate as integrate


def kernel_RBF(k1=1e6, l1_t=5e3, l1_l=6e3, k2=1e5, l2_t=1e2, l2_l=3e2, k3=1e2, l3_t=5e0, l3_l=3e1, bounds=False):
    
    if bounds:
        
        kernel = C(k1, (1e-2,1e3)) * RBF((l1_t,l1_l), ((1e0,1e6),(1e0,1e6))) +\
        C(k2, (1e-2,1e3)) * RBF((l2_t,l2_l), ((1e0,1e5),(1e0,1e5))) +\
        C(k3, (1e-2,1e3)) * RBF((l3_t,l3_l), ((1e0,1e4),(1e0,1e4)))
                    
    else:
        
        kernel = C(k1) * RBF((l1_t,l1_l)) +\
        C(k2) * RBF((l2_t,l2_l)) +\
        C(k3) * RBF((l3_t,l3_l))
    
    return kernel

def kernel_Matern(k=1e0, l_t=5e3, l_l=5e3):
    
    kernel = C(k, (1e-2,1e4)) * Matern((l_t, l_l),((1e0,1e6),(1e0,1e6)), nu=1.5)
    
    return kernel

def kernel_RBF_general(length_time, length_wavelength, lower_lim_time, upper_lim_time, lower_lim_wavelength, upper_lim_wavelength):
    
    kernel = RBF((length_time,length_wavelength),\
                 ((lower_lim_time, upper_lim_time),\
                  (lower_lim_wavelength, upper_lim_wavelength)) )
    
    return kernel


def GP(lc, restarts=0, kernel_type=1):
    
    bands = np.unique(lc.photometry['lambda_cen'])

    X = np.vstack([lc.photometry['mjd']-np.min(lc.photometry['mjd']), lc.photometry['lambda_cen']]).T
    y = lc.photometry['flux'] - np.min(lc.photometry['flux'])
    dy = lc.photometry['fluxerr']
    
    
    if kernel_type==0: ### Fit a 9-parameter (3x3 RBF) kernel
        
        kn = kernel_RBF()
        final_gp = GaussianProcessRegressor(kernel=kn,alpha=dy,n_restarts_optimizer=restarts)
        final_gp.fit(X,y)
        
        kps=final_gp.kernel_.get_params()
        kernel_params = [kps['k1__k1__k1'].constant_value,\
                        kps['k1__k1__k2'].length_scale[0],\
                        kps['k1__k1__k2'].length_scale[1],\
                        kps['k1__k2__k1'].constant_value,\
                        kps['k1__k2__k2'].length_scale[0],\
                        kps['k1__k2__k2'].length_scale[1],\
                        kps['k2__k1'].constant_value,\
                        kps['k2__k2'].length_scale[0],\
                        kps['k2__k2'].length_scale[1],\
                        final_gp.log_marginal_likelihood(),\
                        ]
        
    elif kernel_type==1: ### Iteratively fit to the residuals

        k1 = kernel_RBF_general(5e3, 6e3, 1e0, 1e6, 1e0, 1e6)
        gp1 = GaussianProcessRegressor(kernel=k1,alpha=dy,n_restarts_optimizer=restarts)
        gp1.fit(X,y)
        kp1 = gp1.kernel_.get_params()



        gp1_result = gp1.predict(X)

        ### Fit the residuals to an RBF with smaller length scales
        k2 = kernel_RBF_general(1e2, 3e2, 1e0, kp1['length_scale'][0], 1e0, kp1['length_scale'][1])

        gp2 = GaussianProcessRegressor(kernel=k2,alpha=dy,n_restarts_optimizer=restarts)

        gp2.fit(X,y - gp1_result)

        kp2 = gp2.kernel_.get_params()

        gp2_result = gp2.predict(X)    

        ### Fit the second residuals to a final RBF with the smallest length scales (trying to capture noise).
        k3 = kernel_RBF_general(5e0, 3e1, 1e0, kp2['length_scale'][0], 1e0, kp2['length_scale'][1])

        gp3 = GaussianProcessRegressor(kernel=k3,alpha=dy,n_restarts_optimizer=restarts)

        gp3.fit(X,y - gp1_result - gp2_result)

        kp3 = gp3.kernel_.get_params()

        final_kernel = C(1.,(1e-2,1e3)) * RBF((kp1['length_scale'][0],kp1['length_scale'][1])) +\
        C(1.,(1e-2,1e3)) * RBF((kp2['length_scale'][0],kp2['length_scale'][1])) +\
        C(1.,(1e-2,1e3)) * RBF((kp3['length_scale'][0],kp3['length_scale'][1]))

        final_gp = GaussianProcessRegressor(kernel=final_kernel,alpha=dy,n_restarts_optimizer=restarts)

        final_gp.fit(X,y)

        kps = final_gp.kernel_.get_params()

        kernel_params = [kps['k1__k1__k1'].constant_value,\
                        kps['k1__k1__k2'].length_scale[0],\
                        kps['k1__k1__k2'].length_scale[1],\
                        kps['k1__k2__k1'].constant_value,\
                        kps['k1__k2__k2'].length_scale[0],\
                        kps['k1__k2__k2'].length_scale[1],\
                        kps['k2__k1'].constant_value,\
                        kps['k2__k2'].length_scale[0],\
                        kps['k2__k2'].length_scale[1],\
                        final_gp.log_marginal_likelihood(),\
                        ]
        
    elif kernel_type==2:
        
        kn = kernel_Matern()
        final_gp = GaussianProcessRegressor(kernel=kn,alpha=dy,n_restarts_optimizer=restarts)
        final_gp.fit(X,y)
        
        kps=final_gp.kernel_.get_params()
        kernel_params = [kps['k1'].constant_value,\
                        kps['k2'].length_scale[0],\
                        kps['k2'].length_scale[1],\
                        final_gp.log_marginal_likelihood(),\
                        ]
    else:
        raise ValueError('Parameter "kernel_type" must be one of [0,1,2].')
    
    # generate realization of GP fit on the data
    time_real = np.arange(np.floor(np.min(X[:,0])), np.ceil(np.max(X[:,0]))+1.) 
    tgp,wgp = np.meshgrid(time_real,bands)
    x_real = np.vstack([tgp.flatten(), wgp.flatten()]).T
    
    y_real = final_gp.predict(x_real)
    y_real = y_real.reshape(tgp.shape).T

    # compute features from the realization of the fit
    
    # Peak magnitude of the GP flux prediction in the LSST i band
    pkmag_i = 22.0 - 2.5*np.log10(np.max(y_real[:,3]))
    
    # Ratio of the maximum positive flux to the maximum-minus-minimum flux in the LSST i band
    pos_flux_ratio = np.max(y_real[:,3]) / (np.max(y_real[:,3]) - np.min(y_real[:,3]))
    
    # Normalized difference of the light curve colors at maximum/minimum light. The blue
    # measurement is the difference between the LSST i and g bands, and the red measurement
    # is the difference between the LSST y and i bands. The normalized difference is calculated
    # by taking the difference of the fluxes in the two bands divided by their sum.
    max_light_idx = np.where(y_real == np.max(y_real))[0][0]
    min_light_idx = np.where(y_real == np.min(y_real))[0][0]
    #
    max_fr_blue = (y_real[max_light_idx,3] - y_real[max_light_idx,1]) / (y_real[max_light_idx,3] + y_real[max_light_idx,1])
    min_fr_blue = (y_real[min_light_idx,3] - y_real[min_light_idx,1]) / (y_real[min_light_idx,3] + y_real[min_light_idx,1])
    max_fr_red = (y_real[max_light_idx,5] - y_real[max_light_idx,3]) / (y_real[max_light_idx,5] + y_real[max_light_idx,3])
    min_fr_red = (y_real[min_light_idx,5] - y_real[min_light_idx,3]) / (y_real[min_light_idx,5] + y_real[min_light_idx,3])
    
    # Difference of the time of maximum in the LSST y and g bands in days
    max_dt_yg = time_real[np.argmax(y_real[:,5])] - time_real[np.argmax(y_real[:,1])]
    
    # An estimate of the light curve “width” that is applicable even for non-supernova-like
    # transients and variables. This is implemented as the integral of the positive/negative
    # parts of the GP flux predictions divided by the positive/negative maximum fluxes.
    positive_width = integrate.trapz(y_real[:,3] - np.median(y_real[:,3])) / np.max(y_real[:,3])
    
    # Measurements of the rise and decline times of a light curve. This measurement is defined
    # as the time in days for the light curve to rise (bwd) or decline (fwd) to a given fraction
    # (either 20% or 50%) of maximum light in the LSST i band.
    
    ### NOTE: REVISIT THE DEFAULT EXCEPTION VALUES HERE (taken to be 1)
    ### REPLACE WITH MEAN VALUES
    
    
    try:
        time_fwd_max_20 = np.where(np.logical_and(\
                                                  y_real[:,3] <= (0.2 * (np.max(y_real[:,3]) - np.min(y_real[:,3])) + np.min(y_real[:,3])),\
                                                  time_real > np.argmax(y_real[:,3])\
                                                 ))[0][0] - np.argmax(y_real[:,3])
    except IndexError:
        time_fwd_max_20 = 1.
        
    try:    
        time_fwd_max_50 = np.where(np.logical_and(\
                                                  y_real[:,3] <= (0.5 * (np.max(y_real[:,3]) - np.min(y_real[:,3])) + np.min(y_real[:,3])),\
                                                  time_real > np.argmax(y_real[:,3])\
                                                 ))[0][0] - np.argmax(y_real[:,3])
    except IndexError:
        time_fwd_max_50 = 1.
        
    try:
        time_bwd_max_20 = np.argmax(y_real[:,3]) - np.where(np.logical_and(\
                                                  y_real[:,3] <= (0.2 * (np.max(y_real[:,3]) - np.min(y_real[:,3])) + np.min(y_real[:,3])),\
                                                  time_real < np.argmax(y_real[:,3])\
                                                 ))[0][-1]
    except IndexError:
        time_bwd_max_20 = 1.

    try:
        time_bwd_max_50 = np.argmax(y_real[:,3]) - np.where(np.logical_and(\
                                                  y_real[:,3] <= (0.5 * (np.max(y_real[:,3]) - np.min(y_real[:,3])) + np.min(y_real[:,3])),\
                                                  time_real < np.argmax(y_real[:,3])\
                                                 ))[0][-1]
    except IndexError:
        time_bwd_max_50 = 1.
       
    # Ratio of the rise/decline times calculated as described above in different bands. The blue
    # measurement is the difference between the LSST i and g bands, and the red measurement
    # is the difference between the LSST y and i bands.
    try:
        time_fwd_max_20_ratio_blue = float(time_fwd_max_20 / ((np.where(np.logical_and(y_real[:,1] <= (0.2 * (np.max(y_real[:,1]) - np.min(y_real[:,1])) + np.min(y_real[:,1])), time_real > np.argmax(y_real[:,1])))[0][0])) - np.argmax(y_real[:,1]))
    except IndexError:
        time_fwd_max_20_ratio_blue = 1. #float(time_fwd_max_20 / int(time_real[-1]))
        
    try:
        time_fwd_max_50_ratio_blue = float(time_fwd_max_50 / ((np.where(np.logical_and(y_real[:,1] <= (0.5 * (np.max(y_real[:,1]) - np.min(y_real[:,1])) + np.min(y_real[:,1])), time_real > np.argmax(y_real[:,1])))[0][0])) - np.argmax(y_real[:,1]))
    except IndexError:
        time_fwd_max_50_ratio_blue = 1. #float(time_fwd_max_50 / int(time_real[-1]))
        
    try:
        time_bwd_max_20_ratio_blue = float(time_bwd_max_20 / (np.argmax(y_real[:,1]) - (np.where(np.logical_and(y_real[:,1] <= (0.2 * (np.max(y_real[:,1]) - np.min(y_real[:,1])) + np.min(y_real[:,1])), time_real < np.argmax(y_real[:,1])))[0][-1])))
    except IndexError:
        time_bwd_max_20_ratio_blue = 1. #float(time_bwd_max_20 / int(time_real[1]))
        
    try:
        time_bwd_max_50_ratio_blue = float(time_bwd_max_50 / (np.argmax(y_real[:,1]) - (np.where(np.logical_and(y_real[:,1] <= (0.5 * (np.max(y_real[:,1]) - np.min(y_real[:,1])) + np.min(y_real[:,1])), time_real < np.argmax(y_real[:,1])))[0][-1])))
    except IndexError:
        time_bwd_max_50_ratio_blue = 1. #float(time_bwd_max_50 / int(time_real[1]))
        
    try:
        time_fwd_max_20_ratio_red = float(((np.where(np.logical_and(y_real[:,5] <= (0.2 * (np.max(y_real[:,5]) - np.min(y_real[:,5])) + np.min(y_real[:,5])), time_real > np.argmax(y_real[:,1])))[0][0]) - np.argmax(y_real[:,5])) / time_fwd_max_20)
    except IndexError:
        time_fwd_max_20_ratio_red = 1. #float(int(time_real[-1]) / time_fwd_max_20)
        
    try:
        time_fwd_max_50_ratio_red = float(((np.where(np.logical_and(y_real[:,5] <= (0.5 * (np.max(y_real[:,5]) - np.min(y_real[:,5])) + np.min(y_real[:,5])), time_real > np.argmax(y_real[:,5])))[0][0]) - np.argmax(y_real[:,5])) / time_fwd_max_50)
    except IndexError:
        time_fwd_max_50_ratio_red = 1. #float(int(time_real[-1]) / time_fwd_max_50)
        
    try:
        time_bwd_max_20_ratio_red = float((np.argmax(y_real[:,5]) - (np.where(np.logical_and(y_real[:,5] <= (0.2 * (np.max(y_real[:,5]) - np.min(y_real[:,5])) + np.min(y_real[:,5])), time_real < np.argmax(y_real[:,5])))[0][-1])) / time_bwd_max_20)
    except IndexError:
        time_bwd_max_20_ratio_red = 1. #float(int(time_real[0]) / time_bwd_max_20)
        
    try:
        time_bwd_max_50_ratio_red = float((np.argmax(y_real[:,5]) - (np.where(np.logical_and(y_real[:,5] <= (0.5 * (np.max(y_real[:,5]) - np.min(y_real[:,5])) + np.min(y_real[:,5])), time_real < np.argmax(y_real[:,5])))[0][-1])) / time_bwd_max_50)
    except IndexError:
        time_bwd_max_50_ratio_red = 1. #float(int(time_real[0]) / time_bwd_max_50)
    
    # Fraction of observations that have a signal greater than 5/less than -5 times the noise level.
    frac_s2n_5 = len(np.where(lc.photometry['SNR'] > 5.)[0]) / len(lc.photometry['SNR'])
    
    # Fraction of observations that have an absolute signal-to-noise less than 3
    frac_background = len(np.where(np.abs(lc.photometry['SNR']) < 3.)[0]) / len(lc.photometry['SNR'])
    
    # Time difference in days between the first observation with a signal-to-noise greater than 5
    # and the last such observation (in any band)
    time_width_s2n_5 = lc.photometry['mjd'][np.where(lc.photometry['SNR'] > 5.)[0][-1]] - lc.photometry['mjd'][np.where(lc.photometry['SNR'] > 5.)[0][0]]
    
    # Number of observations in any band within 5 days of maximum light
    max_light_mjd = np.min(lc.photometry['mjd']) + max_light_idx
    #
    count_max_center = len(np.where(np.logical_and(lc.photometry['mjd'] < 5. + max_light_mjd, lc.photometry['mjd'] > max_light_mjd - 5.))[0])
    
    # Number of observations in any band between 20, 50, or 100 days before maximum light
    # and 5 days after maximum light.
    count_max_rise_20 = len(np.where(np.logical_and(lc.photometry['mjd'] < 5. + max_light_mjd, lc.photometry['mjd'] > max_light_mjd - 20.))[0])
    count_max_rise_50 = len(np.where(np.logical_and(lc.photometry['mjd'] < 5. + max_light_mjd, lc.photometry['mjd'] > max_light_mjd - 50.))[0])
    count_max_rise_100 = len(np.where(np.logical_and(lc.photometry['mjd'] < 5. + max_light_mjd, lc.photometry['mjd'] > max_light_mjd - 100.))[0])
    
    # Number of observations in any band between 5 days before maximum light and 20, 50, or
    # 100 days after maximum light.
    count_max_fall_20 = len(np.where(np.logical_and(lc.photometry['mjd'] < 20. + max_light_mjd, lc.photometry['mjd'] > max_light_mjd - 5.))[0])
    count_max_fall_50 = len(np.where(np.logical_and(lc.photometry['mjd'] < 50. + max_light_mjd, lc.photometry['mjd'] > max_light_mjd - 5.))[0])
    count_max_fall_100 = len(np.where(np.logical_and(lc.photometry['mjd'] < 100. + max_light_mjd, lc.photometry['mjd'] > max_light_mjd - 5.))[0])
   
    # Total signal-to-noise of all observations of the object.
    total_s2n = np.sum(lc.photometry['SNR'])
        
    fit_params = [pkmag_i, pos_flux_ratio, max_fr_blue, min_fr_blue, max_fr_red, min_fr_red, max_dt_yg, positive_width,\
                 time_fwd_max_20, time_fwd_max_50, time_bwd_max_20, time_bwd_max_50, time_fwd_max_20_ratio_blue,\
                 time_fwd_max_50_ratio_blue, time_bwd_max_20_ratio_blue, time_bwd_max_50_ratio_blue,\
                 time_fwd_max_20_ratio_red, time_fwd_max_50_ratio_red, time_bwd_max_20_ratio_red,\
                 time_bwd_max_50_ratio_red, frac_s2n_5, frac_background, time_width_s2n_5,\
                 count_max_center, count_max_rise_20, count_max_rise_50, count_max_rise_100,\
                 count_max_fall_20, count_max_fall_50, count_max_fall_100,\
                 total_s2n]
    
    
    return kernel_params, fit_params

def main():
    return None


if __name__ == '__main__':
    main()
