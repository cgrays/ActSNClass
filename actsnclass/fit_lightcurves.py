# Copyright 2019 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 9 August 2019
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from actsnclass.bazin import bazin, fit_scipy
#from actsnclass.GP import run_code
from actsnclass.GP_sklearn import GP, kernel_RBF, kernel_Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd

from astropy.table import Table

import multiprocessing
from multiprocessing import Manager
from functools import partial

from tqdm import tqdm

__all__ = ['LightCurve', 'fit_snpcc_bazin', 'run_parallel_job', 'fit_plasticc_bazin', 'fit_plasticc_GP']


class LightCurve(object):
    """ Light Curve object, holding meta and photometric data.

    Attributes
    ----------
    bazin_features_names: list
        List of names of the Bazin function parameters.
    bazin_features: list
        List with the 5 best-fit Bazin parameters in all filters.
        Concatenated from blue to red.
    bazin_band_cost: list
        List of 6 floats corresponding to the value of the cost function for the fit in each band.
    dataset_name: str
        Name of the survey or data set being analyzed.
    filters: list
        List of broad band filters.
    id: int
        SN identification number
    photometry: pd.DataFrame
        Photometry information. Keys --> [mjd, band, flux, fluxerr, SNR, MAG, MAGERR].
    redshift: float
        Redshift
    sample: str
        Original sample to which this light curve is assigned
    sim_peakmag: np.array
        Simulated peak magnitude in each filter
    sncode: int
        Number identifying the SN model used in the simulation
    sntype: str
        General classification, possibilities are: Ia, II or Ibc

    Methods
    -------
    check_queryable(mjd: float, r_lim: float)
        Check if this light can be queried in a given day.
    load_snpcc_lc(path_to_data: str)
        Reads header and photometric information for 1 light curve
    fit_bazin(band: str) -> list
        Calculates best-fit parameters from the Bazin function in 1 filter
    fit_bazin_all()
        Calculates  best-fit parameters from the Bazin func for all filters
    plot_bazin_fit(save: bool, show: bool, output_file: srt)
        Plot photometric points and Bazin fitted curve

    Examples
    --------
    >>> from actsnclass import LightCurve

    define path to light curve file

    >>> path_to_lc = 'data/SIMGEN_PUBLIC_DES/DES_SN431546.DAT'

    >>> lc = LightCurve()                        # create light curve instance
    >>> lc.load_snpcc_lc(path_to_lc)             # read data
    >>> lc.photometry                            # display photometry
              mjd band     flux  fluxerr   SNR
    0   56207.188    g   9.6560    4.369  2.21
    1   56207.195    r   6.3370    3.461  1.83
    ...        ...  ...      ...      ...   ...
    96  56336.043    r  14.4300    3.098  4.66
    97  56336.055    i  18.9500    5.029  3.77
    [98 rows x 5 columns]

    >>> lc.fit_bazin_all()                  # perform Bazin fit in all filters
    >>> lc.bazin_features                   # display Bazin parameters
    [62.0677260096896, -7.959383808822104, 47.37511467606875, 37.4919069623379,
    ... ... ...
    206.65806244385922, -4.777010246622081]

    plot light curve fit

    >>> lc.plot_bazin_fit(output_file=str(lc.id) + '.png')

    for fitting the entire sample...

    >>> path_to_data_dir = 'data/SIMGEN_PUBLIC_DES/'     # raw data directory
    >>> output_file = 'results/Bazin.dat'       # output file
    >>> fit_snpcc_bazin(path_to_data_dir=path_to_data_dir, features_file=output_file)

    a file with all Bazin fits for this data set was produced
    """

    def __init__(self):
        self.bazin_features = []
        self.bazin_features_names = ['a', 'b', 't0', 'tfall', 'trise']
        self.bazin_band_cost = []
        self.GP_kernel_params = []
        self.GP_kernel = []
        self.GP_features = []
        self.GP_feature_names = ['host_photoz', 'host_photoz_err',\
                                 'pkmag_i', 'pos_flux_ratio', 'max_fr_blue', 'min_fr_blue',\
                                'max_fr_red', 'min_fr_red', 'max_dt_yg', 'positive_width', 'time_fwd_max_20', 'time_fwd_max_50',\
                                'time_bwd_max_20', 'time_bwd_max_50', 'time_fwd_max_20_ratio_blue',
                                'time_fwd_max_50_ratio_blue', 'time_bwd_max_20_ratio_blue', 'time_bwd_max_50_ratio_blue',\
                                'time_fwd_max_20_ratio_red', 'time_fwd_max_50_ratio_red', 'time_bwd_max_20_ratio_red',\
                                'time_bwd_max_50_ratio_red', 'frac_s2n_5', 'frac_background', 'time_width_s2n_5',\
                                'count_max_center', 'count_max_rise_20', 'count_max_rise_50', 'count_max_rise_100',\
                                'count_max_fall_20', 'count_max_fall_50', 'count_max_fall_100',\
                                'total_s2n']
        self.dataset_name = ' '
        self.filters = []
        self.tfluxes = []
        self.id = 0
        self.photometry = pd.DataFrame()
        self.redshift = 0
        self.sample = ' '
        self.sim_peakmag = []
        self.sncode = 0
        self.sntype = ' '
        self.num_classes = 0

    def load_snpcc_lc(self, path_to_data: str):
        """Reads one LC from SNPCC data.

        Populates the attributes: dataset_name, id, sample, redshift, sncode,
        sntype, photometry and sim_peakmag.

        Parameters
        ---------
        path_to_data: str
            Path to text file with data from a single SN.
        """

        # set the designation of the data set
        self.dataset_name = 'SNPCC'

        # set filters
        self.filters = ['g', 'r', 'i', 'z']

        # set SN types
        snii = ['2', '3', '4', '12', '15', '17', '19', '20', '21', '24', '25',
                '26', '27', '30', '31', '32', '33', '34', '35', '36', '37',
                '38', '39', '40', '41', '42', '43', '44']

        snibc = ['1', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16',
                 '18', '22', '23', '29', '45', '28']

        # read light curve data
        op = open(path_to_data, 'r')
        lin = op.readlines()
        op.close()

        # separate elements
        data_all = np.array([elem.split() for elem in lin])

        # flag useful lines
        flag_lines = np.array([True if len(line) > 1 else False for line in data_all])

        # get only informative lines
        data = data_all[flag_lines]

        photometry_raw = []               # store photometry
        header = []                      # store parameter header

        # get header information
        for line in data:
            if line[0] == 'SNID:':
                self.id = int(line[1])
            elif line[0] == 'SNTYPE:':
                if line[1] == '-9':
                    self.sample = 'test'
                else:
                    self.sample = 'train'
            elif line[0] == 'SIM_REDSHIFT:':
                self.redshift = float(line[1])
            elif line[0] == 'SIM_NON1a:':
                self.sncode = line[1]
                if line[1] in snibc:
                    self.sntype = 'Ibc'
                elif line[1] in snii:
                    self.sntype = 'II'
                elif line[1] == '0':
                    self.sntype = 'Ia'
                else:
                    raise ValueError('Unknown supernova type!')
            elif line[0] == 'VARLIST:':
                header: list = line[1:]
            elif line[0] == 'OBS:':
                photometry_raw.append(np.array(line[1:]))
            elif line[0] == 'SIM_PEAKMAG:':
                self.sim_peakmag = np.array([float(item) for item in line[1:5]])

        # transform photometry into array
        photometry_raw = np.array(photometry_raw)

        # put photometry into data frame
        self.photometry['mjd'] = np.array([float(item) for item in photometry_raw[:, header.index('MJD')]])
        self.photometry['band'] = np.array(photometry_raw[:, header.index('FLT')])
        self.photometry['flux'] = np.array([float(item) for item in photometry_raw[:, header.index('FLUXCAL')]])
        self.photometry['fluxerr'] = np.array([float(item) for item in photometry_raw[:, header.index('FLUXCALERR')]])
        self.photometry['SNR'] = np.array([float(item) for item in photometry_raw[:, header.index('SNR')]])
        self.photometry['MAG'] = np.array([float(item) for item in photometry_raw[:, header.index('MAG')]])
        self.photometry['MAGERR'] = np.array([float(item) for item in photometry_raw[:, header.index('MAGERR')]])

    def load_plasticc_lc_fromfile(self, path_to_data: str, path_to_metadata: str, lc_id: int, num_classes = 8):
        """Reads one LC from the astropy table containing the PLAsTiCC data.

        Populates the attributes: dataset_name, id, sample, redshift, sncode
        sntype, and photometry.

        Parameters
        ---------
        path_to_data: str
            Path to text file with data from a single SN.
            
        path_to_metadata: str
            Path to the corresponding metadata file that contains information on the target SN (either the train or test metadata).
            
        lc_id: int
            The light curve ID to pull from the astropy Table.
        """

        # set the designation of the data set
        self.dataset_name = 'PLAsTiCC'
        self.num_classes = num_classes

        # set filters
        self.filters = ['u', 'g', 'r', 'i', 'z', 'y']
        
        passband_dict = {0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y'}

        wavelength_dict = {0: 365, 1: 464, 2: 658, 3: 806, 4: 900, 5: 1020}

        # set SN types
        if self.num_classes == 8:
            SNtypes = {'90': 'Ia',
                      '42': 'II',
                      '62': 'Ibc',
                      '67': 'Ia-91bg',
                      '52': 'Ia-x',
                      '95': 'SLSN-I',
                      '15': 'TDE',
                      '64': 'KN'}
        elif self.num_classes == 3:
            SNtypes = {'90': 'Ia',
                      '42': 'II',
                      '62': 'Ibc'}        
        
        # read metadata
        with open(path_to_metadata,'rb') as f2:
            meta_table = Table.read(f2,format='csv')
            
            assert lc_id in meta_table['object_id'], 'Desired light curve not in metadata file!'
            
            lc_meta = meta_table[meta_table['object_id']==lc_id]
        
        self.tfluxes = [lc_meta['tflux_{0}'.format(f)] for f in self.filters]
        
        true_target = str(int(lc_meta['true_target'].data[0]))
        if true_target not in list(SNtypes.keys()):
            raise ValueError('Unknown transient type!')
        else:
            pass
        
        # read light curve data
        with open(path_to_data,'rb') as f1:
            lc_table = Table.read(f1,format='csv')
            lc_photometry = lc_table[lc_table['object_id']==lc_id]
              
        # get header information
        self.id = int(lc_meta['object_id'])
        
        if 'train' in path_to_metadata:
            self.sample = 'train'
        else:
            self.sample = 'test'
            
        self.redshift = float(lc_meta['true_z'])
        
        self.sntype = SNtypes[true_target]
        self.sncode = true_target

        # put photometry into data frame
        self.photometry['mjd'] = np.array(lc_photometry['mjd'])
        self.photometry['band'] = np.array([passband_dict[key] for key in np.array(lc_photometry['passband'])])
        self.photometry['passband'] = np.array(lc_photometry['passband'])
        self.photometry['lambda_cen'] = np.array([wavelength_dict[key] for key in np.array(lc_photometry['passband'])])
        # add the template flux values to the flux
        self.photometry['flux'] = np.array(lc_photometry['flux']) + np.take(self.tfluxes, self.photometry['passband'])
        self.photometry['fluxerr'] = np.array(lc_photometry['flux_err'])
        self.photometry['SNR'] = (self.photometry['flux'] / self.photometry['fluxerr']) ** 2.
        #self.photometry['MAG'] = 22.0 - 2.5*np.log10(self.photometry['flux'])
        #self.photometry['MAGERR'] = 22.0 - 2.5*np.log10(self.photometry['fluxerr'])
        
        self.GP_features = [lc_meta['hostgal_photoz'].data[0], lc_meta['hostgal_photoz_err'].data[0]]
        
    def load_plasticc_lc(self, meta_table: Table, lc_table: Table, lc_id: int, path_to_metadata: str, num_classes = 8):
        """Reads one LC from the astropy table containing the PLAsTiCC data.

        Populates the attributes: dataset_name, id, sample, redshift, sncode,
        sntype, target, and photometry.

        Parameters
        ---------
        path_to_data: str
            Path to text file with data from a single SN.
            
        path_to_metadata: str
            Path to the corresponding metadata file that contains information on the target SN (either the train or test metadata).
            
        lc_id: int
            The light curve ID to pull from the astropy Table.
        """

        # set the designation of the data set
        self.dataset_name = 'PLAsTiCC'
        self.num_classes = num_classes

        # set filters
        self.filters = ['u', 'g', 'r', 'i', 'z', 'y']
        
        passband_dict = {0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y'}
        
        wavelength_dict = {0: 365, 1: 464, 2: 658, 3: 806, 4: 900, 5: 1020}

        # set SN types
        if self.num_classes == 8:
            SNtypes = {'90': 'Ia',
                      '42': 'II',
                      '62': 'Ibc',
                      '67': 'Ia-91bg',
                      '52': 'Ia-x',
                      '95': 'SLSN-I',
                      '15': 'TDE',
                      '64': 'KN'}
        elif self.num_classes == 3:
            SNtypes = {'90': 'Ia',
                      '42': 'II',
                      '62': 'Ibc'} 
        
        # read metadata
        lc_meta = meta_table[meta_table['object_id']==lc_id]
        
        self.tfluxes = [lc_meta['tflux_{0}'.format(f)] for f in self.filters]
        
        
        true_target = str(int(lc_meta['true_target'].data[0]))
        if true_target not in list(SNtypes.keys()):
            raise ValueError('Unknown transient type!')
        else:
            pass
        
        # read light curve data
        lc_photometry = lc_table[lc_table['object_id']==lc_id]
        
        # get header information
        self.id = int(lc_meta['object_id'])
        
        if 'train' in path_to_metadata:
            self.sample = 'train'
        else:
            self.sample = 'test'
            
        self.redshift = float(lc_meta['true_z'])
        
        self.sntype = SNtypes[true_target]
        self.sncode = true_target
 
        # put photometry into data frame
        self.photometry['mjd'] = np.array(lc_photometry['mjd'])
        self.photometry['band'] = np.array([passband_dict[key] for key in np.array(lc_photometry['passband'])])
        self.photometry['passband'] = np.array(lc_photometry['passband'])
        self.photometry['lambda_cen'] = np.array([wavelength_dict[key] for key in np.array(lc_photometry['passband'])])
        self.photometry['flux'] = np.array(lc_photometry['flux']) + np.take(self.tfluxes, self.photometry['passband'])
        self.photometry['fluxerr'] = np.array(lc_photometry['flux_err'])
        self.photometry['SNR'] = (self.photometry['flux'] / self.photometry['fluxerr']) ** 2.
        #self.photometry['MAG'] = 22.0 - 2.5*np.log10(self.photometry['flux'])
        #self.photometry['MAGERR'] = 22.0 - 2.5*np.log10(self.photometry['fluxerr'])
        
        self.GP_features = [lc_meta['hostgal_photoz'].data[0], lc_meta['hostgal_photoz_err'].data[0]]

    def check_queryable(self, mjd: float, r_lim: float):
        """Check if this light can be queried in a given day.

        This checks only r-band mag limit in a given epoch.
        It there is no observation on that day, use the last available
        observation.

        Parameters
        ----------
        mjd: float
            MJD where the query will take place.
        r_lim: float
            r-band magnitude limit below which query is possible.

        Returns
        -------
        bool
            If true, sample is changed to `queryable`.
        """

        # create photo flag
        photo_flag = self.photometry['mjd'].values <= mjd
        rband_flag = self.photometry['band'].values == 'r'
        surv_flag = np.logical_and(photo_flag, rband_flag)

        # check surviving photometry
        surv_mag = self.photometry['MAG'].values[surv_flag]

        if len(surv_mag) > 0 and 0 < surv_mag[-1] <= r_lim:
            return True
        else:
            return False

    def fit_bazin(self, band: str):
        """Extract Bazin features for one filter.

        Parameters
        ----------
        band: str
            Choice of broad band filter

        Returns
        -------
        bazin_param: list, float
            Best fit parameters for the Bazin function: [a, b, t0, tfall, trise] followed by a float corresponding to 
            the value of the cost function for the fit.
        """

        # build filter flag
        filter_flag = self.photometry['band'] == band

        # get info for this filter
        time = self.photometry['mjd'].values[filter_flag]
        ### Do we want to be subtracting off the minimum flux here?
        flux = self.photometry['flux'].values[filter_flag] - np.min(self.photometry['flux'].values)
        errors = self.photometry['fluxerr'].values[filter_flag]

        # fit Bazin function
        bazin_param = fit_scipy(time - time[0], flux, errors)

        return bazin_param

    def fit_GP(self,restarts=0, kernel_type=1):
        """Construct the GP fit simultaneously for all bands and extract features used for learning.

        Parameters
        ----------

        Returns
        -------
        GP_result: type 
        fill in info here.
        
        Populates the attributes: GP_kernel_params, GP_features
        """        
        fit_result = GP(self, restarts=restarts, kernel_type=kernel_type)

        self.GP_kernel_params = fit_result[0]
        self.GP_features.extend(fit_result[1])
        self.GP_features.extend(fit_result[0])
        
        if kernel_type == 0 or kernel_type == 1:
            self.GP_kernel = ['c1','rbf1_t', 'rbf1_l', 'c2', 'rbf2_t', 'rbf2_l', 'c3', 'rbf3_t', 'rbf3_l']
            
        elif kernel_type == 2:
            self.GP_kernel = ['c1', 'matern_t', 'matern_l']

    def fit_bazin_all(self):
        """Perform Bazin fit for all filters independently and concatenate results.

        Populates the attributes: bazin_features.
        """

        for band in self.filters:
            # build filter flag
            filter_flag = self.photometry['band'] == band

            # ensure that at least 5 data points have been observed in the band
            if sum(filter_flag) > 4:
                fit_result = self.fit_bazin(band)
                best_fit = fit_result[0]
                fit_cost = fit_result[1]

                if sum([str(item) == 'nan' for item in best_fit]) == 0:
                    for fit in best_fit:
                        self.bazin_features.append(fit)
                    self.bazin_band_cost.append(fit_cost)
                else:
                    for i in range(5):
                        self.bazin_features.append('None')
                    self.bazin_band_cost.append('None')
            else:
                for i in range(5):
                    self.bazin_features.append('None')
                self.bazin_band_cost.append('None')

    def plot_bazin_fit(self, save=False, show=True, output_file=' '):
        """
        Plot data and Bazin fitted function.

        Parameters
        ----------
        save: bool (optional)
             Save figure to file. Default is True.
        show: bool (optinal)
             Display plot in windown. Default is False.
        output_file: str
            Name of file to store the plot.
        """

        plt.figure()

        for i in range(len(self.filters)):
            plt.subplot(2, len(self.filters) / 2 + len(self.filters) % 2, i + 1)
            plt.title('Filter: ' + self.filters[i])

            # filter flag
            filter_flag = self.photometry['band'] == self.filters[i]
            x = self.photometry['mjd'][filter_flag].values
            y = self.photometry['flux'][filter_flag].values - np.min(self.photometry['flux'].values)
            yerr = self.photometry['fluxerr'][filter_flag].values

            # shift to avoid large numbers in x-axis
            time = x - min(x)
            xaxis = np.linspace(0, max(time), 500)[:, np.newaxis]
            # calculate fitted function
            fitted_flux = np.array([bazin(t, self.bazin_features[i * 5],
                                          self.bazin_features[i * 5 + 1],
                                          self.bazin_features[i * 5 + 2],
                                          self.bazin_features[i * 5 + 3],
                                          self.bazin_features[i * 5 + 4])
                                    for t in xaxis])

            plt.errorbar(time, y, yerr=yerr, color='blue', fmt='o')
            plt.plot(xaxis, fitted_flux, color='red', lw=1.5)
            plt.xlabel('MJD - ' + str(min(x)))
            plt.ylabel('FLUXCAL')
            plt.tight_layout()

        if save:
            plt.savefig(output_file)
        if show:
            plt.show()
            
    def plot_GP_fit(self, save=False, show=True, output_file=' ', threeDim=False, plot_bazin=True, figsize=(10,10)):
        """
        Plot data and GP fitted function.

        Parameters
        ----------
        save: bool (optional)
             Save figure to file. Default is True.
        show: bool (optional)
             Display plot in window. Default is False.
        output_file: str
            Name of file to store the plot.
        threeDim: bool (optional)
            Plot the fit in 3D. Default is False.
        """
        
        bands = np.unique(self.photometry['lambda_cen'])

        X = np.vstack([self.photometry['mjd']-np.min(self.photometry['mjd']), self.photometry['lambda_cen']]).T
        y = self.photometry['flux'] - np.min(self.photometry['flux'])
        dy = self.photometry['fluxerr']
        
        if len(self.GP_kernel_params[:-1]) == 9:
            kn = kernel_RBF(*self.GP_kernel_params[:-1])
            
        elif len(self.GP_kernel_params[:-1]) == 3:
            kn = kernel_Matern(*self.GP_kernel_params[:-1])
            
        else:
            raise ValueError('Unexpected number of kernel hyperparameters in self.GP_kernel_params.')
            
        gp = GaussianProcessRegressor(kernel=kn,alpha=dy,n_restarts_optimizer=0)
        gp.fit(X,y)
                
        time_plot = np.arange(np.floor(np.min(X[:,0])), np.ceil(np.max(X[:,0]))+1.)
        
        if not threeDim:
       
            h,axes = plt.subplots(2,int(len(self.filters) / 2 + len(self.filters) % 2), figsize=figsize)
            for k,ax in enumerate(axes.flatten()):              
                
                wavelength_plot = bands[k]
                tgp, wgp = np.meshgrid(time_plot, wavelength_plot)
                x = np.vstack([tgp.flatten(), wgp.flatten()]).T

                y_plot, sigma_plot = gp.predict(x, return_std=True)
                
                filter_flag = X[:,1] == bands[k]
                ax.errorbar(X[:,0][filter_flag],y[filter_flag],dy[filter_flag],fmt='.',color='k')

                ax.plot(tgp.mean(0),y_plot, label='GP',color='C1')
                ax.fill_between(tgp.mean(0),y_plot+sigma_plot,y_plot-sigma_plot,alpha=0.3,color='C1')
                
                if plot_bazin:
                    bfilter_flag = self.photometry['band'] == self.filters[k]
                    bx = self.photometry['mjd'][filter_flag].values

                    # shift to avoid large numbers in x-axis
                    btime = bx - min(bx)
                    bxaxis = np.linspace(0, max(btime), 500)[:, np.newaxis]
                    # calculate fitted function
                    fitted_flux = np.array([bazin(t, self.bazin_features[k * 5],
                                                  self.bazin_features[k * 5 + 1],
                                                  self.bazin_features[k * 5 + 2],
                                                  self.bazin_features[k * 5 + 3],
                                                  self.bazin_features[k * 5 + 4])
                                            for t in bxaxis])

                    ax.plot(bxaxis, fitted_flux, color='C0', label='Bazin')
                    
                if k == 0:
                    ax.legend(loc='best')

                ax.set_title('Filter: {0}'.format(self.filters[k]))
                ax.set_ylabel('Flux')
                ax.set_xlabel('MJD - {0}'.format(np.min(self.photometry['mjd'])))
            plt.tight_layout()

        else: #if threeDim
            
            wavelength_plot = np.linspace(np.floor(np.min(X[:,1])), np.ceil(np.max(X[:,1]))+1., 100)
            time_grid, wavelength_grid = np.meshgrid(time_plot, wavelength_plot)
            x = np.vstack([time_grid.flatten(), wavelength_grid.flatten()]).T
            
            y_pred, sigma = gp.predict(x, return_std=True)
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            ax.scatter(X[:,1],X[:,0],y,c='k')

            surf = ax.plot_surface(wavelength_grid,time_grid, y_pred.reshape(np.shape(time_grid)), cmap=plt.cm.viridis)

            ax.set_xlabel('$\lambda$')
            ax.set_ylabel('MJD - {0}'.format(np.min(self.photometry['mjd'])))
            ax.set_zlabel('Flux')
            plt.tight_layout()

        if save:
            plt.savefig(output_file)
        if show:
            plt.show()


def fit_snpcc_bazin(path_to_data_dir: str, features_file: str):
    """Fit Bazin functions to all filters in training and test samples.

    Parameters
    ----------
    path_to_data_dir: str
        Path to directory containing the set of individual files, one for each light curve.
    features_file: str
        Path to output file where results should be stored.
    """

    # read file names
    file_list_all = os.listdir(path_to_data_dir)
    lc_list = [elem for elem in file_list_all if 'DES_SN' in elem]

    # count survivors
    count_surv = 0

    # add headers to files
    with open(features_file, 'w') as param_file:
        param_file.write('id redshift type code sample gA gB gt0 gtfall gtrise rA rB rt0 rtfall rtrise iA iB it0 ' +
                         'itfall itrise zA zB zt0 ztfall ztrise\n')

    for file in lc_list:

        # fit individual light curves
        lc = LightCurve()
        lc.load_snpcc_lc(path_to_data_dir + file)
        lc.fit_bazin_all()

        print(lc_list.index(file), ' - id:', lc.id)

        # append results to the correct matrix
        if 'None' not in lc.bazin_features:
            count_surv = count_surv + 1
            print('Survived: ', count_surv)

            # save features to file
            with open(features_file, 'a') as param_file:
                param_file.write(str(lc.id) + ' ' + str(lc.redshift) + ' ' + str(lc.sntype) + ' ')
                param_file.write(str(lc.sncode) + ' ' + str(lc.sample) + ' ')
                for item in lc.bazin_features:
                    param_file.write(str(item) + ' ')
                param_file.write('\n')

    param_file.close()


def run_parallel_bazin(obj: int, feature_list: list, path_to_metadata: str, namespace, num_classes = 8):
    try:
        lc = LightCurve()
        lc.load_plasticc_lc(namespace.md_table, namespace.lc_table, obj, path_to_metadata = path_to_metadata, num_classes=num_classes)
        lc.fit_bazin_all()

        feats = [str(lc.id), str(lc.redshift), str(lc.sntype), str(lc.sample)]
        feats.extend([str(item) for item in lc.bazin_features])
        feats.extend([str(item) for item in lc.bazin_band_cost])

        feature_list.append(feats)

    except ValueError:
        pass    
    
def fit_plasticc_bazin(path_to_data: str, path_to_metadata: str, features_file: str, parallel: bool = False, num_classes = 8):
    """Fit Bazin functions to all filters in training and test samples.

    Parameters
    ----------
    path_to_data: str
        File path for the data file of choice.
    path_to_metadata: str
        File path for the metadata file of choice (either train or test).
    features_file: str
        Path to output file where results should be stored.
    """

    # read file names
    with open(path_to_metadata,'rb') as f1:
        md_table = Table.read(f1,format='csv')
        
    with open(path_to_data,'rb') as f2:
        lc_table = Table.read(f2,format='csv')
        
    final_md_table = md_table[np.isin(md_table['object_id'],np.unique(lc_table['object_id']))]
    lc_list = final_md_table['object_id'].tolist()
    
    # count survivors
    count_surv = 0

    # add headers to files
    with open(features_file, 'w') as param_file:
        param_file.write('id redshift type sample ' + 
                         'uA uB ut0 utfall utrise gA gB gt0 gtfall gtrise rA rB rt0 rtfall rtrise ' + 
                         'iA iB it0 itfall itrise zA zB zt0 ztfall ztrise yA yB yt0 ytfall ytrise ' + 
                         'ucost gcost rcost icost zcost ycost \n')
    if not parallel:
        for obj in lc_list:
            try:
                # fit individual light curves
                lc = LightCurve()
                lc.load_plasticc_lc(final_md_table, lc_table, obj, path_to_metadata = path_to_metadata, num_classes = num_classes)
                lc.fit_bazin_all()

                print(lc_list.index(obj), ' - id:', lc.id, ' - type:', lc.sntype)

                # append results to the correct matrix
                if 'None' not in lc.bazin_features:
                    count_surv = count_surv + 1
                    print('Survived: ', count_surv)

                    # save features to file
                    with open(features_file, 'a') as param_file:
                        param_file.write(str(lc.id) + ' ' + str(lc.redshift) + ' ' + str(lc.sntype) + ' ')
                        param_file.write(str(lc.sample) + ' ')
                        for item in lc.bazin_features:
                            param_file.write(str(item) + ' ')
                        for item in lc.bazin_band_cost:
                            param_file.write(str(item) + ' ')
                        param_file.write('\n')

            except ValueError:
                continue
                
    else: # if parallel
        
        feature_list = []
        
        manager = Manager()
        m_feature_list = manager.list()
        
        ns = manager.Namespace()
        ns.md_table = final_md_table
        ns.lc_table = lc_table
   
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        
        for _ in tqdm(pool.imap_unordered(partial(\
                         run_parallel_bazin,\
                         feature_list=m_feature_list,namespace=ns,path_to_metadata=path_to_metadata, num_classes = num_classes\
                        ),\
                 lc_list), total=len(lc_list)):
            pass
        pool.close()
        pool.join()
        
        feature_list.extend(m_feature_list)
            
        with open(features_file,'a') as param_file:
            np.savetxt(param_file, feature_list, delimiter=' ', fmt='%s')

            
def run_parallel_GP(obj: int, feature_list: list, path_to_metadata: str, namespace, restarts: int = 0, num_classes = 8, kernel_type=1):
    try:
        lc = LightCurve()
        lc.load_plasticc_lc(namespace.md_table, namespace.lc_table, obj, path_to_metadata = path_to_metadata, num_classes = num_classes)
        lc.fit_GP(restarts=restarts, kernel_type=kernel_type)
        
        assert np.all(np.isfinite(lc.GP_features)), 'One or more of the extracted features fails np.isfinite().'

        feats = [str(lc.id), str(lc.redshift), str(lc.sntype), str(lc.sample)]
        feats.extend([str(item) for item in lc.GP_features])
        
        feature_list.append(feats)

    except ValueError:
        pass    

def fit_plasticc_GP(path_to_data: str, path_to_metadata: str, features_file: str, parallel: bool = True, restarts: int = 0, num_classes = 8, kernel_type=1):
    """Fit Bazin functions to all filters in training and test samples.

    Parameters
    ----------
    path_to_data: str
        File path for the data file of choice.
    path_to_metadata: str
        File path for the metadata file of choice (either train or test).
    features_file: str
        Path to output file where results should be stored.
    """

    # read file names
    with open(path_to_metadata,'rb') as f1:
        md_table = Table.read(f1,format='csv')
        
    with open(path_to_data,'rb') as f2:
        lc_table = Table.read(f2,format='csv')
        
    final_md_table = md_table[np.isin(md_table['object_id'],np.unique(lc_table['object_id']))]
    lc_list = final_md_table['object_id'].tolist()
    
    # count survivors
    count_surv = 0

    # add headers to files
    with open(features_file, 'w') as param_file:
        
        if kernel_type == 0 or kernel_type == 1:
            param_file.write('id redshift type sample ' +\
                             'host_photoz host_photoz_err ' +\
                             'pkmag_i pos_flux_ratio max_fr_blue min_fr_blue ' + \
                             'max_fr_red min_fr_red max_dt_yg positive_width time_fwd_max_20 time_fwd_max_50 ' +\
                             'time_bwd_max_20 time_bwd_max_50 time_fwd_max_20_ratio_blue ' +\
                             'time_fwd_max_50_ratio_blue time_bwd_max_20_ratio_blue time_bwd_max_50_ratio_blue ' +\
                             'time_fwd_max_20_ratio_red time_fwd_max_50_ratio_red time_bwd_max_20_ratio_red ' +\
                             'time_bwd_max_50_ratio_red frac_s2n_5 frac_background time_width_s2n_5 ' +\
                             'count_max_center count_max_rise_20 count_max_rise_50 count_max_rise_100 ' +\
                             'count_max_fall_20 count_max_fall_50 count_max_fall_100 total_s2n ' +\
                             'c1 rbf1_t rbf1_l c2 rbf2_t rbf2_l c3 rbf3_t rbf3_l loglike \n')
            
        elif kernel_type == 2:
            param_file.write('id redshift type sample ' +\
                             'host_photoz host_photoz_err ' +\
                             'pkmag_i pos_flux_ratio max_fr_blue min_fr_blue ' + \
                             'max_fr_red min_fr_red max_dt_yg positive_width time_fwd_max_20 time_fwd_max_50 ' +\
                             'time_bwd_max_20 time_bwd_max_50 time_fwd_max_20_ratio_blue ' +\
                             'time_fwd_max_50_ratio_blue time_bwd_max_20_ratio_blue time_bwd_max_50_ratio_blue ' +\
                             'time_fwd_max_20_ratio_red time_fwd_max_50_ratio_red time_bwd_max_20_ratio_red ' +\
                             'time_bwd_max_50_ratio_red frac_s2n_5 frac_background time_width_s2n_5 ' +\
                             'count_max_center count_max_rise_20 count_max_rise_50 count_max_rise_100 ' +\
                             'count_max_fall_20 count_max_fall_50 count_max_fall_100 total_s2n ' +\
                             'c1 matern_t matern_l loglike \n')
       
        
    if not parallel:
        for obj in lc_list:
            try:
                # fit individual light curves
                lc = LightCurve()
                lc.load_plasticc_lc(final_md_table, lc_table, obj, path_to_metadata = path_to_metadata, num_classes = num_classes)
                lc.fit_GP(restarts=restarts, kernel_type=kernel_type)
                
                assert np.all(np.isfinite(lc.GP_features)), 'One or more of the extracted features fails np.isfinite().'

                print(lc_list.index(obj), ' - id:', lc.id, ' - type:', lc.sntype)

                # append results to the correct matrix
                if len(lc.GP_features) > 2:
                    count_surv = count_surv + 1
                    print('Survived: ', count_surv)

                    # save features to file
                    with open(features_file, 'a') as param_file:
                        param_file.write(str(lc.id) + ' ' + str(lc.redshift) + ' ' + str(lc.sntype) + ' ')
                        param_file.write(str(lc.sample) + ' ')
                        for item in lc.GP_features:
                            param_file.write(str(item) + ' ')
                            
                        ### These are now included in GP_features
                        #for item in lc.GP_kernel_params:
                        #    param_file.write(str(item) + ' ')
                        param_file.write('\n')

            except ValueError:
                continue
                
    else: # if parallel
        
        feature_list = []
        
        manager = Manager()
        m_feature_list = manager.list()
        
        ns = manager.Namespace()
        ns.md_table = final_md_table
        ns.lc_table = lc_table
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for _ in tqdm(pool.imap(partial(run_parallel_GP,\
                                        feature_list=m_feature_list, path_to_metadata=path_to_metadata, namespace=ns,\
                                        restarts=0, num_classes=num_classes, kernel_type=kernel_type),\
                                lc_list), total=len(lc_list)):
            pass
        pool.close()
        pool.join()
        
        feature_list.extend(m_feature_list)
        
        print('Successfully fit {0} light curves'.format(len(feature_list)))
            
        with open(features_file,'a') as param_file:
            np.savetxt(param_file, feature_list, delimiter=' ', fmt='%s')
            
def main():
    return None

if __name__ == '__main__':
    main()
