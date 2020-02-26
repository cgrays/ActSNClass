"""
# Author: Alexandre Boucaud and Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 25 January 2018
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
"""

import numpy as np
from scipy.optimize import least_squares


def bazin(time, a, b, t0, tfall, trise):
    """
    Parametric light curve function proposed by Bazin et al., 2009.

    Parameters
    ----------
    time : np.array
        exploratory variable (time of observation)
    a: float
        Normalization parameter
    b: float
        Shift parameter
    t0: float
        Time of maximum
    tfall: float
        Characteristic decline time
    trise: float
        Characteristic raise time

    Returns
    -------
    array_like
        response variable (flux)

    """
    X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))

    return a * X + b


def errfunc(params, time, flux, weights):
    """
    Absolute difference between theoretical and measured flux.

    Parameters
    ----------
    params : list of float
        light curve parameters: (a, b, t0, tfall, trise)
    time : array_like
        exploratory variable (time of observation)
    flux : array_like
        response variable (measured flux)
    weights: array_like
        weights relating to the error in the data

    Returns
    -------
    diff : float
        absolute difference between theoretical and observed flux

    """

    return abs(flux - bazin(time, *params)) * weights


def fit_scipy(time, flux, errors):
    """
    Find best-fit parameters using scipy.least_squares.

    Parameters
    ----------
    time : array_like
        exploratory variable (time of observation)
    flux : array_like
        response variable (measured flux)
    errors: array_like
        error in flux

    Returns
    -------
    output : list of float, float
        best fit parameter values, followed by the value of the cost function for the fit

    """
    flux = np.asarray(flux)
    errors = np.asarray(errors)
    
    sn = np.power(flux / errors, 2.)
    start_point = (sn * flux).argmax()
    
    amp_init = flux[start_point]
    t0_init = time[start_point]
    
    weights = 1. / (1. + errors)
    
    #t0 = time[flux.argmax()] - time[0]
    #guess = [0, 0, t0, 40, -5]
    
    guess = [amp_init, 0, t0_init, 40, -5]

    
    result = least_squares(errfunc, guess, args=(time, flux, weights), method='lm')

    return result.x, result.cost


def main():
    return None


if __name__ == '__main__':
    main()
