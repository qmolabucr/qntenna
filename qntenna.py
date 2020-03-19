'''
qntenna.py

version 1.1
last updated: March 2020

by Trevor Arp
Quantum Materials Optoelectronics Laboratory
Department of Physics and Astronomy
University of California, Riverside, USA

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Description:
This code is published alongside 'Quieting a noisy antenna reproduces photosynthetic light harvesting
spectra' and performs calculations described in that paper. The main result is taking an input solar
spectrum and calculating the ideal absorption peaks for a two channel absorber by quieting a noisy
antenna and optimizing power bandwidth. See paper for details.

See accompanying README.txt for instructions on using this code.

Software Dependencies:
Python version 3.5+
Numpy version 1.13+
Scipy version 1.0+
Pathos version 0.2+
'''

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.signal import argrelmin, peak_prominences

from pathos.multiprocessing import ProcessingPool as Pool
from pathos.helpers import freeze_support

import datetime
from timeit import default_timer as timer
from os.path import join, exists
from os import makedirs
import argparse

__version__ = 1.1

def delta_integral(spectrumfile, w, lambda_0=None, delta_lambda=None, autosave=False, optimize=True, warning=True):
    '''
    Calculates the Delta^op value described in 'Quieting a noisy antenna reproduces
    photosynthetic light harvesting spectra' as a function of an input solar spectrum.
               /\
    Delta^op = | [a(lambda) - b(lambda)]I(lambda)dlambda
              \/
    For the parameter space defined by (lambda_0, Delta lambda, w)

    Warning: For any non-ideal spectrum the optimization landscape is nontrivial, especially
    at small w. There may be multiple maxima in the calculation that are nearly degenerate,
    or spurious optima not allowed due to operable bandwidth considerations. It is
    recommended that the user read supplement section S1 and S2 in depth to understand
    the complexities of this calculation.

    Args:
        spectrumfile : path to the file containing the solar spectrum to perform the calculation on
        w : the width(s) of the absorbers, either a single floating point value or
            a numpy array of values
        lambda_0 : The range of center wavelengths to calculate over, if None (default) the
            range will be from the minimum wavelength of the spectrum to 1000nm in 2nm
            increments. If it is a tuple with two values, the range will be between those
            two values in 2 nm increments.  If it is a numpy array, those values will be
            used. User specified ranges exceeding the solar spectrum range will throw a
            ValueError, user specified ranges that have poor resolution of spectral features,
            may yield unreliable values.
        delta_lambda : The range of the Delta Lambda parameter space. If None (default) the
            range will be from min(w) to 5*max(w) in 1 nm increments. If it is a tuple with
            two values, the range will be between those two values in 1 nm increments. If
            it is a numpy array, those values will be used. It is recommended that user
            specified ranges include 2\sqrt(2)w (the theoretical optimum).
        autosave : If true will automatically save the output of the calculation to the
            local directory calculations.
        optimize - If true (default) will use pathos.multiprocess to parallelize the
            calculation on multiple processor cores, if False will limit the calculation
            to a single processor.
        warning - If true (default) will display a warning for excessively long spectrum files

    Returns:
        A tuple containing the calulated values and spectrum data, (calc_data, spectrum_data).
            Where the calc_data is an array with [l0, dl, w, Delta^op].
            Where l0, dl, w are all parameters, and Delta^op is the power bandwidth as a
            3D numpy arrays with dl for rows, l0 for columns and w for the third axis.
    '''
    spectral_data = load_spectrum_data(spectrumfile, warn=warning)

    # Set the w parameter
    if type(w) is not np.ndarray:
        if type(w) is list:
            w = np.array(w)
        else:
            try:
                float(w)
            except ValueError:
                raise ValueError("Width parameter, w, must be a numpy array or float")
            w = [w]


    # Set the lambda_0 parameter
    if lambda_0 is None:
        l1 = int(np.min(spectral_data[:,0])+1)
        l2 = min([1000, int(np.max(spectral_data[:,0]))])
        l0 = np.arange(l1, l2, 2)
    elif type(lambda_0) is tuple:
        l1 = lambda_0[0]
        l2 = lambda_0[1]
        l0 = np.arange(l1, l2, 2)
    else:
        if type(lambda_0) is np.ndarray:
            l0 = lambda_0
            l1 = np.min(l0)
            l2 = np.max(l0)
        else:
            raise ValueError("parameter lambda_0 must be a numpy array, tuple or None")
    if np.min(l1) < np.min(spectral_data[:,0]) or np.max(l2) > np.max(spectral_data[:,0]):
        raise ValueError("all values of parameter lambda_0 must be within the solar spectrum data")

    # Set Delta lambda parameter
    if delta_lambda is None:
        mindl = np.min(w)
        maxdl = 5.0*np.max(w)
        dl = np.arange(mindl, maxdl, 1)
    elif type(delta_lambda) is tuple:
        mindl = delta_lambda[0]
        maxdl = delta_lambda[1]
        dl = np.arange(mindl, maxdl, 1)
    else:
        if type(delta_lambda) is np.ndarray:
            dl = delta_lambda
        else:
            raise ValueError("parameter delta_lambda must be a numpy array, tuple or None")

    print("Beginning calculations")
    if optimize:
        calc_data = _power_bandwidth_variance(spectral_data, l0, dl, w)
    else:
        calc_data = _power_bandwidth_variance(spectral_data, l0, dl, w, ncores=1)

    if autosave:
        save_calculation(calc_data, spectral_data)

    return calc_data, spectral_data
#

def gauss(l, w, l0):
    '''
    Gaussian profile of an absorbing channel used in the Noisy Antenna Model:
         1
    ------------ exp[-(l-l0)^2 / w^2]
    w*\sqrt(2*pi)

    Args:
        l : numpy array of wavelength values to calculate over
        l0 : the center wavelength of the Gaussian
        w : the width of the Gaussian

    Returns:
        Numpy array containing the Gaussian profile of an absorbing channel.
    '''
    return (1.0/(w*np.sqrt(2*np.pi)))*np.exp(-1.0*((l-l0)/w)**2)
#

def find_optimum_peaks(l0, dl, w, Delta, npeaks=2):
    '''
    Finds the optimum peaks for each value of w for power bandwidth cube, using the
    divider algorithm

    Warning: For any non-ideal spectrum the optimization landscape is nontrivial,
    especially at small w. There may be multiple maxima in the calculation that are
    nearly degenerate, or spurious optima not allowed due to operable bandwidth
    considerations. It is recommended that the user read supplement section S2 in
    depth to understand the complexities of this calculation. This function finds the
    highest peaks in the optimization landscape, which is the start, but not the end,
    of any analysis.

    Args:
        l0 : the \lambda_{0} parameter space, standard notation
        dl : the \Delta \lambda parameter space, standard notation
        w : the width parameter space, standard notation
        Delta : the power bandwidth optimization parameter
        npeaks : the number of peak pairs to look for (default 2 for relatively  simple spectra)

    Returns:
        A list of arrays, one for each optimum peak, where each row gives
            [w[i], l0, dl, Delta] for the optimum peak at that value of w
    '''
    N = w.size
    peaks = []
    for j in range(npeaks):
        peaks.append(np.zeros((N, 4)))

    for i in range(N):
        try:
            divs, maxes = _find_maxes_between_mins(npeaks-1, Delta[:,:,i])
            for j in range(npeaks):
                ix0, ix1 = maxes[j]
                peaks[j][i, 0] = w[i]
                peaks[j][i, 1] = l0[ix1]
                peaks[j][i, 2] = dl[ix0]
                peaks[j][i, 3] = Delta[ix0, ix1, i]
        except IndexError:
            print('Error at w= ' + str(w[i]))
    return peaks
# end find_optimum_peaks

def load_spectrum_data(spectrumfile, warn=True):
    '''
    Loads in the spectrum data file, which should be two columns, first column being
    wavelength and second column being irradiance.

    Args:
        spectrumfile - path to the spectrum file.
        warn - if True (default) will warn the user when the spectral data is over
            1000 points.

    Returns:
        Spectral data as a two-column numpy array
    '''
    s = spectrumfile.split('.')
    if s[len(s)-1] == 'csv':
        spectral_data = np.loadtxt(spectrumfile, delimiter=',')
    else:
        spectral_data = np.loadtxt(spectrumfile)
    rows, cols = spectral_data.shape
    if cols != 2:
        raise IOError("Invalid format, spectrum file must have two columns")

    if rows > 1000 and warn:
        helpstr = "Warning: Spectral data exceeding 1000 data points may take a long time to process. "
        helpstr += "If you are not okay with this we recommend using our preprocess_spectrum script "
        helpstr += "to reduce the file size. Use as follows: "
        print(helpstr)
        print("python preprocess_spectrum [PATH TO FILE] --reduce --savefile [NEW FILE]")
        if not _yes_or_no("Proceed anyway?"):
            exit()
    return spectral_data
#

def save_calculation(calc_data, spectrum, directory=None):
    '''
    Saves the output of the calculation as text files, either to a local directory or a
    specified directory.

    The output will be saved into a directory, the name of which is determined by
    the local time when the function is called (unless dirname is specified).

    The format is space separated data files containing:
    - 1D arrays with the values of l0, dl, and w in files l0.txt, dl.txt, w.txt
    - 2D arrays in a series of files Delta_[INDEX].txt where [INDEX] is the index of w
      that array corresponds to

    If no directory is specified it will by default save to local directory calculations
    (creating directory if it doesn't already exist)

    Args:
        calc_data : the output of the calculation in the format of delta_integral:
            [l0, dl, w, A, B, Delta^op]
        spectrum : is the spectra that was input into this calculation, standard format
        directory : The path to a directory to save the files (will be created if it doesn't
            exist), by default a local directory called calculations with an automatically
            generated subfolder based on the local date and time.
    '''
    [l0, dl, w, Delta] = calc_data
    if directory is None:
        dirname = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        filename = join('calculations', dirname)
    else:
        filename = directory
    if not exists(filename):
        makedirs(filename)
    np.savetxt(join(filename, 'l0.txt'), l0)
    np.savetxt(join(filename, 'dl.txt'), dl)
    np.savetxt(join(filename, 'w.txt'), w)
    np.savetxt(join(filename, 'spectrum.txt'), spectrum)
    for i in range(len(w)):
        np.savetxt(join(filename, 'Delta_' + str(i) + '.txt'), Delta[:,:,i])
#

def load_calculation(directory):
    '''
    Loads the calculation data found in directory, saved in the standard format by
    qntenna.save_calculation. If it can't find the directory will search local 'calculations'
    directory for autosave.

    Returns:
        A tuple containing the calulated values and spectrum data, (calc_data, spectrum_data).
            calc_data is an array with [l0, dl, w, Delta^op]. Where l0, dl, w are all
            parameters, and Delta^op is the power bandwidth as 3D numpy array with dl for
            rows, l0 for columns and w for the third axis.
    '''
    if not exists(directory):
        if exists(join('calculations', directory)):
            directory = join('calculations', directory)
        else:
            print('Error could not find ' + str(directory))
            return
    l0 = np.loadtxt(join(directory, 'l0.txt'))
    dl = np.loadtxt(join(directory, 'dl.txt'))
    w = np.loadtxt(join(directory, 'w.txt'))
    try: # Fix issue with calculations of a single w value
        w[0]
    except IndexError:
        w = np.array([w])
    spectrum = np.loadtxt(join(directory, 'spectrum.txt'))
    rows = dl.size
    cols = l0.size
    N = w.size
    Delta = np.zeros((rows, cols, N))
    for i in range(N):
        Delta[:,:,i] = np.loadtxt(join(directory, 'Delta_' + str(i) + '.txt'))
    return [l0, dl, w, Delta], spectrum
#

def _multiprocess2D(func, args_array, ncores=4, display=True):
    '''
    Multipurpose parallel processing

    Takes a function and an array of arguments, evaluates the function with the given
    arguments for each point, processing in parallel using ncores number of parallel
    processes.

    WARNING: needs to be protected by a if __name__ == "__main__" block or else
    multiprocessing.pool will have problems.

    Args:
        func : The function to evaluate, can only accept one argument but it can be a list
            or tuple
        args_array is the array of arguments to the input function $func.
        ncores : The number of nodes to pass to multiprocessing.Pool
        display : Will display progress if true.

    Returns:
        The results of the calculation as a numpy ndarray.
    '''
    pool = Pool(nodes=ncores)
    rows = len(args_array)
    cols = len(args_array[0])
    output = np.zeros((rows, cols))
    if rows > 10:
        disp_rows = np.arange(rows/10, rows, rows/10)
    else:
        disp_rows = np.arange(1, rows, 1)
    if display:
        print("Parallel Processing Started with " + str(ncores) + " subprocesses")
    t0 = timer()
    for i in range(rows):
        worker_args = []
        for j in range(cols):
            worker_args.append(args_array[i][j])
        try:
            out = pool.map(func, worker_args)
            for j in range(cols):
                output[i,j] = out[j]
            if display and i in disp_rows:
                print(str(round(100*i/float(rows))) + "% Complete")
        except Exception as e:
            print("Exception in _multiprocessing2D: Cannot Process")
            print("_multiprocessing2D: Exiting Process Early")
            pool.terminate()
            raise e
    tf = timer()
    if display:
        print(" ")
        dt = tf-t0
        print("Computations Completed in: " + str(datetime.timedelta(seconds=dt)))
    return output
#

class _integrator():
    '''
    Integration implemented as a class to prevents some issues from occuring in parallel
    processing. For info see this thread: https://github.com/uqfoundation/pathos/issues/118
    '''
    def __init__(self, w, spectrum):
        self.integral = quad
        self.w = w
        self.spectrum = spectrum
        self.sqrt = np.sqrt
        self.exp = np.exp
        self.pi = np.pi
    #

    def setw(self, w):
        self.w = w
    #

    def _gauss(self, l, w, l0):
        '''
        Gaussian profile of an absorbing channel used in the Noisy Antenna Model:
             1
        ------------ exp[-(l-l0)^2 / w^2]
        w*\sqrt(2*pi)

        Args:
            l : numpy array of wavelength values to calculate over
            l0 : the center wavelength of the Gaussian
            w : the width of the Gaussian

        Returns:
            Numpy array containing the Gaussian profile of an absorbing channel.
        '''
        return (1.0/(w*self.sqrt(2*self.pi)))*self.exp(-1.0*((l-l0)/w)**2)
    #

    def _integrand_plus(self, l, l0, dl):
        '''
        Integrand for a single absorber on the positive side of lambda_0

        Args:
            l : is lambda the integration variable
            l0 : is lambda_0 the center wavelength
            dl : is Delta lambda, the absorber separation
            spectrum : is an interpolation function, from data describing the spectrum
        '''
        return self.spectrum(l)*self._gauss(l, self.w, l0 + dl/2)
    #

    def _integrand_minus(self, l, l0, dl):
        '''
        Integrand for a single absorber on the negative side of lambda_0

        Args:
            l : is lambda the integration variable
            l0 : is lambda_0 the center wavelength
            dl : is Delta lambda, the absorber separation
            spectrum : is an interpolation function, from data describing the spectrum
        '''
        return self.spectrum(l)*self._gauss(l, self.w, l0 - dl/2)
    #

    def ua_integral(self, arg):
        l0 = arg[0]
        dl = arg[1]
        a = l0 - 2*self.w
        b = l0 + 2*self.w
        y = self.integral(self._integrand_plus, a, b, args=(l0, dl), full_output=1)
        return y[0]
    # end ua

    def ub_integral(self, arg):
        l0 = arg[0]
        dl = arg[1]
        a = l0 - 2*self.w
        b = l0 + 2*self.w
        y = self.integral(self._integrand_minus, a, b, args=(l0, dl), full_output=1)
        return y[0]
    # end ub

def _power_bandwidth_variance(spectral_data, l0, dl, w, ncores=8):
    '''
    Calculated the power bandwidth optimization parameter for the given spectral data.

    Args:
    spectral_data : the spectrum to calculate Delta^op for, two columns, first column is
        wavelength, second column is normalized spectral data. Will calculate Delta^op over
        the whole range of the spectrum. Spectrum should be fairly free of noise, filter
        noisy data first.
    w : a numpy array containing the peak widths to calculate
    lstep : the resolution of lambda_0 in nm, default 2 nm
    dlmin : the minimum Delta lambda to calculate
    dlmax :the maximum Delta lambda to calculate
    dlN : the number of values to calculate Delta lambda for

    Returns:
    An array with parameters and calculated values in the form:
        [l0, dl, w, Delta^op]
        Where l0, dl, w are all parameters, and Delta^op is the power bandwidth optimization
        parameter respectively as numpy arrays with dl for rows, l0 for columns and w for
        the third axis (2D array is only one values of w is given)
    '''
    sx = spectral_data[:,0]
    sy = spectral_data[:,1]

    cols = len(l0)
    rows = len(dl)
    N = len(w)

    ua = np.zeros((rows, cols, N))
    ub = np.zeros((rows, cols, N))
    du = np.zeros((rows, cols, N))

    spectrum = interp1d(sx, sy, kind='cubic', bounds_error=False, fill_value=np.min(sy))
    args_array = []
    for i in range(rows):
        args_array.append([])
        for j in range(cols):
            args_array[i].append([l0[j], dl[i]])

    int = _integrator(w[0], spectrum)

    # Loop over values of w
    t0 = timer()

    for i in range(N):
        ts = timer()
        int.setw(w[i])
        ua[:,:,i] = _multiprocess2D(int.ua_integral, args_array, ncores=ncores, display=False)
        ub[:,:,i] = _multiprocess2D(int.ub_integral, args_array, ncores=ncores, display=False)
        for ii in range(rows):
            for jj in range(cols):
                du[ii,jj,i] = np.abs(ua[ii,jj,i] - ub[ii,jj,i])
        tf = timer()
        print(str(i+1)+'/'+str(N)+' complete ' + str(datetime.timedelta(seconds=tf-ts)))
    Pool(nodes=ncores).clear() # Because pathos is designed to leave Pools running, and sometimes doesn't get rid of them after the caluculation is complete
    print('Calculations Complete in ' + str(datetime.timedelta(seconds=tf-t0)))
    return [l0, dl, w, du]
#

def _find_maxes_between_mins(ndivs, d):
    '''
    Finds the most topologically prominent minima (of the array summed along the x-axis)
    of d that divide the map into sections. Then finds the local maxima of each section.

    Args:
        ndivs : The number of divisions to search for
        d : the data to search

    Returns:
        A tuple containing (divs, maxes) where:
            divs is the x-axis indices that divide up the array into sections containing maxima
            maxes is a list of tuples (row, col) of the indices of the maxima
    '''
    rows, cols = d.shape

    # Find the minima along the x-axis
    sd = np.sum(d, axis=0)
    divs = argrelmin(sd)[0]
    prom, lbase, rbase  = peak_prominences(-1.0*sd, divs)

    # Pull out only the nmins most topologically prominent minima
    args = np.argsort(prom)
    prom = prom[args]
    divs = divs[args]
    st = int(len(divs) - ndivs)
    divs = divs[st:]

    divs = np.sort(divs) # put back in sorted order
    ix1 = 0
    maxes = []
    for i in range(ndivs+1): # Find the maxima in each division
        if i == ndivs:
            ix2 = cols
        else:
            ix2 = divs[i]
        ixf = np.argmax(d[:,ix1:ix2])
        lr, lc = np.unravel_index(ixf, d[:,ix1:ix2].shape)
        maxes.append((lr, ix1+lc))
        ix1 = ix2
    return divs, maxes
# end _find_maxes_between_mins

def _yes_or_no(question):
    '''
    Command prompt yes or no question
    '''
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False
        print("invalid answer")
# end _yes_or_no

if __name__ == "__main__":
    freeze_support() # To prevent pathos from having issues when run from windows command line

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the spectral data file")
    parser.add_argument("-sf", "--savefile", help="Path to directory to save output files, by default will save to local calculations directory")
    parser.add_argument("-w1","--width1", type=float, help="The beginning of the range of absorber widths (default 5 nm)")
    parser.add_argument("-w2","--width2", type=float, help="The end of the range of absorber widths (default 30 nm)")
    parser.add_argument("-wn", "--wnumber", type=float, help='The number of points in the range in w from -w1 to -w2 (default 6)')
    parser.add_argument("-l1", "--lstart", type=float, help='The start of the range of center wavelength')
    parser.add_argument("-l2", "--lstop", type=float, help='The end of the range of center wavelength')
    parser.add_argument("--limit", action="store_true", help="Limits calculation to a single processor core.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppresses warning for long spectum files")
    args = parser.parse_args()


    autosave = (args.savefile is None)
    if args.savefile is None:
        autosave = True
    else:
        if not exists(args.savefile):
            if _yes_or_no(args.savefile + " does not exist. Create directory?"):
                makedirs(args.savefile)
            else:
                print("Invalid savefile, using default savefile")
                autosave = True
    #

    if args.width1 is None:
        w1 = 5
    else:
        w1 = float(args.width1)

    if args.width2 is None:
        w2 = 30
    else:
        w2 = float(args.width2)

    if args.wnumber is None:
        wN = 6
    else:
        wN = int(args.wnumber)

    if w1 == w2:
        wN = 1

    w = np.linspace(w1, w2, wN)

    if args.lstart is not None and args.lstop is not None:
        lambda0 = (args.lstart, args.lstop)
    elif (args.lstart is not None and args.lstop is None) or (args.lstop is not None and args.lstart is None):
        print("user specified range of lambda_0 requires both --lstart and --lstop arguments")
        print("Invalid arguments, exiting program")
        exit()
    else:
        lambda0 = None

    optimize = not args.limit

    warn = not args.quiet

    calc_data, spectrum = delta_integral(args.path, w, lambda_0=lambda0, autosave=autosave, optimize=optimize, warning=warn)
    if not autosave:
        save_calculation(calc_data, spectrum, directory=args.savefile)
