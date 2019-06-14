'''
qnttenna.py

version 1.0
last updated: June 2019

by Trevor Arp
Quantum Materials Optoelectronics Laboratory
Department of Physics and Astronomy
University of California, Riverside, USA

All rights reserved.

Description:
This code is published alongside [PAPER TITLE] and performs calculations described in that paper.
The main result is taking an input solar spectrum and calculating the ideal absorption peaks for
a two channel absorber by quieting a noisy antenna and optimizing power bandwidth. See paper for details.

See accompanying README.txt for instructions on using this code

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

import datetime
from timeit import default_timer as timer
from os.path import join, exists
from os import mkdir
import argparse

__version__ = 1.0

'''
Calculates the Delta value described in [PAPER TITLE] as a function of an input solar spectrum.
        /\
Delta = | [a(lambda) - b(lambda)]I(lambda)dlambda
       \/
For the parameter space defined by (lambda_0, Delta lambda, w)

Parameters:
spectrumfile - path to the file containing the solar spectrum to perform the calculation on

w - the width(s) of the absorbers, either a single floating point value or a numpy array of values

lambda_0 - The range of center wavelengths to calculate over, if None (default) the range will be
from the minimum wavelength of the spectrum to 1000nm in 2nm increments. If the range is 'full' it
will use the full range of the input spectrum in 2 nm increments. If it is a tuple with two values,
the range will be between those two values in 2 nm increments.  If it is a numpy array, those values
will be used. User specified ranges exceeding the solar spectrum range will throw a ValueError, user
specified ranges that have poor resolution of spectral features, may yield unreliable values.

delta_lambda - The range of the Delta Lambda parameter space. If None (default) the range will be
from min(w) to 5*max(w) in 1 nm increments. If it is a tuple with two values, the range will be
between those two values in 1 nm increments. If it is a numpy array, those values will be used.
It is recommended that user specified ranges include 2\sqrt(2)w (the theoretical optimum).

autosave - If true will automatically save the output of the calculation to the local directory calculations

optimize - If true (default) will use pathos.multiprocess to parallelize the calculation on multiple
processor cores, if False will limit the calculation to a single processor.

warning - If true (default) will display a warning for excessively long spectrum files

Returns:
1- Array with parameters and calculated values in the form:
[l0, dl, w, A, B, Delta]
Where l0, dl, w are all parameters, and A, B, Delta are the powers in the two channels and
the power bandwidth respectively as 3D numpy arrays with dl for rows, l0 for columns and w for
the third axis

2 - The spectrum data loaded from spectrumfile
'''
def delta_integral(spectrumfile, w, lambda_0=None, delta_lambda=None, autosave=False, optimize=True, warning=True):
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


'''
Gaussian profile of an absorbing channel used in the Noisy Antenna Model

Parameters:
l - numpy array of wavelength values to calculate over
l0 - the center wavelength of the Gaussian
w - the width of the Gaussian

Returns:
Numpy array containing the Gaussian profile of an absorbing channel:
     1
------------ exp[-(l-l0)^2 / w^2]
w*\sqrt(2*pi)
'''
def gauss(l, w, l0):
    return (1.0/(w*np.sqrt(2*np.pi)))*np.exp(-1.0*((l-l0)/w)**2)
#

'''
Finds the optimum peaks for each value of w for power bandwidth cube, using the divider algorithm

Parameters:
- l0, dl, w the parameter space, standard notation
- Delta, the power bandwidth A-B
- npeaks the number of peak pairs to look for (default 2 for relatively  simple spectra)

Returns:
A list of arrays, one for each optimum peak, where each row gives [w[i], l0, dl, Delta] for
the optimum peak at that value of w
'''
def find_optimum_peaks(l0, dl, w, Delta, npeaks=2):
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
# end find_optimum_peaks_div

'''
Loads in the spectrum data file, which should be two columns, first column being wavelength
and second column being irradiance.

Parameters:
spectrumfile - path to the spectrum file

warn - if True (default) will warn the user when the spectral data is over 1000 points

Returns:
Spectral data as a two-column numpy array

'''
def load_spectrum_data(spectrumfile, warn=True):
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

'''
Saves the output of the calculation as text files, either to a local directory or a specified directory.

The output will be saved into a directory, the name of which is determined by the local time when
the function is called (unless dirname is specified).

The format is space separated data files containing:
- 1D arrays with the values of l0, dl, and w in files l0.txt, dl.txt, w.txt
- 2D arrays in a series of files Delta_[INDEX].txt where [INDEX] is the index of w that array corresponds to

If no directory is specified it will by default save to local directory calculations (creating directory if it
doesn't already exist)

Parameters:
calc_data - the output of the calculation in the format of delta_integral, [l0, dl, w, A, B, Delta]

spectrum - is the spectra that was input into this calculation, standard format

directory - The path to a directory to save the files (will be created if it doesn't exist), a local directory
called calculations by default.

dirname - Name of the directory to save the files to, if None (default) will generate a name based
on the local date and time.
'''
def save_calculation(calc_data, spectrum, directory='calculations', dirname=None):
    [l0, dl, w, A, B, Delta] = calc_data
    if not exists(directory):
        mkdir(directory)
    if dirname is None:
        dirname = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    filename = join(directory, dirname)
    if not exists(filename):
        mkdir(filename)
    np.savetxt(join(filename, 'l0.txt'), l0)
    np.savetxt(join(filename, 'dl.txt'), dl)
    np.savetxt(join(filename, 'w.txt'), w)
    np.savetxt(join(filename, 'spectrum.txt'), spectrum)
    for i in range(len(w)):
        np.savetxt(join(filename, 'Delta_' + str(i) + '.txt'), Delta[:,:,i])
#

'''
Multipurpose parallel processing

Takes a function and an array of arguments, evaluates the function with the given arguments for each point,
processing in parallel using ncores number of parallel processes. Returns the results as a numpy ndarray

args_array is the array of arguments to the input function $func. $func can only accept one argument,
but it can be a list or tuple

Will display progress if display is True
'''
def _multiprocess2D(func, args_array, ncores=4, display=True):
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
            print(traceback.print_exc())
            print("_multiprocessing2D: Exiting Process Early")
            pool.terminate()
            break
    tf = timer()
    if display:
        print(" ")
        dt = tf-t0
        print("Computations Completed in: " + str(datetime.timedelta(seconds=dt)))
    return output
#

'''
integrand for a single absorber on the positive side of lambda_0

$l is lambda the integration variable
$l0 is lambda_0 the center wavelength
$dl is Delta lambda, the absorber separation
$w is the Gaussian width of the peaks
$spectrum is an interpolation function, from data describing the spectrum
'''
def _integrand_plus(l, l0, dl, w, spectrum):
    return spectrum(l)*gauss(l, w, l0 + dl/2)
#

'''
integrand for a single absorber on the negative side of lambda_0

$l is lambda the integration variable
$l0 is lambda_0 the center wavelength
$dl is Delta lambda, the absorber separation
$w is the Gaussian width of the peaks
$spectrum is an interpolation function, from data describing the spectrum
'''
def _integrand_minus(l, l0, dl, w, spectrum):
    return spectrum(l)*gauss(l, w, l0 - dl/2)
#

'''
Calculated the power bandwidth, Delta U, for the given spectral data.

Parameters:
$spectral_data, the spectrum to calculate Delta U for, two columns,
first column is wavelength, second column is normalized spectral data. Will calculate Delta U over
the whole range of the spectrum. Spectrum should be fairly free of noise, filter noisy data first.

w is a numpy array containing the peak widths to calculate for

$lstep is the resolution of lambda_0 in nm, default 2 nm

$dlmin, $dlmax, $dlN are the minimum and maximum amounts of Delta lambda to calculate, and
the number of values to calculate Delta lambda for

Returns:
Returns an array with parameters and calculated values in the form:
[l0, dl, w, A, B, Delta]
Where l0, dl, w are all parameters, and ua, ub, du are the powers in the two channels and
the power bandwidth respectively as numpy arrays with dl for rows, l0 for columns and w for
the third axis (2D array is only one values of w is given)
'''
def _power_bandwidth_variance(spectral_data, l0, dl, w, ncores=8):
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

    # Loop over values of w
    t0 = timer()

    for i in range(N):
        # Functions to integrate
        def ua_integral(arg):
            l0 = arg[0]
            dl = arg[1]
            a = l0 - 2*w[i]
            b = l0 + 2*w[i]
            y = quad(_integrand_plus, a, b, args=(l0, dl, w[i], spectrum), full_output=1)
            return y[0]
        # end ua

        def ub_integral(arg):
            l0 = arg[0]
            dl = arg[1]
            a = l0 - 2*w[i]
            b = l0 + 2*w[i]
            y = quad(_integrand_minus, a, b, args=(l0, dl, w[i], spectrum), full_output=1)
            return y[0]
        # end ub

        ts = timer()

        ua[:,:,i] = _multiprocess2D(ua_integral, args_array, ncores=ncores, display=False)
        ub[:,:,i] = _multiprocess2D(ub_integral, args_array, ncores=ncores, display=False)

        for ii in range(rows):
            for jj in range(cols):
                du[ii,jj,i] = np.abs(ua[ii,jj,i] - ub[ii,jj,i])

        tf = timer()
        print(str(i+1)+'/'+str(N)+' complete ' + str(datetime.timedelta(seconds=tf-ts)))
    print('Calculations Complete in ' + str(datetime.timedelta(seconds=tf-t0)))
    return [l0, dl, w, ua, ub, du]
#

'''
Finds the most topologically prominent minima (of the array summed along the x-axis) of d that divide the map into sections.
Then finds the local maxima of each section.

Parameters:
ndivs - The number of divisions to search for
d - the data to search

Returns:
divs - the x-axis indices that divide up the array into sections containing maxima
maxes - a list of tuples (row, col) of the indices of the maxima
'''
def _find_maxes_between_mins(ndivs, d):
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

'''
Command prompt yes or no question
'''
def _yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False
        print("invalid answer")
# end _yes_or_no

if __name__ == "__main__":
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
                mkdir(args.savefile)
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
        wN = float(args.wnumber)

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
