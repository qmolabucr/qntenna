'''
A script to pre-process spectra to make them better suited for speedy qntenna.py calculations
'''
import numpy as np
from qnttenna import load_spectrum_data, _yes_or_no

from scipy.signal import butter, filtfilt

import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists
import argparse

'''
A generic lowpass filter

Parameters:
data is the data to be lowpassed

cutoff is the cutoff frequency in units of the nyquist frequency, must be less than 1

samplerate is the smaple rate in Hz

'''
def lowpass(data, cutoff=0.05, samprate=1.0):
    b,a = butter(2,cutoff/(samprate/2.0), btype='low', analog=0, output='ba')
    data_f = filtfilt(b,a,data)
    return data_f
#

'''
Reduces the resolution of the spectrum, so that the size is less than 1000 points, by interpolating
then resamplign the data.

Parameters:
spectrum is the spectral singal, in the standard two-column format
newsize is the size to resample to (default 1000), if spectrum is maller than this will return the
original array
'''
def reduce_by_interpolation(spectrum, newsize=1000):
    rows, cols = spectrum.shape
    if rows <= newsize:
        print("Warning: Spectral data already less than "+str(newsize)+' data points, no reduction performed')
        return spectrum
    else:
        l1 = np.min(spectrum[:,0])
        l2 = np.max(spectrum[:,0])
        xnew = np.linspace(l1, l2, newsize)
        ynew = np.interp(xnew, spectrum[:,0], spectrum[:,1])
        newspectrum = np.zeros((newsize, 2))
        newspectrum[:,0] = xnew
        newspectrum[:,1] = ynew
        return newspectrum
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the spectral data file")
    parser.add_argument("-sf", "--savefile", help="Location to save output")
    parser.add_argument("-lp", "--lowpass", action="store_true", help="Lowpass filter the spectrum to remove high frequency noise")
    parser.add_argument("-r", "--reduce", action="store_true", help="Re-sample, reducing the resolution of the spectrum so that is 1000 points long")
    parser.add_argument("-rN", "--reduceN", type=int, help="Re-sample, reducing the resolution of the spectrum so that is N points long")
    parser.add_argument("-l1", "--lstart", type=float, help='The start of the range of wavelength, unchanged by default')
    parser.add_argument("-l2", "--lstop", type=float, help='The end of the range of wavelength, unchanged by default')
    parser.add_argument("-v", "--visible", action="store_true", help='Restrict wavelength to visible range 350-800 nm')
    args = parser.parse_args()

    if not args.lowpass and not args.reduce and not args.visible and (args.lstart is None) and (args.lstop is None):
        print("Error: At least optional argument required, use -h or --help to see full list.")
        exit()

    spectrum = load_spectrum_data(args.path, warn=False)
    oldspectrum = np.copy(spectrum)

    if args.visible or (args.lstart is not None) or (args.lstop is not None):
        if args.lstart is not None:
            l1 = args.lstart
        else:
            l1 = np.min(spectrum[:,0])

        if args.lstop is not None:
            l2 = args.lstop
        else:
            l2 = np.max(spectrum[:,0])

        if args.visible:
            l1 = 350
            l2 = 800

        ix1 = np.searchsorted(spectrum[:,0], l1)
        ix2 = np.searchsorted(spectrum[:,0], l2)
        sx = spectrum[ix1:ix2,0]
        sy = spectrum[ix1:ix2,1]
        spectrum = np.zeros((len(spectrum[ix1:ix2,0]), 2))
        spectrum[:,0] = sx
        spectrum[:,1] = sy
    #

    if args.reduce:
        spectrum = reduce_by_interpolation(spectrum)
    elif args.reduceN is not None:
        spectrum = reduce_by_interpolation(spectrum, newsize=args.reduceN)

    if args.lowpass:
        spectrum[:,1] = lowpass(spectrum[:,1])

    if args.savefile is None:
        if _yes_or_no("No savefile specified, overwrite original? (not recommended)"):
            np.savetxt(args.path, spectrum)
        else:
            savefile = str(input('Enter Savefile:'))
            try:
                np.savetxt(savefile, spectrum)
            except OSError:
                print("Cannot save to " + savefile)
    else:
        np.savetxt(args.savefile, spectrum)

    '''
    Display processed spectrum

    Uncomment to show the new spectrum
    '''
    # xinches = 6.0
    # yinches = 5.0
    # fig1 = plt.figure('preprocess_spectrum', figsize=(xinches, yinches), facecolor='w')
    #
    # width = 5
    # xmargin = 0.7
    # height = 4
    # ymargin = 0.5
    # yint = 0.5
    #
    # ax1 = plt.axes([xmargin/xinches, ymargin/yinches, width/xinches, height/yinches])
    #
    # ax1.plot(oldspectrum[:,0], oldspectrum[:,1], '-', color='grey', label='raw')
    # ax1.plot(spectrum[:,0], spectrum[:,1], '-', color='k', label='processed')
    # ax1.set_xlabel('Wavelength (nm)')
    # ax1.set_ylabel('Irradiance (args.)')
    # ax1.legend()
    #
    # fig1.show()
    # plt.show()

#
