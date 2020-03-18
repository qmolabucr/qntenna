Instructions for using the qnttenna.py code

version 1.1
last updated: March 2020

by Trevor Arp
Quantum Materials Optoelectronics Laboratory
Department of Physics and Astronomy
University of California, Riverside, USA

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

DESCRIPTION:

This code is published alongside 'Quieting a noisy antenna reproduces photosynthetic light harvesting spectra'
and performs calculations described in that paper. The main result is taking an input solar spectrum and calculating
the ideal absorption peaks for a two channel absorber by quieting a noisy antenna and optimizing power bandwidth.
See the paper for model details.

For any non-ideal spectrum the optimization calculation is nontrivial, especially at small w. There may be multiple
solutions that are nearly degenerate, or spurious solutions not allowed due to operable bandwidth considerations.
It is strongly recommended that the user read supplement section S2 in depth to understand the complexities of
this calculation before using this code.

INSTALLATION INSTRUCTIONS:

qnttenna requires python (version 3) and the following python packages:

numpy version 1.13+
scipy version 1.1+
pathos version 0.2+
matplotlib 2.0+ (for display scripts, not required for qnttenna module, but recommended)

All of these can be installed using python's pip package manager.

To test the installation run the blackbody_qnttenna.py program from the command line,
> python blackbody_qnttenna.py
if it completes and displays the Delta calculation for the blackbody spectrum then the installation is working.

COMMAND LINE INTERFACE INSTRUCTIONS:

qnttenna.py was written with a command line interface, full options can be seen using:
> python qnttenna.py -h
The most basic use of the program is to process a file of spectrum data (see data format below). For example,
to process the included NREL-etr.txt file use the following command:
> python qnttenna.py spectra\NREL-etr-visible.txt
This will perform the calculation for absorber widths of 5,10,15,20,25,30 nanometers.

To specify a different range of spectral widths use the -w1, -w2 and -wn arguments, which define a range in w [w1,w2] with
wn values including the endpoints. For example, to perform the calculation for w = 5,6,7,8,9,10 nm, use the following command:
>python qnttenna.py spectra\NREL-etr-visible.txt -w1 5 -w2 10 -wn 6

To perform the calculation for a single absorber width simply set -w1 and -w2 to the same value, for example for w = 5.5
> python qnttenna.py spectra\NREL-etr-visible.txt -w1 5.5 -w2 5.5

By default, the output will be saved to a local 'calculations' directory with a filename based on the date and time of the
calculation. To save to a different directory use the following argument:
> python qnttenna.py spectra\NREL-etr-visible.txt -sf PATH\TO\SAVE\DIRECTORY

In addition to qnttenna.py four other python scripts were included

- blackbody_qnttenna.py : A script that calculates and displays the results for a blackbody spectrum. Use this to test
the installation, as follows:
> python blackbody_qnttenna.py

- load_saved_calculation.py : A script to load and display the output of the calculation. Use as follows:
> python load_saved_calculation.py PATH\TO\SAVE\DIRECTORY -w WIDTH

- preprocess_spectrum.py : A script to process spectral data to make the calculation faster. Spectral datafiles with
more than 1000 data points may take a long time to compute. qnttenna.py will display a warning for long data files.
If the user doesn't want long processing times, we've included a script that will process spectral data to make it
faster. There are several options, all of which can be viewed using
> python preprocess_spectrum.py -h
but for long files we recommended the following option:
> python preprocess_spectrum.py PATH\TO\FILE --reduce -sf PATH\TO\NEWFILE
The --reduce option will re-sample the data so that it is 1000 data points long.

- discrete_toy_model.py: An implementation of the Quieting and Noisy Antenna model in the discrete limit. See section
S1.3 of the supplementary materials for more information about the model. Has three command line arguments corresponding
to different parameters of the calculation. -n is the number of absorbing pairs, -ept is the number of potential absorption events
per timestep and -p is phi, the free parameter that sets the probabilities.


INPUT DATA FORMAT:

Input spectral data should have the following format. The data should be in two columns, either space
separated or comma separated. Comma separated files must have the file extension .csv. Excel files
will not work but can easily be saved in the .csv format. The first column should be wavelength, and the
second column should be irradiance. Units of wavelength are nanometers, and units of irradiance are arbitrary.

Long or excessively dense data files will take a long time to compute (on the order of hours to days). Noisy
data will also increase the processing time, and may result in erroneous results for small values of w For
fast computation, restrict the spectrum to the region of interest, decrease resolution or filter noise.
We have included a python script called preprocess_spectrum.py that will prepare the data appropriately (see above).

We have included several data files in the 'spectra' folder.
- 'BB-5500K.txt' the spectrum of a blackbody at T = 5500 K, for testing.
- 'NREL-full.txt', 'leaffilter3.txt' and '2munderwater.txt' spectra were used to calculate the results in Fig. 2G, H, I respectively.
- 'NREL-visible.txt' which is also the NREL ASTM ETR spectrum but restricted to the visible range
In addition in the subfolder "raw_spectra" there are several unfiltered data files used to generate the above spectra and show the
underwater solar spectra shown in Fig 4A. References for data can be found in the main text of the paper and a description
of how the leaffilter3 data was obtained and the underwater data was calculated can be found in the supplementary materials section S3.

CODING INSTRUCTIONS:

Users proficient at coding may wish to incorporate the qnttenna.py module into their own scripts.
qnttenna.py module contains four public functions (following the standard conventions  that function names
beginning with an underscore are considered private). Below are the basic descriptions of these functions,
for more details look at the comments in qnttenna.py

NOTE: Given that qnttenna.py uses the pathos.multiprocess module, a form of python's multiprocessing module,
the calculation is unstable outside an "if __name__ == 'main'" block. If delta_integral starts throwing weird
errors check that it is inside an "if __name__ == 'main'".

delta_integral(spectrumfile, w):
Calculates the Delta value described in 'Quieting a noisy antenna reproduces photosynthetic light harvesting spectra'
as a function of an input solar spectrum. The integral is detailed in  Seaction S1.3 of the supplementary materials.

gauss(l,w,l0):
Gaussian profile of an absorbing channel used in the calculation.

find_optimum_peaks(l0, dl, w, Delta, npeaks=2):
Finds npeaks sets of peaks in regions seperated by the npeaks-1 highest order local minima. Peaks correspond
to the maxima of Delta in the parameter space (l0, dl, w). Requires a number of peaks to search for (default 2).
This function finds the largest peaks, not accounting for any complexity in the optimization, read Supplement
Section S2 for a detailed discussion of how to treat the optimization peaks.

load_spectrum_data(spectrumfile):
Loads a spectrum file that is in standard format (see input data format above) and warns if the file is excessively
long or in the wrong format.

save_calculation(calc_data, spectrum):
Saves the output of the calculation as text files, to either a local "calculations" directory or a specified directory.

load_calculation(directory):
Loads the output of the calculation as text files, from either a local "calculations" directory or a specified directory.
