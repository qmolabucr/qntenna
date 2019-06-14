'''
discrete_toy_model.py

version 1.0
last updated: June 2019

by Trevor Arp
Quantum Materials Optoelectronics Laboratory
Department of Physics and Astronomy
University of California, Riverside, USA

All rights reserved.

Description:
A visualization of a simple discrete model of a finite number of absorbers, see section S1.3 of the
supplementary materials for more information.

See accompanying README.txt for instructions on using this code
'''

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import warnings
from scipy.optimize import curve_fit

from random import randint
import argparse

seed = randint(0,2**31)

'''
Defines the relationship between the probabilities Pa and Pb bsed on the two-channel regulating
condition: Pa*Ua + Pb*Ub = Omega

Returns:
- slope, intercept : The slope and intercept of the line relating Pa and Pb, i.e. Pa = slope*Pb + intercept
- Pmin, Pmax, the minimum and maximum values of Pb
'''
def P_regulation_line(Ub, Ua, Omega):
    Pmin = (Ub-Omega)/(Ub-Ua)
    Pmax = Omega / Ua
    slope = -(Ua / Ub)
    intercept = (Omega / Ub)
    return slope, intercept, Pmin, Pmax
#

'''
Samples a uniform distribution with two probabilities, returns 1 with probability p, returns -1 with
probability q returns 0 with probability 1-p-q.
'''
def pq_sample(p, q, N=100):
    if p+q > 1.0001:
        print("WARNING pq_sample: Sum of input probabilities (" +str(p+q) + ") is greater than 1")
    r = np.random.uniform(size=N)
    out = np.zeros(N)
    for i in range(len(r)):
        if r[i] <= p :
            out[i] = 1
        elif r[i] <= p+q:
            out[i] = -1
    #
    return out
#

'''
Averages a timeseries over N absorption events, resulting in a series with N events per timestep
'''
def finite_avg(series, N):
    M = len(series)
    if M//N == (M-1)//N:
        l = M//N + 1
    else:
        l = M//N
    output = np.zeros(l)
    for i in range(0, M, N):
        output[i//N] = np.mean(series[i:i+N])
    return output
# end finite_avg

'''
The simulation of N absorbing pairs, with the regulation condition enforced

Parameters:
- N the number of absorbing paris
- Ub and Ua, the low and high values of the absorbing pairs
- Omega, the equilibrium value of the absorbing pairs
- phi, the free parameters that sets the probabilities in the equilibrium condition.
- modtype, the type of external modulation, either 'random' or 'constant'
- amp, the amplitude of the external modulation
- length, the number of absorption events to calculate for
- extmodlength, the length of the external modulation in terms of the number of absorbing events
'''
def N_regulated_absorbers(N, Ub, Ua, Omega, phi, modtype='random', amp=0.25, length=500, extmodlength=100):
    m, b, Pb_min, Pb_max = P_regulation_line(Ub, Ua, Omega)
    Pb = Pb_min*(1.0-phi) + Pb_max*phi
    Pa = m*Pb + b

    length = int(length)
    E = np.zeros(length)
    np.random.seed(seed)
    if length//extmodlength == (length-1)//extmodlength:
        extlen = length//extmodlength + 1
    else:
        extlen = length//extmodlength

    if modtype == 'constant':
        extmod = np.zeros(extlen)
    else:
        extmod = np.zeros(extlen)
        extmod[1:] = np.random.uniform(-1.0, 1.0, extlen-1)

    for j in range(N):
        r = pq_sample(Pa, Pb, length)
        for i in range(0,length):
            if r[i] == 1:
                E[i] = E[i] + Ub + amp*Ub*extmod[i//extmodlength]
            elif r[i] == -1:
                E[i] = E[i] + Ua + amp*Ua*extmod[i//extmodlength]
    return E
# N_regulated_absorbers

'''
A one dimensional Gaussian function given by

f(x,y) = A*Exp[ -(x-x0)^2/(2*sigma^2)]
'''
def gaussfunc(x, A, x0, sigma):
    return A*np.exp(-(x-x0)**2/(2*sigma**2))
# end gauss

'''
Fits data x and y to a symmetric exponential function defined by gaussfunc

Returns the fit parameters.

Default parameters:
p0 is the starting fit parameters, leave as-1 to estimate starting parameters from data
'''
def gauss_fit(x, y, p0=-1):
    l = len(y)
    if len(x) != l :
        print("Error gauss_fit: X and Y data must have the same length")
        return
    if p0 == -1:
        a = np.max(y)
        sigma = 1.0
        x0 = x[int(len(x)/2)]
        p0=(a, sigma, x0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, plconv = curve_fit(gaussfunc, x, y, p0=p0)
    except Exception as e:
        p = p0
    return p
# end gauss_fit

'''
Takes a histogram and returns the output of np.histogram, plus the average bin values

Parameters:
$bins is passed directly to numpy.histogram
$density is the density parameter to pass to histogram, if true normalizes to probability density

Returns:
- histogram values
- The average values of the bins (not the bin edges)
'''
def avg_bin_histogram(data, bins, density=False):
    hist, edges = np.histogram(data, bins=bins, density=density)
    M = len(edges)
    avg_bins = np.zeros(M-1)
    for i in range(1,M):
        avg_bins[i-1] = (edges[i-1] + edges[i])/2
    return hist, avg_bins
# avg_bin_histogram

'''
Plots the histrogram of the excitation energy.

Parameters:
- ax1, the matplotlib axes to plot on
- sequence is the energy timeseries to calculate the histogram for
- Ua, Ub, N the energy values and number of absorbers, to display on the y-axis.
- fit, the guassian fit to the intial histogram, will calculate if None
- nbins, the number of histogram bins
'''
def excitation_histogram(ax1, sequence, Ua, Ub, N, fit=None, nbins=34):
    x1 = 0
    x2 = 2*N
    freq_l, vals_l = avg_bin_histogram(sequence, nbins, density=True)
    pms = gauss_fit(vals_l, freq_l, p0=(0.2, N, N*0.25))

    bin_width = (x2-x1)/nbins
    edges = np.linspace(x1, x2, nbins)
    M = len(edges)
    vals = np.zeros(M-1)
    for i in range(1,M):
        vals[i-1] = (edges[i-1] + edges[i])/2

    n, bns, patches = ax1.hist(sequence, edges, orientation='horizontal', color='y', alpha=0.75, density=True, edgecolor='black', linewidth=0.5)
    patches = list(patches)
    for i in range(M-1):
        if vals[i] < N-bin_width/2:
            patches[i].set_facecolor('b')
        elif vals[i] > N+bin_width/2:
            patches[i].set_facecolor('r')

    ax1.set_ylim(x1, x2)
    ax1.set_xlim(0, max([0.25, np.max(freq_l)+0.01]))

    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")

    ax1.set_yticks([N*Ua, N, N*Ub])
    ax1.set_yticklabels([r'$\mathcal{P}_{B}$', r'$\Omega$', r'$\mathcal{P}_{A}$'])
    ax1.set_ylabel('excitation energy')
    ax1.set_xlabel('integrated time', labelpad=3)

    ftx = np.linspace(x1, x2, 200)
    if fit is None:
        fit = gaussfunc(ftx, pms[0], pms[1], pms[2])
        ax1.plot(fit, ftx, '-', color='black')
        return fit
    else:
        pass
        ax1.plot(fit, ftx, '-', color='black')
# show_excitation_histogram

'''
Calculate the timeseries to display.

Parameters:
- N, the number of absorbing pairs
- du, the values of Delta to calculate for
- tmax the number of timesteps that will be displayed in the timeseries plot
- phi, the probability parameter
- Omega=1.0, the central energy value
- avg_steps=1, the number of absorption events to average over for each timestep.
'''
def calc_timeseries(N, du, tmax, phi, Omega=1.0, avg_steps=1):
    Ub = Omega-du/2
    Ua = Omega+du/2
    extmodlength = 20*avg_steps
    sim = N_regulated_absorbers(N, Ub, Ua, Omega, phi, length=int(15*tmax*avg_steps+1), extmodlength=extmodlength)
    simc = N_regulated_absorbers(N, Ub, Ua, Omega, phi, length=int(tmax*avg_steps+1), modtype='constant')
    if avg_steps > 1:
        sim = finite_avg(sim, avg_steps)
        simc = finite_avg(simc, avg_steps)
    sim_ = sim[:tmax]
    simc_ = simc[:tmax]
    return sim_, simc_, sim, Ub, Ua
# end calc_timeseries

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, help="Number of absorbing pairs, 3 min.")
    parser.add_argument("-ept", "--eventsper", type=int, help="Number of potential absorption events per timestep, max. 20")
    parser.add_argument("-p", "--phi", type=float, help="Free parameter for the probabilities P_A and P_B. Must be within the interval [0,1].")
    args = parser.parse_args()

    if args.number is None:
        N = 10
    else:
        N = int(args.number)
        if N < 3:
            print("Error: Minimum of three absorbers")
            exit()

    if args.eventsper is None:
        per_timestep = 1
    else:
        per_timestep = int(args.eventsper)
        if per_timestep > 200:
            print("Error: Maximum of 20 events per timestep.")
            exit()

    if args.phi is None:
        phi = 0.5
    else:
        phi = float(args.phi)
        if phi > 1.0 or phi < 0.0:
            print("Error: phi must be within the interval [0,1].")
            exit()

    du = 1.0
    tmax = 1000
    sim, simc, full, Ub, Ua = calc_timeseries(N, du, tmax, phi, avg_steps=per_timestep)

    xinches = 10.5
    yinches = 3.1
    fig = plt.figure('discrete_toy_model', figsize=(xinches, yinches), facecolor='w')

    height = 2.0
    width = 7.0
    xmargin = 0.7
    ymargin = 0.1

    ystart = 7.5*ymargin
    ax = plt.axes([xmargin/xinches, ystart/yinches, width/xinches, height/yinches])
    ax_duslider = plt.axes([xmargin/xinches, ymargin/yinches, width/xinches, 0.2/yinches])
    axhist = plt.axes([(xmargin+width+0.05*height)/xinches, ystart/yinches, height/xinches, height/yinches])
    axreset = plt.axes([(xmargin+width+0.05*height)/xinches, ymargin/yinches, 0.4*height/xinches, 0.2/yinches])

    ax.set_ylim(0, 2*N)
    ax.set_xlim(0, tmax)
    ixs = tmax//3
    t = np.arange(0,tmax,1)

    brk=10
    ln_fluc, = ax.plot(t[ixs+brk:], sim[ixs+brk:], c='b')
    ln_const, = ax.plot(t[:ixs-brk], simc[:ixs-brk], c='b')

    ax.axhline(N, color='k', ls='--')

    ax.text(1/6, 0.9, 'no external fluctuations', ha='center', transform=ax.transAxes)
    ax.text(2/3, 0.9, 'random external fluctuations', ha='center', transform=ax.transAxes)
    ax.set_xticks([100, 300, 500, 700, 900])
    ax.set_yticks([N*Ub, N, N*Ua])
    ax.set_yticklabels([r'$\mathcal{P}_{B}$', r'$\Omega$', r'$\mathcal{P}_{A}$'])
    ax.set_xlabel('timesteps', labelpad=3)
    ax.set_ylabel('excitation energy', labelpad=5)

    initial = excitation_histogram(axhist, full, Ua, Ub, N)

    def update_plot(du):
        sim, simc, full, Ub, Ua  = calc_timeseries(N, du, tmax, phi, avg_steps=per_timestep)
        ln_const.set_ydata(simc[:ixs-brk])
        ln_fluc.set_ydata(sim[ixs+brk:])
        ax.set_yticks([N*Ub, N, N*Ua])

        axhist.cla()
        excitation_histogram(axhist, full, Ua, Ub, N, fit=initial)

        fig.canvas.draw_idle()

    duslider = Slider(ax_duslider, r'$\Delta$', 0.1, 1.9, valinit=du, dragging=False)
    duslider.on_changed(update_plot)

    button = Button(axreset, 'Reset')
    button.on_clicked(lambda e : duslider.reset())

    ttl = str(N) + " absorbers" + "        "
    ttl = ttl + str(per_timestep) + " events/timestep" + "        "
    ttl = ttl + r"$\phi =$ " + str(phi)
    plt.suptitle(ttl, fontsize=14)

    fig.show()
    plt.show()
