""" Generates Figures 4 and 5 of the the paper
    Sascha Spors, Frank Schultz, and Hagen Wierstorf. Non-smooth secondary
    source distributions in wave 
field synthesis. In German Annual Conference
    on Acoustics (DAGA), March 2015.

    Sound  field/level synthesized by a semi-infintely rectangular array
    driven by 2.5-dimensional WFS for a virtual point source.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import sfs

# simulation switches
# Figure 4: compute_field = True
# Figure 5: compute_field = False
compute_field = True

# simulation parameters
xref = [0, 0, 0]  # reference point
falias = 1200
normalization = 0.211  # normalization used for plotting

if compute_field is True:
    f = np.asarray([500])  # frequency
    src_angles = np.asarray([135])
    grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.02)
else:
    f = np.logspace(np.log10(100), np.log10(20000), num=1000)
    src_angles = np.linspace(180, 90, num=180)  # virtual source angles
    grid = sfs.util.xyz_grid(xref[0], xref[1], 0, spacing=1)  # evaluated positions

omega = 2 * np.pi * f  # angular frequency


def compute_sound_field(x0, n0, a0, omega, angle):
    npw = sfs.util.direction_vector(np.radians(angle), np.radians(90))
    xs = xref + (np.sqrt(xref[0]**2 + xref[1]**2) + 4) * np.asarray(npw)

    d = sfs.mono.drivingfunction.wfs_25d_point(omega, x0, n0, xs, xref=xref,
                                               omalias=2*np.pi*falias)
    a = sfs.mono.drivingfunction.source_selection_point(n0, x0, xs)
    twin = sfs.tapering.none(a)

    p = sfs.mono.synthesized.generic(omega, x0, n0, d * twin * a0, grid,
                                 source=sfs.mono.source.point)

    return p, twin, xs


def plot_sound_field_level(p, xs, twin):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    im = sfs.plot.level(p, grid, xnorm=None, colorbar=False, vmax=18, vmin=-9)

    ax1.add_artist(matplotlib.patches.Arc([0, 0], 8, 8, 0, 120, 180,
                                          linestyle='dashed'))
    sfs.plot.virtualsource_2d(xs, type='point')
    sfs.plot.loudspeaker_2d(x0, n0, twin, size=0.14)
    sfs.plot.reference_2d(xref)
    plt.annotate('4m', (-2, 2.3), (2.1, 2.25),
                 arrowprops={'arrowstyle': '<->'}) 

    plt.axis([-4.2, 2.2, -2.2, 3])
    plt.axis('off')

    ax2 = fig.add_axes([0.75, -0.02, 0.2, 1])
    plt.axis('off')
    cbar = plt.colorbar(im, ax=ax2, shrink=.6)
    cbar.set_label('relative level (dB)', rotation=270)

    plt.tight_layout(pad=0.2)
    myfig = plt.gcf()    
    plt.show()


def plot_sound_field(p, xs, twin):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    im = sfs.plot.soundfield(p, grid, xnorm=None, colorbar=False, vmax=4, 
                             vmin=-4)

    ax1.add_artist(matplotlib.patches.Arc([0, 0], 8, 8, 0, 120, 180,
                                          linestyle='dashed'))
    sfs.plot.virtualsource_2d(xs, type='point')
    sfs.plot.loudspeaker_2d(x0, n0, twin, size=0.14)
    sfs.plot.reference_2d(xref)
    plt.annotate('4m', (-2, 2.3), (2.1, 2.25),
                 arrowprops={'arrowstyle': '<->'})

    plt.axis([-4.2, 2.5, -2.2, 3])
    plt.axis('off')

    plt.tight_layout(pad=0)
    myfig = plt.gcf()    
    plt.show()


# get secondary source positions
x0, n0, a0 = sfs.array.load('/Users/spors/Documents/src/SFS-python/data/arrays/university_rostock.csv')

# correct weights for non-equidistant sampling
a0 = sfs.array.weights_closed(x0)

# compute field at the given positions for given virutal source positions
p = []
trajectory = []
lsactive = []

for omegan in omega:

    ptmp = []
    xstmp = []
    twintmp = []
    for angle in src_angles:
        tmp, twin, xs = compute_sound_field(x0, n0, a0, omegan, angle)
        ptmp.append(tmp)
        xstmp.append(xs)
        twintmp.append(twin)

    p.append(ptmp)
    trajectory.append(xstmp)
    lsactive.append(twintmp)

p = np.asarray(p)
trajectory = np.asarray(trajectory)
lsactive = np.asarray(lsactive)


# plot synthesized sound field for one virtual source position
if compute_field is True:
    p = np.squeeze(p)
    lsactive = np.squeeze(lsactive)
    trajectory = np.squeeze(trajectory)

    plot_sound_field(p/normalization, trajectory, lsactive)
    plot_sound_field_level(p/normalization, trajectory, lsactive)

# plot frequency response at one position in sound field for various positions
if compute_field is False:

    fig = plt.figure()

    plt.imshow(20*np.log10(np.abs(p/normalization)), cmap='coolwarm_clip', 
               origin='lower', extent=[src_angles[0], src_angles[-1], f[0],
                                       f[-1]], vmax=9, vmin=-3, aspect='auto')
    plt.yscale('log')
    plt.xlabel('angle (deg)')
    plt.ylabel('frequency (Hz)')
    
    cbar = plt.colorbar()
    cbar.set_label('relative level (dB)', rotation=270)

    plt.tight_layout(pad=0.2)
    myfig = plt.gcf()    
    plt.show()
