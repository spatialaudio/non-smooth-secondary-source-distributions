""" Generates Figure 2a of the the paper
    Sascha Spors, Frank Schultz, and Hagen Wierstorf. Non-smooth secondary
    source distributions in wave 
field synthesis. In German Annual Conference
    on Acoustics (DAGA), March 2015.

    Sound field synthesized by a semi-infintely rectangular array driven by
    two-dimensional WFS for a virtual line source.
"""

import numpy as np
import matplotlib.pyplot as plt
import sfs


# simulation parameters
xref = [0, 0, 0]  # reference point
normalization = 0.0577  # normalization used for plotting
dx = 0.10  # secondary source distance
N = 1000  # number of secondary sources for one array
Nr = 10  # number of secondary sources on rounded edge
f = 500  # frequency
omega = 2 * np.pi * f  # angular frequency
src_angle = 135
grid = sfs.util.xyz_grid([-2.02, 2], [-2, 2.02], 0, spacing=0.02)


def compute_sound_field(x0, n0, a0, omega, angle):
    npw = sfs.util.direction_vector(np.radians(angle), np.radians(90))
    xs = xref + (np.sqrt(xref[0]**2 + xref[1]**2) + 4) * np.asarray(npw)

    d = sfs.mono.drivingfunction.wfs_2d_line(omega, x0, n0, xs)
    a = sfs.mono.drivingfunction.source_selection_point(n0, x0, xs)

    twin = sfs.tapering.none(a)

    p = sfs.mono.synthesized.generic(omega, x0, n0, d * twin * a0, grid,
                                     source=sfs.mono.source.line)

    return p, twin, xs


def plot_objects(ax):
    sfs.plot.loudspeaker_2d(x0, n0, np.ones(len(x0)), grid=grid)
    sfs.plot.virtualsource_2d(xs, type='point', ax=ax)
    sfs.plot.reference_2d(xref, ax=ax)


def plot_sound_field(p, xs, twin, diff=0):

    fig = plt.figure()
    ax1 = fig.add_axes([0.0, 0.0, 0.7, 1])
    im = sfs.plot.soundfield(p, grid, xnorm=None, colorbar=False, vmax=1.5,
                             vmin=-1.5)
    plot_objects(plt.gca())
    plt.axis([-3.0, 2.2, -2.2, 3.2])
    plt.axis('off')

    myfig = plt.gcf()    
    plt.show()


def plot_sound_field_level(p, xs, twin):

    fig = plt.figure()
    ax1 = fig.add_axes([0.0, 0.0, 0.7, 1])

    im = sfs.plot.level(p, grid, xnorm=None, colorbar=False, vmax=3, vmin=-3)
    plot_objects(plt.gca())
    plt.annotate('4m', (-2.5, 2), (-2.75, -2.4), 
                 arrowprops={'arrowstyle': '<->'}) 
    plt.axis([-3.0, 2.2, -2.2, 3.2])
    plt.axis('off')

    ax2 = fig.add_axes([0.55, -0.05, 0.25, 1])
    plt.axis('off')
    cbar = plt.colorbar(im, ax=ax2, shrink=.7)
    cbar.set_label('relative level (dB)', rotation=270, labelpad=10)

    myfig = plt.gcf()
    plt.show()

#  get secondary source positions
x0, n0, a0 = sfs.array.rounded_edge(N, Nr, dx, n0=[0, -1, 0],
                                    center=[-2, 2, 0])

# compute field at the given positions for given virutal source
p, twin, xs = compute_sound_field(x0, n0, a0, omega, src_angle)

# plot synthesized sound field for multiple virtual source position
plot_sound_field(p/normalization, xs, twin)
plot_sound_field_level(p/normalization, xs, twin)
