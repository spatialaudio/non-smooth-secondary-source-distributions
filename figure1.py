""" Generates Figure 1 of the the paper
    Sascha Spors, Frank Schultz, and Hagen Wierstorf. Non-smooth secondary
    source distributions in wave 
field synthesis. In German Annual Conference
    on Acoustics (DAGA), March 2015.

    Sound field synthesized by an infitely long vs. semi-infintely long
    linear/rectangular array driven by two-dimensional WFS for a virtual
    line source.
"""

import numpy as np
import matplotlib.pyplot as plt
import sfs


# what Figure to generate
# Figure 1(a): infinite=True, rect=False
# Figure 1(b): infinite=False, rect=False
# Figure 1(c): infinite=False, rect=True
infinite = True  # infinite linear array
rect = False  # rectangular array

# simulation parameters
xref = [0, 0, 0]  # reference point
dx = 0.05  # secondary source distance
N = 2000  # number of secondary sources for one array
f = 500  # frequency
omega = 2 * np.pi * f  # angular frequency
src_angle = 135

if not rect:
    grid = sfs.util.xyz_grid([-2, 2], [-2, 3], 0, spacing=0.02)
else:
    grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.02)


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
    if infinite and not rect:
        ax.plot((-2, -2), (-2.2, 3.2), 'k-', lw=4)
    if not infinite:
        ax.plot((-2, -2), (-2.2, 2), 'k-', lw=4)
    if rect:
        ax.plot((-2, 2.2), (2, 2), 'k-', lw=4)

    sfs.plot.virtualsource_2d(xs, type='point', ax=ax)
    sfs.plot.reference_2d(xref, ax=ax)


def plot_sound_field(p, xs, twin, diff=0):

    fig = plt.figure()
    ax1 = fig.add_axes([0.0, 0.0, 0.7, 1])
    im = sfs.plot.soundfield(p, grid, xnorm=None, colorbar=False,
                             vmax=1.5, vmin=-1.5)
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


# define secondary source positions

#  infinitely long linear array
x0, n0, a0 = sfs.array.linear(2*N, dx, center=[-2, -dx/2, 0])
# semi-infinte linear array
if not infinite:
    idx = x0[:, 1] <= 2+dx/2
    x0 = x0[idx, :]
    n0 = n0[idx, :]
    a0 = a0[idx]
    a0[-1] = 1/2 * a0[-1]
# semi-infinite edge + infinte edge
if rect:
    x00, n00 = sfs.array._rotate_array(x0, n0, [1, 0, 0], [0, -1, 0])
    x00[:, 0] = - x00[:, 0]
    x00 = np.flipud(x00)
    a00 = np.flipud(a0)
    x0 = np.concatenate((x0, x00))
    n0 = np.concatenate((n0, n00))
    a0 = np.concatenate((a0, a00))
    # infinte edge as reference
    x0ref, n0ref, a0ref = sfs.array.linear(2*N, dx, center=[-2, -dx/2, 0])
    x00, n00 = sfs.array._rotate_array(x0ref, n0ref, [1, 0, 0], [0, -1, 0])
    x0ref = np.concatenate((x0ref, x00))
    n0ref = np.concatenate((n0ref, n00))
    a0ref = np.concatenate((a0ref, a0ref))


# compute field
p, twin, xs = compute_sound_field(x0, n0, a0, omega, src_angle)


# plot synthesized sound field and its level
if not rect:
    normalization = 0.066  # normalization used for plotting
else:
    normalization = 0.059  # normalization used for plotting (rect)

plot_sound_field(p/normalization, xs, twin)
plot_sound_field_level(p/normalization, xs, twin)
