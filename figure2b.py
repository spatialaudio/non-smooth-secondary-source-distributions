""" Generates Figure 2b of the the paper
    Sascha Spors, Frank Schultz, and Hagen Wierstorf. Non-smooth secondary
    source distributions in wave 
field synthesis. In German Annual Conference
    on Acoustics (DAGA), March 2015.

    Level synthesized by a semi-infintely rectangular array driven by
    two-dimensional WFS for a virtual line source.
"""

import numpy as np
import matplotlib.pyplot as plt
import sfs


# simulation parameters
xref = [0, 0, 0]  # reference point
N = 1000
Nr = [0, 1, 10]
dx = 0.10
normalization = 0.0577  # normalization used for plotting

f = 500
omega = 2 * np.pi * f  # angular frequency

src_angles = np.linspace(180, 90, num=180)  # virtual source angles ps
grid = sfs.util.xyz_grid(xref[0], xref[1], 0, spacing=1)  # evaluated position


def compute_sound_field(x0, n0, a0, omega, angle):
    npw = sfs.util.direction_vector(np.radians(angle), np.radians(90))
    xs = xref + (np.sqrt(xref[0]**2 + xref[1]**2) + 4) * np.asarray(npw)

    d = sfs.mono.drivingfunction.wfs_2d_line(omega, x0, n0, xs)
    a = sfs.mono.drivingfunction.source_selection_point(n0, x0, xs)

    twin = sfs.tapering.none(a)

    p = sfs.mono.synthesized.generic(omega, x0, n0, d * twin * a0, grid,
                                     source=sfs.mono.source.line)

    return p, twin, xs


# compute field at the given position for given virutal source positions
p = []
trajectory = []
lsactive = []

for Nr0 in Nr:
    #  get secondary source positions
    x0, n0, a0 = sfs.array.rounded_edge(N, Nr0, dx, n0=[0, -1, 0],
                                        center=[-2, 2, 0])

    ptmp = []
    xstmp = []
    twintmp = []
    for angle in src_angles:
        tmp, twin, xs = compute_sound_field(x0, n0, a0, omega, angle)
        ptmp.append(tmp)
        xstmp.append(xs)
        twintmp.append(twin)

    p.append(ptmp)
    trajectory.append(xstmp)
    lsactive.append(twintmp)

p = np.asarray(p)
trajectory = np.asarray(trajectory)
lsactive = np.asarray(lsactive)


fig = plt.figure()
ax = plt.gca()

im = plt.plot(src_angles, 20*np.log10(np.abs(p.T/normalization)))
plt.axis([90, 180, -1, 3])
ax.invert_xaxis()
plt.xlabel('angle (deg)')
plt.ylabel('relative level (dB)')
plt.grid()
ax.legend(['rect', '$N_r = 1$', '$N_r = 10$'], loc='lower center', ncol=3)

myfig = plt.gcf()
plt.show()
