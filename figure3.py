""" Generates Figure 3 of the the paper
    Sascha Spors, Frank Schultz, and Hagen Wierstorf. Non-smooth secondary
    source distributions in wave 
field synthesis. In German Annual Conference
    on Acoustics (DAGA), March 2015.

    Level synthesized by a semi-infintely rectangular array driven by
    2/2.5-dimensional WFS for a virtual source.
"""

import numpy as np
import matplotlib.pyplot as plt
import sfs

# simulation parameters
xref = [0, 0, 0]  # reference point
N = 1000
dx = 0.10
normalization2 = 0.0591  # normalization used for plotting
normalization25 = 2.78  # normalization used for plotting

f = 500
omega = 2 * np.pi * f  # angular frequency

src_angles = np.linspace(180, 90, num=180)  # virtual source angles ps
grid = sfs.util.xyz_grid(xref[0], xref[1], 0, spacing=1)  # evaluated position


def compute_sound_field(x0, n0, a0, omega, angle, twod=True):
    npw = sfs.util.direction_vector(np.radians(angle), np.radians(90))
    xs = xref + (np.sqrt(xref[0]**2 + xref[1]**2) + 4) * np.asarray(npw)

    if twod is True:
        d = sfs.mono.drivingfunction.wfs_2d_line(omega, x0, n0, xs)
        a = sfs.mono.drivingfunction.source_selection_point(n0, x0, xs)
        twin = sfs.tapering.none(a)

        p = sfs.mono.synthesized.generic(omega, x0, n0, d * twin * a0, grid, 
                                         source=sfs.mono.source.line)
    else:
        d = sfs.mono.drivingfunction.wfs_25d_point(omega, x0, n0, xs, xref=xref)
        a = sfs.mono.drivingfunction.source_selection_point(n0, x0, xs)
        twin = sfs.tapering.none(a)

        p = sfs.mono.synthesized.generic(omega, x0, n0, d * twin * a0, grid, 
                                         source=sfs.mono.source.point)

    return p, twin, xs


#  infinitely long linear array
x0, n0, a0 = sfs.array.linear(2*N, dx, center=[-2, -dx/2, 0])
# semi-infinte linear array
idx = np.squeeze(np.where(x0[:, 1] <= 2+dx/2))
x0 = x0[idx, :]
n0 = n0[idx, :]
a0 = a0[idx]
a0[-1] = 1/2 * a0[-1]
# semi-infinite edge + infinte edge
x00, n00 = sfs.array._rotate_array(x0, n0, [1, 0, 0], [0, -1, 0])
x00[:, 0] = - x00[:, 0]
x00 = np.flipud(x00)
a00 = np.flipud(a0)
x0 = np.concatenate((x0, x00))
n0 = np.concatenate((n0, n00))
a0 = np.concatenate((a0, a00))


# compute field at the given positions for given virutal source positions
p2 = []
trajectory = []
lsactive = []

for angle in src_angles:
    tmp, twin, xs = compute_sound_field(x0, n0, a0, omega, angle, twod=True)
    p2.append(tmp)
    trajectory.append(xs)
    lsactive.append(twin)

p25 = []
for angle in src_angles:
    tmp, twin, xs = compute_sound_field(x0, n0, a0, omega, angle, twod=False)
    p25.append(tmp)


p2 = np.asarray(p2)
p25 = np.asarray(p25)
trajectory = np.asarray(trajectory)
lsactive = np.asarray(lsactive)


# plot results
fig = plt.figure()
ax = plt.gca()

im = plt.plot(src_angles, 20*np.log10(np.abs(p2/normalization2)), src_angles,
              20*np.log10(np.abs(p25/normalization25)))
plt.axis([90, 180, -1, 3])
ax.invert_xaxis()
plt.xlabel('angle (deg)')
plt.ylabel('relative level (dB)')
plt.grid()
ax.legend(['2D', '2.5D'], loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2)

myfig = plt.gcf()
plt.show()
