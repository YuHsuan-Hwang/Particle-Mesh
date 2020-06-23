# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.colors import cnames
"""
Plot 3D

@author: KEY
"""
# read file
path = r'C:\01 Course\Computational Astrophysics\00ParticleMesh\ParticleMesh_FFT\ParticleMesh_FFT\Data.csv'

with open(path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    particles = list(reader)
p = np.array(particles, dtype=float)

t = len(p)
L = 0.1*np.max(np.abs(p))
n = int(p.shape[1]/3)

particles = np.empty((n, 3, t))
for i in range(n):
    for j in range(3):
        for k in range(t):
            particles[i][j][k] = p[k][j+i*3]


# %% create figure
L = 1.0*np.max(np.abs(p))

# Set up figure & 3D axis for animation
fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
# ax.axis('off')

# prepare the axes limits
ax.set_xlim((-0.1*L, L))
ax.set_ylim((-0.1*L, L))
ax.set_zlim((-0.1*L, L))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
# ax.view_init(30, 0)

# %% animation 3D with trajectory
x_t = np.empty((n, t, 3))
# data arange
for i in range(n):
    for j in range(3):
        for k in range(t):       
            x_t[i][k][j] = particles[i][j][k]

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, n))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    # ax.view_init(30, 0.3 * i)
    # fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init
                               , interval=1, blit=False)

plt.show()



# %% animation 3D without trajectory
L = 1.0*np.max(np.abs(p))
x_t = np.empty((n, t, 3))
# data arange
for i in range(n):
    for j in range(3):
        for k in range(t):       
            x_t[i][k][j] = particles[i][j][k]

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, n))

# set up lines and points
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

# initialization function: plot the background of each frame
def init():
    for pt in pts:
        pt.set_data([], [])
        pt.set_3d_properties([])
    return pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for pt, xi in zip(pts, x_t):
        x, y, z = xi[:i].T
        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    # ax.view_init(30, 0.3 * i)
    # fig.canvas.draw()
    return pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init
                               , interval=30, blit=False)

plt.show()