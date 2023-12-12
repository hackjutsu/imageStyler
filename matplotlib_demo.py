# matplotlib_demo.py

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot data
ax.plot(t, s)

# Set labels and title
ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='Simple Plot')

# Enable grid
ax.grid()

# Show the plot
plt.show()
