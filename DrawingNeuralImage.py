#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:29:18 2024

@author: seba
"""
import nengo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars

# Load the data from the CSV file
data_df = pd.read_csv('/Users/seba/Documents/GitHub/patriotic-neurons/path_coordinates-5.csv')

# Extract y and x coordinates (note: coordinates are in (row, column) format)
y_coords = data_df['y'].values
x_coords = data_df['x'].values

# Invert y-axis to match the standard coordinate system
image_size = max(np.max(x_coords), np.max(y_coords)) + 1  # Assuming square image
y_coords = image_size - y_coords

# Stack x and y into data array
data = np.column_stack((x_coords, y_coords))

# Normalize the data to range [-1, 1]
data_normalized = (data / image_size) * 2 - 1

# Determine the number of points
num_points = data_normalized.shape[0]

# Simulation step
dt = 1e-3

# Total simulation time (adjusted based on number of points)
sim_time = num_points * dt  

# Function to output data points over time
def input_function(t):
    index = int(t / sim_time * num_points)
    if index >= num_points:
        index = num_points - 1
    return data_normalized[index]

# Build the Nengo model
model = nengo.Network(label='Trace Path with Raster Plot')
with model:
    # Input node providing the points over time
    stim = nengo.Node(input_function)

    # Adjust radius according to data range
    radius = np.max(np.abs(data_normalized))  # Should be 1 after normalization

    # Ensemble of neurons
    neurons = nengo.Ensemble(
        n_neurons=5000,       # Total number of neurons
        dimensions=2,
        radius=radius,
        neuron_type=nengo.LIF(),  # Using LIF neurons
        intercepts=nengo.dists.Uniform(-0.5, 0.5),  # Adjusted for better performance
        max_rates=nengo.dists.Uniform(200, 499),     # Adjusted firing rates
        seed=22
    )

    # Connect the input to the ensemble
    nengo.Connection(stim, neurons, synapse=None)

    # Probe the output of the ensemble
    neuron_probe = nengo.Probe(neurons, synapse=0.002)
    # Probe the spikes of the neurons
    spikes_probe = nengo.Probe(neurons.neurons)

# Run the simulation with specified dt and add a progress bar
with nengo.Simulator(model, dt=dt) as sim:
    total_steps = int(sim_time / dt)
    steps_per_bar = 10000  # Number of steps per progress update
    num_bars = total_steps // steps_per_bar
    remaining_steps = total_steps % steps_per_bar

    with tqdm(total=total_steps, desc='Running Simulation', unit='step') as pbar:
        for _ in range(num_bars):
            sim.run_steps(steps_per_bar)
            pbar.update(steps_per_bar)
        if remaining_steps > 0:
            sim.run_steps(remaining_steps)
            pbar.update(remaining_steps)

# Extract the simulation data
times = sim.trange()
outputs = sim.data[neuron_probe]
spikes = sim.data[spikes_probe]

# Prepare the colormap
cmap = cm.get_cmap('viridis', num_points)

# Limit the raster plot to the first 100 neurons
num_neurons_to_plot = 100  # Number of neurons to display in the raster plot
spikes_limited = spikes[:, :num_neurons_to_plot]

# Create the animation with a progress bar
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), dpi=80)  # Reduced figure size and resolution

# Initialize tqdm for animation frames
frame_progress = tqdm(total=len(times), desc='Creating Animation', unit='frame')

# Function to update each frame of the animation
def update(frame):
    # Clear axes
    ax1.clear()
    ax2.clear()

    # Plot the image tracing
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.set_title('Image Tracing')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Plot all previous points with color changing over time
    sc = ax1.scatter(outputs[:frame+1, 0], outputs[:frame+1, 1],
                     c=np.linspace(0, 1, frame+1), cmap='viridis', s=5)

    # Plot the raster plot for the first 100 neurons
    ax2.set_xlim(0, sim_time)
    ax2.set_ylim(0, num_neurons_to_plot)
    ax2.set_ylabel('Neuron Index')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Neuron Spiking Activity (First 100 Neurons)')
    ax2.set_yticks(range(0, num_neurons_to_plot, 10))
    # Plot spikes up to current time
    current_time = times[frame]
    spike_times, neuron_indices = np.nonzero(spikes_limited[:frame+1])
    spike_times = spike_times * sim.dt
    neuron_indices = neuron_indices  # Neuron indices from 0 to num_neurons_to_plot - 1
    ax2.scatter(spike_times, neuron_indices, s=1, color='black')
    ax2.axvline(x=current_time, color='red', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    
    # Update the progress bar
    frame_progress.update(1)
    
    return sc,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(times), blit=False, interval=20)

# Save the animation as a GIF
ani.save('tracing_with_raster_3.gif', writer='pillow', fps=30)

# Close the progress bar for animation
frame_progress.close()

plt.show()
