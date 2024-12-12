import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants and parameters
N_l = 10  # Number of lines (parallel to the axis)
N_c = 50  # Number of circles (perpendicular to the axis)
tau = 1.0e-11  # Lifetime of triplets in femtoseconds (1 ps = 1.0e-12 seconds)
epsilon = 0.1  # Interaction strength
k_B = 1.38e-23  # Boltzmann constant (J/K)
T = 300  # Simulation temperature in Kelvin (adjustable)
epsilon_s = 0.1  # Strength of the external potential
a_s = 0.5  # Lattice spacing for the external potential
steps = 500  # Number of time steps in the simulation
grid_spacing_nm = 0.1  # Grid spacing in nanometers (1 picometer)
grid_spacing_m = grid_spacing_nm * 1e-9  # Convert grid spacing to meters

# Friction coefficient
gamma = 0.1  # Friction coefficient, adjust this value to control friction strength

# Critical temperature based on hydrogen bond energy
T_c = epsilon / (k_B * np.log(N_l))

# Function to initialize the grid
def initialize_grid():
    grid = np.zeros((N_l, N_c), dtype=str)  # Initialize as empty strings
    for i in range(N_l):
        grid[i, 0] = 'W'  # Initialize with water molecules at the beginning
        grid[i, -1] = 'W'  # Water molecules at the end
    return grid

# Function to form dimers
def form_dimers(grid):
    for i in range(1, N_c-1):
        for j in range(N_l):
            grid[j, i] = 'D'  # Dimer in all other positions

# Function to calculate the probability of forming a triplet based on temperature
def triplet_probability(T, T_c):
    if T <= T_c:
        return 0.2  # More likely to form dimers
    else:
        return 0.8  # More likely to form triplets

# External periodic potential function (summing over all positions)
def external_periodic_potential_sum(grid, a_s, epsilon_s):
    total_potential = 0
    for i in range(N_l):
        for j in range(1, N_c-1):
            x_n = j  # Use the position index along the nanotube
            potential_energy = (epsilon_s / 2) * (1 - np.cos(2 * np.pi * x_n / a_s))
            total_potential += potential_energy
    return total_potential

# Monte Carlo function with potential influence and friction
def monte_carlo_with_potential_and_friction(grid, T, T_c, epsilon_s, a_s, gamma):
    prob_triplet = triplet_probability(T, T_c)

    for i in range(N_l):
        for j in range(1, N_c-1):
            # Calculate the total external periodic potential over all sites
            potential_energy = external_periodic_potential_sum(grid, a_s, epsilon_s)
            if grid[i, j] == 'D':
                # Adjust decision probability based on total potential and temperature
                if random.random() < prob_triplet * np.exp(-potential_energy / (k_B * T)):
                    grid[i, j] = 'T'
            elif grid[i, j] == 'T':
                # Apply friction to the triplet's motion
                if random.random() < 0.5 * np.exp(-potential_energy / (k_B * T)):
                    grid[i, j] = 'D'
                    if j + 1 < N_c - 1:
                        # Update position with friction applied
                        move_prob = (1 - gamma)  # Friction reduces the chance of moving
                        if random.random() < move_prob:
                            grid[i, j+1] = 'T'  # Move triplet forward
    return grid

# Function to gather statistics on dimers and triplets
def collect_statistics(grid):
    num_dimers = np.sum(grid == 'D')
    num_triplets = np.sum(grid == 'T')
    return num_dimers, num_triplets

# Function to calculate triplet speeds in m/s, with friction applied
def calculate_triplet_speed(triplet_positions_t0, triplet_positions_t1, tau, grid_spacing_m, gamma):
    speeds = []
    for (i0, j0), (i1, j1) in zip(triplet_positions_t0, triplet_positions_t1):
        displacement_nm = np.sqrt((i1 - i0)**2 + (j1 - j0)**2) * grid_spacing_m  # Convert to meters
        speed_m_per_s = displacement_nm / tau
        # Apply friction to the speed
        speed_m_per_s = speed_m_per_s * (1 - gamma)  # Reduce speed due to friction
        speeds.append(speed_m_per_s)
    if speeds:
        return np.mean(speeds)  # Mean speed of triplets
    else:
        return 0.0  # If no triplets

# Function to create a heatmap from the grid
def visualize_grid(grid, step):
    # Replace W, D, and T with numbers for visualization
    visual_grid = np.zeros((N_l, N_c))
    for i in range(N_l):
        for j in range(N_c):
            if grid[i, j] == 'W':
                visual_grid[i, j] = 1  # Water molecule
            elif grid[i, j] == 'D':
                visual_grid[i, j] = 2  # Dimer
            elif grid[i, j] == 'T':
                visual_grid[i, j] = 3  # Triplet
    plt.figure(figsize=(8, 6))
    sns.heatmap(visual_grid, annot=False, cmap="coolwarm", cbar=True, vmin=0, vmax=3)
    plt.title(f"Grid Visualization at Step {step}")
    plt.show()

# Simulation function with friction applied
def simulate_with_visualizations(steps, tau, T, T_c, epsilon_s, a_s, gamma, grid_spacing_m):
    grid = initialize_grid()  # Initialize the grid inside the function
    form_dimers(grid)  # Form dimers initially
    dimer_counts = []
    triplet_counts = []
    ratio_T_D = []
    triplet_speeds = []
    triplet_positions_t0 = [(i, 0) for i in range(N_l)]  # Initial triplet positions (start from column 0)

    for t in range(steps):
        grid = monte_carlo_with_potential_and_friction(grid, T, T_c, epsilon_s, a_s, gamma)
        num_dimers, num_triplets = collect_statistics(grid)
        dimer_counts.append(num_dimers)
        triplet_counts.append(num_triplets)

        # Calculate the ratio T/D, handle division by zero case
        if num_dimers > 0:
            ratio_T_D.append(num_triplets / num_dimers)
        else:
            ratio_T_D.append(np.inf)  # If no dimers, ratio is infinite

        # Track new triplet positions and calculate triplet speed in m/s
        triplet_positions_t1 = [(i, j) for i in range(N_l) for j in range(N_c) if grid[i, j] == 'T']
        speed = calculate_triplet_speed(triplet_positions_t0, triplet_positions_t1, tau, grid_spacing_m, gamma)
        triplet_speeds.append(speed)

        # Update triplet positions for next step
        triplet_positions_t0 = triplet_positions_t1

        print(f"Time step {t}: Dimers = {num_dimers}, Triplets = {num_triplets}, Ratio T/D = {ratio_T_D[-1]}, Triplet Speed (m/s) = {speed}")
        visualize_grid(grid, t)  # Visualize the grid at each step

    return dimer_counts, triplet_counts, ratio_T_D, triplet_speeds

# Run the simulation and collect statistics
dimer_counts, triplet_counts, ratio_T_D, triplet_speeds = simulate_with_visualizations(steps, tau, T, T_c, epsilon_s, a_s, gamma, grid_spacing_m)

# Plot the results, including dimers, triplets, ratio T/D, and triplet speed
plt.figure()
plt.plot(range(steps), dimer_counts, label='Dimers', marker='o', color='green')
plt.plot(range(steps), triplet_counts, label='Triplets', marker='x', color='red')
plt.xlabel('Time Steps')
plt.ylabel('Count')
plt.title('Dimer and Triplet Evolution Over Time')
plt.legend()
plt.show()

# Plot the ratio of Triplets (T) to Dimers (D)
plt.figure()
plt.plot(range(steps), ratio_T_D, label='Ratio T/D', marker='d', color='purple')
plt.xlabel('Time Steps')
plt.ylabel('Ratio T/D')
plt.title('Ratio of Triplets (T) to Dimers (D) Over Time')
plt.legend()
plt.show()

# Plot the mean speed of Triplets (T) in m/s over time, with friction applied
plt.figure()
plt.plot(range(steps), triplet_speeds, label='Mean Triplet Speed (m/s)', marker='x', color='orange')
plt.xlabel('Time Steps')
plt.ylabel('Speed (m/s)')
plt.title('Mean Speed of Triplets (m/s) Over Time with Friction')
plt.legend()
plt.show()