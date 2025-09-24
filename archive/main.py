import numpy as np
import matplotlib.pyplot as plt
import time

from scp import SCP

def main():
    print("------ WOW Fleet Collision-Free 2D Trajectory Generation ------")

    n_vehicles = 5
    time_horizon = 4.0          # [s] - longer time horizon for easier paths
    time_step = 0.1             # [s]
    min_distance = 0.1          # [m]
    space_dims = [10, 10]       # [m]
    
    print(f"Configuration:")
    print(f"  Number of vehicles: {n_vehicles}")
    print(f"  Time horizon: {time_horizon} s")
    print(f"  Time step: {time_step} s")
    print(f"  Minimum distance: {min_distance} m")
    print(f"  Space dimensions: {space_dims} m")
    print()

    planner = SCP(
        n_vehicles=n_vehicles,
        time_horizon=time_horizon,
        time_step=time_step,
        min_distance=min_distance,
        safety_margin=safety_margin,
        space_dims=space_dims
    )

    initial_positions = np.array([
        [2.0, 2.0],    # bottom left
        [8.0, 2.0],    # bottom right
        [5.0, 5.0],    # center
        [2.0, 8.0],    # top left
        [8.0, 8.0],    # top right
    ])

    # Final positions - swapped configuration
    final_positions = np.array([
        [8.0, 8.0],    # bottom left -> top right
        [2.0, 8.0],    # bottom right -> top left
        [5.0, 2.0],    # center -> bottom center
        [8.0, 2.0],    # top left -> bottom right
        [2.0, 2.0],    # top right -> bottom left
    ])

    # Set initial & final states
    planner.set_initial_states(initial_positions)
    planner.set_final_states(final_positions)