import time

import numpy as np

from ..scenarios.position_generator import generate_positions, print_distance_analysis
from ..solvers.scp import SCP


def main():
    """
    Create collision-free trajectories for a set of initial and final positions.
    """
    print("------ WOW Fleet Collision-Free 2D Trajectory Generation ------")

    # Configuration parameters
    n_vehicles = 5
    time_horizon = 10  # [s]
    time_step = 0.2  # [s]
    min_distance = 0.8  # [m]
    space_dims = [0, 0, 20, 20]  # [m]

    print("Configuration:")
    print(f"  Number of vehicles: {n_vehicles}")
    print(f"  Time horizon: {time_horizon} s")
    print(f"  Time step: {time_step} s")
    print(f"  Minimum margin: {min_distance} m")
    print(f"  Space dimensions: {space_dims} m")
    print()

    # Create the solver instance
    solver = SCP(
        n_vehicles=n_vehicles,
        time_horizon=time_horizon,
        time_step=time_step,
        min_distance=min_distance,
        space_dims=space_dims,
    )

    # Set initial positions
    initial_positions = np.array(
        [
            [2.0, 2.0],
            [8.0, 2.0],
            [5.0, 5.0],
            [2.0, 8.0],
            [8.0, 8.0],
        ]
    )

    # Set final positions (swapped configuration)
    final_positions = np.array(
        [
            [8.0, 8.0],
            [2.0, 8.0],
            [5.0, 2.0],
            [8.0, 2.0],
            [2.0, 2.0],
        ]
    )

    # OR: do it via the position_generator
    initial_positions, final_positions = generate_positions(n_vehicles, min_distance)
    print(f"Successfully generated positions for {n_vehicles} vehicles")

    print_distance_analysis(initial_positions, final_positions)
    # Set initial & final states
    solver.set_initial_states(initial_positions)
    solver.set_final_states(final_positions)

    # Generate the trajectories
    print("Generating trajectories...")
    start_time = time.time()
    try:
        # Generate collision_free trajectories
        solver.generate_trajectories(max_iterations=15)
        end_time = time.time()

        # Print performance metrics
        print("\nTrajectory generation complete!")
        print(f"Total computation time: {end_time - start_time:.3f} seconds")
        print(f"Number of time steps: {solver.K}")
        print(f"Total trajectory duration: {solver.T} seconds")

        # Visualize the trajectories
        print("\nVisualizing 2D trajectories...")
        solver.visualize_trajectories(show_animation=True, save_path="wow_trajectories_2d.png")

        # Visualize time snapshots
        print("\nVisualizing time snapshots")
        solver.visualize_time_snapshots(num_snapshots=5, save_path="wow_trajectories_snapshots.png")

        print("\nTrajectory files saved as:")
        print("- wow_trajectories_2d.png")
        print("- wow_trajectories_snapshots.png")

    except Exception as e:
        print(f"Error during trajectory generation: {e}")


if __name__ == "__main__":
    main()
