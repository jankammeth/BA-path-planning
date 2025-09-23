import numpy as np
import matplotlib.pyplot as plt
import time
from scp_wow_2d_fixed import SCPCraft2DTrajectory
from test import SCPOSQPTrajectory
from scp import SCP
from position_generator import generate_positions, print_distance_analysis

def main():
    """
    Improved main function with better performance.
    """
    print("------ WOW Fleet Collision-Free 2D Trajectory Generation ------")
    
    # Configuration parameters - more conservative parameters
    n_vehicles = 2
    time_horizon = 20.0         # [s]
    time_step = 0.1             # [s]
    min_distance = 1.0         # [m]
    space_dims = [0,0,20, 20]   # [m]
    
    print(f"Configuration:")
    print(f"  Number of vehicles: {n_vehicles}")
    print(f"  Time horizon: {time_horizon} s")
    print(f"  Time step: {time_step} s")
    print(f"  Minimum margin: {min_distance} m")
    print(f"  Space dimensions: {space_dims} m")
    print()
    
    # Create the trajectory planner
    planner = SCP(
        n_vehicles=n_vehicles,
        time_horizon=time_horizon,
        time_step=time_step,
        min_distance=min_distance,
        space_dims=space_dims
    )
    
    # Initial positions in a more spread out pattern
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
    
    initial_positions, final_positions = generate_positions(
            n_vehicles, min_distance
        )
    print(f"Successfully generated positions for {n_vehicles} vehicles")

    distance_analysis = print_distance_analysis(initial_positions, final_positions)
    # Set initial & final states
    planner.set_initial_states(initial_positions)
    planner.set_final_states(final_positions)
    
    # Generate the trajectories
    print("Generating trajectories...")
    start_time = time.time()
    try:
        trajectories = planner.generate_trajectories(max_iterations=15)
        end_time = time.time()
        
        # Display results
        print("\nTrajectory generation complete!")
        print(f"Total computation time: {end_time - start_time:.3f} seconds")
        print(f"Number of time steps: {planner.K}")
        print(f"Total trajectory duration: {planner.T} seconds")
        
        # Visualize the trajectories
        print("\nVisualizing 2D trajectories...")
        planner.visualize_trajectories(show_animation=True, save_path="wow_trajectories_2d.png")
        
        # Visualize time snapshots
        print("\nVisualizing time snapshots")
        planner.visualize_time_snapshots(num_snapshots=5, save_path="wow_trajectories_snapshots.png")
        
        print("\nTrajectory files saved as:")
        print("- wow_trajectories_2d.png")
        print("- wow_trajectories_snapshots.png")
    
    except Exception as e:
        print(f"Error during trajectory generation: {e}")

if __name__ == "__main__":
    main()