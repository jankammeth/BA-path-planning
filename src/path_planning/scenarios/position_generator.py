import random

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


def generate_positions(n_vehicles, min_distance=0.4, max_attempts=1000):
    """
    Generate initial and final positions for vehicles according to the specified layout:
    - 20m x 20m position box
    - 4 circles (diameter 5m) at corners with 1m margin from borders
    - 1 diamond (6m side length when unrotated) at center with 1m margin from circles
    - Initial positions: randomly on the 4 circles
    - Final positions: randomly on circles and diamond border

    Args:
        n_vehicles: number of vehicles
        min_distance: minimum distance between vehicles
        max_attempts: maximum attempts to find valid positions

    Returns:
        initial_positions: (n_vehicles, 2) array
        final_positions: (n_vehicles, 2) array
    """

    # Define layout parameters
    # box_size = 20.0
    circle_diameter = 5.0
    circle_radius = circle_diameter / 2.0
    square_side = 6.0
    # margin = 1.0

    circle_centers = np.array(
        [
            [3.5, 3.5],  # bottom-left
            [16.5, 3.5],  # bottom-right
            [3.5, 16.5],  # top-left
            [16.5, 16.5],  # top-right
        ]
    )

    diamond_center = np.array([10.0, 10.0])
    diamond_size = square_side / np.sqrt(2)  # Distance from center to vertex

    diamond_vertices = np.array(
        [
            [diamond_center[0], diamond_center[1] + diamond_size],  # top
            [diamond_center[0] + diamond_size, diamond_center[1]],  # right
            [diamond_center[0], diamond_center[1] - diamond_size],  # bottom
            [diamond_center[0] - diamond_size, diamond_center[1]],  # left
        ]
    )

    def sample_point_on_circle(center, radius):
        """Sample a random point on the circumference of a circle."""
        angle = random.uniform(0, 2 * np.pi)
        return center + radius * np.array([np.cos(angle), np.sin(angle)])

    def sample_point_on_diamond_border(vertices):
        """Sample a random point on the border of the diamond."""
        edge = random.randint(0, 3)  # Chooses one edge randomly to sample point

        v1 = vertices[edge]
        v2 = vertices[(edge + 1) % 4]

        t = random.uniform(0, 1)
        point = v1 + t * (v2 - v1)

        return point

    def is_valid_position(new_pos, existing_positions, min_dist):
        """Check if new position maintains minimum distance from existing ones."""
        for pos in existing_positions:
            if np.linalg.norm(new_pos - pos) < min_dist:
                return False
        return True

    # Generate initial positions (only on circles)
    initial_positions = []
    attempts = 0

    while len(initial_positions) < n_vehicles and attempts < max_attempts:
        # Randomly select a circle
        circle_idx = random.randint(0, 3)
        center = circle_centers[circle_idx]

        # Sample point on circle circumference
        new_pos = sample_point_on_circle(center, circle_radius)

        # Check if position is valid
        if is_valid_position(new_pos, initial_positions, min_distance):
            initial_positions.append(new_pos)

        attempts += 1

    if len(initial_positions) < n_vehicles:
        raise ValueError(
            f"Could not generate {n_vehicles} valid initial positions after {max_attempts} attempts"
        )

    # Generate final positions (on circles and square)
    final_positions = []
    attempts = 0

    while len(final_positions) < n_vehicles and attempts < max_attempts:
        # Randomly choose between diamond border (80%) and circles (20%)
        if random.random() < 0.9:  # Diamond border
            new_pos = sample_point_on_diamond_border(diamond_vertices)
        else:  # Circle
            circle_idx = random.randint(0, 3)
            center = circle_centers[circle_idx]
            new_pos = sample_point_on_circle(center, circle_radius)

        # Check if position is valid
        if is_valid_position(new_pos, final_positions, min_distance):
            final_positions.append(new_pos)

        attempts += 1

    if len(final_positions) < n_vehicles:
        raise ValueError(
            f"Could not generate {n_vehicles} valid final positions after {max_attempts} attempts"
        )

    return np.array(initial_positions), np.array(final_positions)


def quadrant_colors(initial_positions, center=(10.0, 10.0)):
    """
    Assign a color to each craft based on the quadrant of its *initial* position.
    Quadrants (relative to `center`):
      Q0: top-right,  Q1: top-left,  Q2: bottom-left,  Q3: bottom-right
    Returns:
      colors: list of color strings (len = n_vehicles)
      quad_ids: list of ints in {0,1,2,3}
    """
    cx, cy = center
    # Use distinct, readable colors
    palette = [
        (0.17, 0.28, 0.46),
        (0.54, 0.31, 0.56),
        (1.00, 0.39, 0.38),
        (1.00, 0.65, 0.00),
    ]  # Q0,Q1,Q2,Q3

    quad_ids = []
    colors = []
    for p in initial_positions:
        x, y = p
        if x >= cx and y >= cy:
            q = 0  # top-right
        elif x < cx and y >= cy:
            q = 1  # top-left
        elif x < cx and y < cy:
            q = 2  # bottom-left
        else:
            q = 3  # bottom-right
        quad_ids.append(q)
        colors.append(palette[q])
    return colors, quad_ids


def visualize_layout(initial_positions, final_positions, min_distance=0.4):
    """
    Visualize the generated positions with colors tied to the initial-position quadrant.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",  # serif font (similar to LaTeX default)
            "font.size": 16,  # general font size
            "axes.titlesize": 20,  # title
            "axes.labelsize": 18,  # x/y labels
            "xtick.labelsize": 14,  # x-ticks
            "ytick.labelsize": 14,  # y-ticks
            "legend.fontsize": 14,  # legend text
        }
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw the 20x20 boundary box
    boundary = Rectangle(
        (0, 0), 20, 20, linewidth=2, edgecolor="black", facecolor="none", linestyle="--", alpha=0.7
    )
    ax.add_patch(boundary)

    # Corner circles
    circle_centers = np.array([[3.5, 3.5], [16.5, 3.5], [3.5, 16.5], [16.5, 16.5]])
    for center in circle_centers:
        circle = Circle(center, 2.5, linewidth=1.5, edgecolor="grey", facecolor="none", alpha=0.7)
        ax.add_patch(circle)

    # Central diamond
    diamond_center = np.array([10.0, 10.0])
    diamond_size = 6 / np.sqrt(2)
    diamond_vertices = np.array(
        [
            [diamond_center[0], diamond_center[1] + diamond_size],
            [diamond_center[0] + diamond_size, diamond_center[1]],
            [diamond_center[0], diamond_center[1] - diamond_size],
            [diamond_center[0] - diamond_size, diamond_center[1]],
        ]
    )
    diamond_x = np.append(diamond_vertices[:, 0], diamond_vertices[0, 0])
    diamond_y = np.append(diamond_vertices[:, 1], diamond_vertices[0, 1])
    ax.plot(diamond_x, diamond_y, linewidth=1.5, color="grey", alpha=0.7)

    # --- NEW: colors based on initial-position quadrants ---
    colors, quad_ids = quadrant_colors(initial_positions, center=(10.0, 10.0))

    # Plot initial positions (circles)
    for i, pos in enumerate(initial_positions):
        ax.scatter(pos[0], pos[1], marker="o", s=150, color=colors[i], label=None, alpha=1.0)
        ax.add_patch(
            Circle(
                pos, min_distance / 2, linewidth=1, edgecolor=colors[i], facecolor="none", alpha=0.3
            )
        )

    # Plot final positions (squares) using the SAME color per craft
    for i, pos in enumerate(final_positions):
        ax.scatter(pos[0], pos[1], marker="s", s=150, color=colors[i], label=None, alpha=1.0)
        ax.add_patch(
            Circle(
                pos, min_distance / 2, linewidth=1, edgecolor=colors[i], facecolor="none", alpha=0.3
            )
        )

    # Connect start to end with the SAME color
    for i in range(len(initial_positions)):
        ax.plot(
            [initial_positions[i, 0], final_positions[i, 0]],
            [initial_positions[i, 1], final_positions[i, 1]],
            color=colors[i],
            alpha=0.3,
            linewidth=1.5,
        )

    # --- simplified legend: only start vs stop ---
    start_handle = mlines.Line2D(
        [], [], color="black", marker="o", linestyle="None", markersize=8, label="Start"
    )
    stop_handle = mlines.Line2D(
        [], [], color="black", marker="s", linestyle="None", markersize=8, label="Stop"
    )

    ax.legend(
        handles=[start_handle, stop_handle],
        loc="lower right",
        bbox_to_anchor=(0.95, 0.05),  # nudge slightly inward
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.6,
        borderpad=1.2,  # padding inside the box
        labelspacing=1.0,  # vertical spacing between labels
        handlelength=1.8,  # length of marker in legend
        handletextpad=0.8,  # space between marker and text
        fontsize=11,
    )  # slightly bigger text

    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 21)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [m]", fontsize=18)
    ax.set_ylabel(r"$y$ [m]", fontsize=18)
    ax.set_title("Initial and Final Craft Positions")

    plt.tight_layout()
    plt.savefig("position_generator.pdf", dpi=400)
    plt.show()

    return fig, ax


def main_with_generated_positions():
    """
    Updated main function using generated positions.
    """
    print("------ WOW Fleet Collision-Free 2D Trajectory Generation ------")

    # Configuration parameters
    n_vehicles = 5
    time_horizon = 10.0  # [s]
    time_step = 0.1  # [s]
    min_distance = 0.4 + 0.1  # [m]
    space_dims = [0, 0, 20, 20]  # [m] - Updated to 20x20

    print("Configuration:")
    print(f"  Number of vehicles: {n_vehicles}")
    print(f"  Time horizon: {time_horizon} s")
    print(f"  Time step: {time_step} s")
    print(f"  Minimum margin: {min_distance} m")
    print(f"  Space dimensions: {space_dims} m")
    print()

    # Generate positions
    print("Generating initial and final positions...")
    try:
        initial_positions, final_positions = generate_positions(
            n_vehicles, min_distance, max_attempts=2000
        )
        print(f"Successfully generated positions for {n_vehicles} vehicles")

        # Visualize the layout
        print("Visualizing generated positions...")
        visualize_layout(initial_positions, final_positions, min_distance)

    except ValueError as e:
        print(f"Error generating positions: {e}")
        print("Try reducing the number of vehicles or minimum distance")
        return

    print(f"Initial positions:\n{initial_positions}")
    print(f"Final positions:\n{final_positions}")


def calculate_minimum_distances(positions):
    """
    Calculate the minimum distance between all pairs of crafts.

    Args:
        positions: (n_vehicles, 2) array of positions

    Returns:
        min_distance: minimum distance between any two crafts
        min_pair: tuple of indices (i, j) of the closest pair
        all_distances: dictionary with all pairwise distances
    """
    n_vehicles = len(positions)
    min_distance = float("inf")
    min_pair = None
    all_distances = {}

    for i in range(n_vehicles):
        for j in range(i + 1, n_vehicles):
            distance = np.linalg.norm(positions[i] - positions[j])
            all_distances[(i, j)] = distance

            if distance < min_distance:
                min_distance = distance
                min_pair = (i, j)

    return min_distance, min_pair, all_distances


def print_distance_analysis(initial_positions, final_positions):
    """
    Print global minimum distance (initial/final) and longest path traveled.

    Args:
        initial_positions: (n_vehicles, 2) array
        final_positions: (n_vehicles, 2) array
    """
    # --- Minimum distances ---
    init_min_dist, _, _ = calculate_minimum_distances(initial_positions)
    final_min_dist, _, _ = calculate_minimum_distances(final_positions)
    global_min_dist = min(init_min_dist, final_min_dist)

    # --- Longest path traveled ---
    displacements = np.linalg.norm(final_positions - initial_positions, axis=1)
    longest_path = np.max(displacements)
    longest_vehicle = np.argmax(displacements)

    # --- Print summary ---
    print("\n" + "=" * 40)
    print("DISTANCE SUMMARY")
    print("=" * 40)
    print(f"Global minimum distance: {global_min_dist:.3f} m")
    print(f"Longest path traveled:  {longest_path:.3f} m (Vehicle {longest_vehicle})")
    print("=" * 40 + "\n")

    return {
        "global_min_distance": global_min_dist,
        "longest_path": longest_path,
        "longest_vehicle": longest_vehicle,
    }


if __name__ == "__main__":
    # Test the position generation
    try:
        initial_pos, final_pos = generate_positions(20, min_distance=1.0)
        print("Generated positions successfully!")
        print(f"Initial positions:\n{initial_pos}")
        print(f"Final positions:\n{final_pos}")

        # Analyze distances
        distance_analysis = print_distance_analysis(initial_pos, final_pos)

        # Visualize
        visualize_layout(initial_pos, final_pos)

    except Exception as e:
        print(f"Error: {e}")
