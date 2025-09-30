#!/usr/bin/env python3
"""
WOW Fleet Position Generator & Analysis

- Generate initial/final positions in a 20x20 box
- Analyze distances (minimum spacing, longest path)
- Visualize layout with quadrant colors
"""

import random

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

# ----------------- Layout constants -----------------
BOX_SIZE = 20.0
CIRCLE_DIAM = 5.0
CIRCLE_RADIUS = CIRCLE_DIAM / 2.0
DIAMOND_SIDE = 6.0
DIAMOND_CENTER = np.array([BOX_SIZE / 2, BOX_SIZE / 2])
CIRCLE_CENTERS = np.array(
    [
        [3.5, 3.5],
        [16.5, 3.5],
        [3.5, 16.5],
        [16.5, 16.5],
    ]
)
# Distance from center to diamond vertex
DIAMOND_SIZE = DIAMOND_SIDE / np.sqrt(2)
DIAMOND_VERTICES = np.array(
    [
        [DIAMOND_CENTER[0], DIAMOND_CENTER[1] + DIAMOND_SIZE],
        [DIAMOND_CENTER[0] + DIAMOND_SIZE, DIAMOND_CENTER[1]],
        [DIAMOND_CENTER[0], DIAMOND_CENTER[1] - DIAMOND_SIZE],
        [DIAMOND_CENTER[0] - DIAMOND_SIZE, DIAMOND_CENTER[1]],
    ]
)
# -----------------------------------------------------


def generate_positions(n_vehicles, min_distance=0.4, max_attempts=1000):
    """
    Generate random initial and final positions:
      - Initial: on corner circles
      - Final: mostly on diamond border, sometimes on circles
    """
    # Initial positions (on circles)
    initial_positions, attempts = [], 0
    while len(initial_positions) < n_vehicles and attempts < max_attempts:
        center = CIRCLE_CENTERS[random.randint(0, 3)]
        new_pos = _sample_point_on_circle(center)
        if _is_valid_position(new_pos, initial_positions, min_distance):
            initial_positions.append(new_pos)
        attempts += 1
    if len(initial_positions) < n_vehicles:
        raise ValueError("Could not generate enough initial positions.")

    # Final positions (diamond 90%, circles 10%)
    final_positions, attempts = [], 0
    while len(final_positions) < n_vehicles and attempts < max_attempts:
        if random.random() < 0.9:
            new_pos = _sample_point_on_diamond_border()
        else:
            center = CIRCLE_CENTERS[random.randint(0, 3)]
            new_pos = _sample_point_on_circle(center)
        if _is_valid_position(new_pos, final_positions, min_distance):
            final_positions.append(new_pos)
        attempts += 1
    if len(final_positions) < n_vehicles:
        raise ValueError("Could not generate enough final positions.")

    return np.array(initial_positions), np.array(final_positions)


def visualize_scenario(initial_positions, final_positions, min_distance=0.4):
    """
    Plot positions in the 20x20 box:
      - Start (circle), End (square), same color per craft
      - Connections, safety margins, layout shapes
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    # Boundary + layout
    ax.add_patch(
        Rectangle(
            (0, 0),
            BOX_SIZE,
            BOX_SIZE,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            alpha=0.7,
        )
    )
    for c in CIRCLE_CENTERS:
        ax.add_patch(Circle(c, CIRCLE_RADIUS, edgecolor="grey", facecolor="none", alpha=0.7))
    diamond_x = np.append(DIAMOND_VERTICES[:, 0], DIAMOND_VERTICES[0, 0])
    diamond_y = np.append(DIAMOND_VERTICES[:, 1], DIAMOND_VERTICES[0, 1])
    ax.plot(diamond_x, diamond_y, color="grey", alpha=0.7)

    # Craft positions
    colors, _ = _quadrant_colors(initial_positions)
    for i in range(len(initial_positions)):
        # Start
        ax.scatter(*initial_positions[i], marker="o", s=150, color=colors[i])
        ax.add_patch(
            Circle(
                initial_positions[i],
                min_distance / 2,
                edgecolor=colors[i],
                facecolor="none",
                alpha=0.3,
            )
        )
        # End
        ax.scatter(*final_positions[i], marker="s", s=150, color=colors[i])
        ax.add_patch(
            Circle(
                final_positions[i],
                min_distance / 2,
                edgecolor=colors[i],
                facecolor="none",
                alpha=0.3,
            )
        )
        # Path
        ax.plot(
            [initial_positions[i, 0], final_positions[i, 0]],
            [initial_positions[i, 1], final_positions[i, 1]],
            color=colors[i],
            alpha=0.3,
            linewidth=1.5,
        )

    # Legend
    ax.legend(
        handles=[
            mlines.Line2D([], [], color="black", marker="o", ls="None", label="Start"),
            mlines.Line2D([], [], color="black", marker="s", ls="None", label="Stop"),
        ],
        loc="lower right",
    )

    ax.set_xlim(-1, BOX_SIZE + 1)
    ax.set_ylim(-1, BOX_SIZE + 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_title("Initial and Final Craft Positions")

    plt.tight_layout()
    plt.savefig("plots/position_generator.pdf", dpi=400)
    plt.show()
    return fig, ax


def print_distance_analysis(initial_positions, final_positions):
    """Print min distances and longest path."""
    init_min, _, _ = _calculate_minimum_distances(initial_positions)
    final_min, _, _ = _calculate_minimum_distances(final_positions)
    global_min = min(init_min, final_min)

    displacements = np.linalg.norm(final_positions - initial_positions, axis=1)
    longest_path = displacements.max()
    longest_vehicle = displacements.argmax()

    print("\n" + "=" * 40)
    print("DISTANCE SUMMARY")
    print("=" * 40)
    print(f"Global minimum distance: {global_min:.3f} m")
    print(f"Longest path traveled:  {longest_path:.3f} m (Vehicle {longest_vehicle})")
    print("=" * 40 + "\n")

    return dict(
        global_min_distance=global_min, longest_path=longest_path, longest_vehicle=longest_vehicle
    )


def _calculate_minimum_distances(positions):
    """Return min distance, closest pair, and all pairwise distances."""
    n = len(positions)
    all_dists, min_dist, min_pair = {}, float("inf"), None
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            all_dists[(i, j)] = d
            if d < min_dist:
                min_dist, min_pair = d, (i, j)
    return min_dist, min_pair, all_dists


def _quadrant_colors(initial_positions, center=DIAMOND_CENTER):
    """
    Assign quadrant-based colors from initial positions.
    """
    palette = [
        (0.17, 0.28, 0.46),  # Q0 top-right
        (0.54, 0.31, 0.56),  # Q1 top-left
        (1.00, 0.39, 0.38),  # Q2 bottom-left
        (1.00, 0.65, 0.00),  # Q3 bottom-right
    ]
    colors, quad_ids = [], []
    cx, cy = center
    for x, y in initial_positions:
        if x >= cx and y >= cy:
            q = 0
        elif x < cx and y >= cy:
            q = 1
        elif x < cx and y < cy:
            q = 2
        else:
            q = 3
        quad_ids.append(q)
        colors.append(palette[q])
    return colors, quad_ids


# ---------- Sampling helpers ----------
def _sample_point_on_circle(center, radius=CIRCLE_RADIUS):
    angle = random.uniform(0, 2 * np.pi)
    return center + radius * np.array([np.cos(angle), np.sin(angle)])


def _sample_point_on_diamond_border(vertices=DIAMOND_VERTICES):
    edge = random.randint(0, 3)
    v1, v2 = vertices[edge], vertices[(edge + 1) % 4]
    t = random.uniform(0, 1)
    return v1 + t * (v2 - v1)


def _is_valid_position(new_pos, existing_positions, min_dist):
    return all(np.linalg.norm(new_pos - pos) >= min_dist for pos in existing_positions)


# TODO: rename position_generator to scenario generator for clarity
if __name__ == "__main__":
    """
    Example usage of the position generator.
    """
    try:
        # Generate initial and final positions
        initial_pos, final_pos = generate_positions(20, min_distance=1.0)
        print("Generated positions successfully!")

        # Analyze distances
        # 1) maximum straight distance any robot has to travel (lower bound)
        # 2) global minimum position between any two robots at initial of final time
        distance_analysis = print_distance_analysis(initial_pos, final_pos)

        # Visualize
        visualize_scenario(initial_pos, final_pos)

    except Exception as e:
        print(f"Error: {e}")
