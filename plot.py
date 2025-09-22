import matplotlib.pyplot as plt
import numpy as np

# ==== INPUT DATA ====
data_8  = [
    (0.60, 1.478, 1.478),
    (0.96, 0.994, 0.998),
    (1.50, 0.567, 0.597),
]
data_16 = [
    (0.35, 1.648, 1.648),
    (0.90, 1.648, 1.648),
    (1.46, 0.63, 0.881),
]

X_MIN, X_MAX = 0.0, 1.6
Y_MIN, Y_MAX = 0.0, 1.8

def extract_xy_with_errors(data):
    xs, ys, yerr_low, yerr_high = [], [], [], []
    for f, ymin, ymax in data:
        if ymin is not None and ymax is not None:
            mid = 0.5 * (ymin + ymax)
            xs.append(f); ys.append(mid)
            yerr_low.append(mid - ymin); yerr_high.append(ymax - mid)
        elif ymin is not None:
            xs.append(f); ys.append(ymin); yerr_low.append(0); yerr_high.append(0)
        elif ymax is not None:
            xs.append(f); ys.append(ymax); yerr_low.append(0); yerr_high.append(0)
    return np.array(xs), np.array(ys), np.array(yerr_low), np.array(yerr_high)

def add_trendline(ax, xs, ys, idxs, label="Trendlinie", color="red"):
    """Fit y = m x + b using points xs[idxs], ys[idxs]; draw & return (m, b)."""
    x_fit = xs[idxs]
    y_fit = ys[idxs]
    m, b = np.polyfit(x_fit, y_fit, 1)
    x_line = np.linspace(min(x_fit), max(x_fit), 100)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, '--', color=color, label=label)
    return m, b

def plot_panel(ax, data, title, trend_idxs, extra_segment=None):
    xs, ys, e_lo, e_hi = extract_xy_with_errors(data)
    ax.errorbar(xs, ys, yerr=[e_lo, e_hi], fmt='o', capsize=5, elinewidth=1.5,
                markersize=6, label="Messwerte")

    # Trendline
    m, b = add_trendline(ax, xs, ys, trend_idxs, label="Trendlinie", color="red")
    print(f"{title}: y = {m:.3f} * x + {b:.3f}")

    # Optional extra connecting segment (e.g., first two points)
    if extra_segment is not None and len(extra_segment) == 2:
        i, j = extra_segment
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], '-', linewidth=1.5,
                label="Verbindung P1–P2")

    ax.set_title(title)
    ax.set_xlabel("Rotationsgeschwindigkeit in Hz")
    ax.set_ylabel("Höhe")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")   # 👈 legend in bottom right

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

# 8-Windungen: fit through all 3 points
plot_panel(axes[0], data_8, "8-Windungen", trend_idxs=[0,1,2])

# 16-Windungen: fit through last 2 points; also connect first two points
plot_panel(axes[1], data_16, "16-Windungen", trend_idxs=[1,2], extra_segment=(0,1))

plt.tight_layout()
plt.savefig("errorbars_trendlines_console_legends.png", dpi=300, bbox_inches="tight")
plt.show()