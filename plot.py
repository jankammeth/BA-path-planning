import matplotlib.pyplot as plt

# ==== INPUT DATA (edit these) ====
# Each tuple: (frequency_Hz, min_height, max_height)
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

def plot_errorbars(ax, data, title):
    xs, ys, yerr_low, yerr_high = [], [], [], []
    for f, ymin, ymax in data:
        if ymin is not None and ymax is not None:
            mid = 0.5 * (ymin + ymax)
            xs.append(f)
            ys.append(mid)
            yerr_low.append(mid - ymin)
            yerr_high.append(ymax - mid)
        elif ymin is not None:  # only one value
            xs.append(f)
            ys.append(ymin)
            yerr_low.append(0)
            yerr_high.append(0)
        elif ymax is not None:
            xs.append(f)
            ys.append(ymax)
            yerr_low.append(0)
            yerr_high.append(0)

    ax.errorbar(xs, ys, 
                yerr=[yerr_low, yerr_high],
                fmt='o', capsize=5, elinewidth=1.5, markersize=6)
    ax.set_title(title)
    ax.set_xlabel("Rotationsgeschwindigkeit in Hz")
    ax.set_ylabel("Höhe")
    ax.grid(True, linestyle="--", alpha=0.4)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
plot_errorbars(axes[0], data_8, "8-Windungen")
plot_errorbars(axes[1], data_16, "16-Windungen")
plt.tight_layout()
plt.show()