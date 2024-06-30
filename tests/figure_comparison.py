import numpy as np
import plottinglib as pl
import matplotlib.pyplot as plt

pl.update_settings(usetex=True, style="APS", colors=pl.colors_rsb)

# Data
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = pl.create_fig()


# Plot
ax.plot(x, y, label="sin(x)")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.legend()

pl.save_fig(fig, "figure_comparison.pdf")
