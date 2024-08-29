import numpy as np
import plotting_lib as pl
import matplotlib.pyplot as plt


for style in ["APS", "Nature", "Quantum"]:
    pl.update_settings(usetex=True, style=style, colors=pl.colors_rsb)

    # Data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = pl.create_fig()

    # Plot
    ax.plot(x, y, label=r"$\sin(x)$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend()
    pl.add_label(ax)

    pl.tight_layout()

    pl.save_fig(fig, f"figure_comparison_{style.lower()}.pdf")


for style in ["APS", "Nature", "Quantum"]:
    pl.update_settings(usetex=True, style=style, colors=pl.colors_rsb)

    fig, ax = pl.create_fig(ncols=2, sharey=True, single_col=True)
    for axis in ax:
        axis.plot(x, -y)
        axis.set_xlabel(r"$x$")
        # axis.set_ylabel(r"$y$")

    pl.add_label(ax, x0=-0.06, y0=0.87)
    pl.tight_layout()
