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

    print(pl.update_settings.current_style)

    pl.save_fig(fig, f"figure_comparison_{style.lower()}.pdf")
