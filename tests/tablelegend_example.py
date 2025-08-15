import numpy as np
import plotting_lib as pl
import matplotlib.pyplot as plt

from plotting_lib.plotting import plot_marker_style

for style in ["APS", "Nature", "Quantum"]:
    pl.update_settings(usetex=True, style=style, colors=pl.colors_rsb)

    fig, ax = pl.create_fig()

    # Data
    x = np.linspace(0, 10, 100)

    ax.plot(x, np.sin(x * 1.0), 
            **plot_marker_style(color="C1", ls="solid", marker="s"), markevery=10,
            label=" ")
    ax.plot(x, np.sin(x * 1.25), 
            **plot_marker_style(color="C1", ls="solid", marker="o"), markevery=10,
            label=" ")
    
    ax.plot(x, np.sin(x * 1.5), 
            **plot_marker_style(color="C2", ls="solid", marker="s"), markevery=10,
            label=" ")
    ax.plot(x, np.sin(x * 1.75), 
            **plot_marker_style(color="C2", ls="solid", marker="o"), markevery=10,
            label=" ")
    
    ax.set_ylim(None, 2)
    
    col_labels = ["A", "B"]
    row_labels = ["1", "2"]
    title_label = "T"

    pl.tablelegend(ax,
                   ncol=2,
                   col_labels=col_labels,
                   row_labels=row_labels,
                   title_label=title_label,
                   sort_idx=[0, 1, 2, 3])
    
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    
    pl.tight_layout()

    pl.save_fig(fig, f"tests/figure_tablelegend_{style.lower()}.pdf")