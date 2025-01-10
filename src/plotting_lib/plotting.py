import matplotlib as mpl
from typing import List, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
import numpy as np
from .update_settings import update_settings


def edge_change(cs, cbar=None):
    """
    Helper function to settle the bug that a contour plot saved as pdf shows
    the figure background as no contour edges are drawn.
    This functions draws the edge of each contour with the respective color.

    Parameters
    -----------

    cs: matplotlib countour QuadContourSet,
        this is returned from matplotlibs 'contourf'

    cbar: matplotlib colorbar
        Returned from mpl.colorbar

    Returns
    -----------

    """
    for c in cs.collections:
        c.set_edgecolor("face")

    if cbar:
        cbar.solids.set_edgecolor("face")
    return


def rasterize_contourf(cs):
    """
    Rasterizes contourf levels.

    Parameters
    ----------
    cs: matplotlib.collection
        returned from plt.contourf()

    Returns
    -------
    None.

    """
    for c in cs.collections:
        c.set_rasterized(True)

    return


def set_ylabel_side(ax=None, pos="right"):
    """
    Wrapper for setting y-xis label and ticks to the right (left) side of the
    figure.

    Parameters
    ----------
    ax: plt.axes instnace, optional
        axes instance which should be changed. The default is 'None'
    pos: str, optional
        y-axis tick position, either 'left' or 'right'. The default is 'right'.

    Returns
    -------
    None.

    """
    if ax is None:
        ax = plt.gca()

    ax.yaxis.set_label_position(pos)

    if pos == "right":
        ax.yaxis.tick_right()
    else:
        ax.yaxis.tick_left()

    ax.yaxis.set_ticks_position("both")
    return


def text_box(
    text,
    ax=None,
    loc="lower left",
    pad=0.3,
    borderpad=0.55,
    prop=None,
    frame=True,
    **kwargs,
):
    """
    Adds a text box to the specified ax. Placement of textbox can be easily
    controlled in a similar fashion than the legend placement.

    Parameters
    -----------

    text: string,
        text to be shown.

    ax: matplotlib axes instance, optional,
        axes object on which the text box should be drawn. Default is the last
        active ax.

    loc: string, optional,
        controlls the placement of the text box. Currently loc=`best` is
        unavaible and will lead to a ValueError. Default is `lower left`.

    pad: float, optional,
        pad around the child for drawing a frame. given in fraction of
        fontsize. Default is 0.4.

    borderpad: float, optional,
        pad between offsetbox frame and the bbox_to_anchor.

    prop: dict, optional,
        used to determine font properties. This is only used as a reference for
        paddings.

    frame: bool, optional,
        whether to put text into a frame or not. Default is `True`.

    Returns
    ----------

    ax: matplotlib axes instance
    """

    if loc == "best":
        print("loc `best` is unavaible. loc has changed back to `lower left`")
        loc = "lower left"

    anchored_text = AnchoredText(
        text, loc=loc, pad=pad, borderpad=borderpad, frameon=frame, prop=prop, **kwargs
    )
    if ax is None:
        ax = plt.gca()
    else:
        pass
    ax.add_artist(anchored_text)
    return ax


def create_fig(
    scale: Union[tuple, float] = 1.0,
    single_col: bool = False,
    width: Optional[float] = None,
    height: Optional[float] = None,
    **kwargs,
):
    """
    Creates a figure based on choosen style.

    Parameters
    -----------

    scale: tuple or float,
        rescale the figure size. If tuple, the first element is the width and
        the second element is the height. If float, the overall size is scaled by the
        float value.

    single_col: bool, optional,
        whether to use single column layout or not. Default is False.

    width: float, optional,
        width of the figure. Default is None.

    height: float, optional,

    Returns
    -----------

    fig: matplotlib figure instance
    ax: matplotlib axes instance
    """

    cols = kwargs.get("ncols", 1)
    rows = kwargs.get("nrows", 1)

    w, h = plt.rcParams["figure.figsize"]

    if single_col:
        w = width if width is not None else w
        h = height if height is not None else (h * rows)

    else:
        w = width if width is not None else (w * min(cols, 2))
        h = height if height is not None else (h * rows)

    if isinstance(scale, tuple):
        r_w, r_h = scale
    else:
        r_w = r_h = scale

    fig, ax = plt.subplots(figsize=(w * r_w, h * r_h), **kwargs)

    # good starting point for single column
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.17)

    return fig, ax


def tight_layout(fig=None, pad=0.3, **kwargs):
    """
    Tight layout wrapper that changes the default pad=1.05 to pad=0.3.

    Parameters
    ----------
    fig: matplotlib.figure instance, optional
        figure on which tight_layout should be applied to. If None it is
        applied to the last figure created. The default is None.
    pad: float, optional
        tight_layout padding in units of the fontsize. The default is 0.1.
    **kwargs: dict
        Additional keyword arguments of plt.tight_layout.

    Returns
    -------
    None.

    """
    if fig is None:
        fig = plt.gcf()
    else:
        pass

    fig.tight_layout(pad=pad, **kwargs)

    return


def tablelegend(
    ax: plt.Axes,
    ncol: int,
    col_labels: Union[List[str], None] = None,
    row_labels: Union[List[str], None] = None,
    title_label: str = "",
    sort_idx: Union[List[int], None] = None,
    *args,
    **kwargs,
):
    """
    Place a table legend on the axes.

    Creates a legend where the labels are not directly placed with the artists,
    but are used as row and column headers, looking like this:

    +---------------+---------------+---------------+---------------+
    | title_label   | col_labels[0] | col_labels[1] | col_labels[2] |
    +===============+===============+===============+===============+
    | row_labels[0] |                                               |
    +---------------+---------------+---------------+---------------+
    | row_labels[1] |             + <artists go there>              |
    +---------------+---------------+---------------+---------------+
    | row_labels[2] |                                               |
    +---------------+---------------+---------------+---------------+

    Parameters
    ----------

    ax: `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.

    ncol: int
        Number of columns.

    col_labels: list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.

    row_labels: list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.

    title_label: str, optional
        Label for the top left corner in the legend table.

    sort_idx: list of int, optional
        A list of indices to resort the handles in the legend table. If None,
        the handles are not resorted.


    Other Parameters
    ----------------

    Refer to `matplotlib.legend.Legend` for other parameters.

    Notes
    -----

    Adapted from https://stackoverflow.com/a/60345118/7119086

    """

    # obtain handles and lables from ax.legend
    handles, labels, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)
    if sort_idx:
        handles = [handles[i] for i in sort_idx]

    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_

    # modifications for table legend
    else:
        handletextpad = kwargs.pop("handletextpad", 0 if col_labels is None else -1.7)
        title_label = [title_label]

        # blank rectangle handle, used to add column and row names.
        extra = [
            Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
        ]

        # empty label
        empty = [r""]

        # number of rows infered from number of handles and desired number of
        # columns
        nrow = len(handles) // ncol

        # organise the list of handles and labels for table construction
        if col_labels is None:
            if nrow != len(row_labels):
                raise ValueError(
                    "nrow = len(handles) // ncol = {0} but must equal to"
                    " len(row_labels) = {1}.".format(nrow, len(row_labels))
                )
            leg_handles = extra * nrow
            leg_labels = row_labels

        elif row_labels is None:
            if ncol != len(col_labels):
                raise ValueError(
                    "ncol={0}, but should be equal to len(col_labels)={1}.".format(
                        ncol, len(col_labels)
                    )
                )
            leg_handles = []
            leg_labels = []

        else:
            if nrow != len(row_labels):
                raise ValueError(
                    "nrow = len(handles) // ncol = {0}, but should be equal"
                    " to len(row_labels) = {1}".format(nrow, len(row_labels))
                )

            if ncol != len(col_labels):
                raise ValueError(
                    "ncol = {0}, but should be equal to len(col_labels) = {1}.".format(
                        ncol, len(col_labels)
                    )
                )
            leg_handles = extra + extra * nrow
            leg_labels = title_label + row_labels

        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels += [col_labels[col]]
            leg_handles += handles[col * nrow : (col + 1) * nrow]
            leg_labels += empty * nrow

        # Create legend
        ax.legend_ = mlegend.Legend(
            ax,
            leg_handles,
            leg_labels,
            ncol=ncol + int(row_labels is not None),
            handletextpad=handletextpad,
            **kwargs,
        )
        ax.legend_.set_zorder(20)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_


def rescale_figure(scale, fig=None):
    """
    Helper to rescale a previously created figure.

    Parameters
    ----------
    scale : TYPE
        DESCRIPTION.
    fig : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if fig is None:
        fig = plt.gcf()
    else:
        pass

    if isinstance(scale, list):
        w, h = plt.rcParams["figure.figsize"]
        fig.set_size_inches(w * scale[0], h * scale[1])
    elif isinstance(scale, float):
        w, h = plt.rcParams["figure.figsize"]
        fig.set_size_inches(w * scale, h * scale)
    else:
        raise TypeError("Scale must be a float or list of floats")
    return


def add_label(ax, text="(a)", x0=None, y0=None, **kwargs):
    """
    Adds labels to multi-axis figures.
    """

    if isinstance(ax, np.ndarray):
        if not update_settings.current_style == "nature":
            abc = [f"({letter})" for letter in "abcdefghijklmnopqrstuvwxyz"]
        else:
            abc = [f"{letter}" for letter in "abcdefghijklmnopqrstuvwxyz"]

        f = ax.flat[0].get_figure()
        if y0 is None:
            y0 = f.subplotpars.top
        if x0 is None:
            x0 = f.subplotpars.left
        for i, _ax in enumerate(ax.flat):
            _ax.annotate(
                r"\textbf{{{}}} ".format(abc[i]),
                xy=(-x0, y0),
                annotation_clip=False,
                xycoords="axes fraction",
                weight="bold",
                fontsize=plt.rcParams["font.size"],
                va="bottom",
                **kwargs,
            )
    else:
        fig = ax.get_figure()
        text = text.strip("()") if update_settings.current_style == "nature" else text
        if x0 is None:
            x0 = fig.subplotpars.left
        if y0 is None:
            y0 = fig.subplotpars.top
        if plt.rcParams["text.usetex"] is True:
            ax.annotate(
                r"\textbf {{{}}}".format(text),
                xy=(-x0, y0),
                annotation_clip=False,
                xycoords="axes fraction",
                weight="bold",
                fontsize=plt.rcParams["font.size"],
                va="top",
                # ha="center",
                **kwargs,
            )
        else:
            ax.annotate(
                r"{}".format(text),
                xy=(-x0, y0),
                annotation_clip=False,
                xycoords="axes fraction",
                weight="bold",
                fontsize=plt.rcParams["font.size"],
            )

    return


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """

    if num == 0:
        return r"${0:.{1}f}$".format(num, decimal_digits)

    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f} \times 10^{{{1:d}}}$".format(coeff, exponent, precision)


def restore_minor_ticks_log_plot(ax: Optional[plt.Axes] = None, n_subticks=9) -> None:
    """For axes with a logrithmic scale where the span (max-min) exceeds
    10 orders of magnitude, matplotlib will not set logarithmic minor ticks.
    If you don't like this, call this function to restore minor ticks.

    Args:
        ax:
        n_subticks: Number of Should be either 4 or 9.

    Returns:
        None
    """
    if ax is None:
        ax = plt.gca()
    # Method from SO user importanceofbeingernest at
    # https://stackoverflow.com/a/44079725/5972175
    # if which==''
    locmaj = mpl.ticker.LogLocator(base=10, numticks=1000)
    ax.xaxis.set_major_locator(locmaj)
    locmin = mpl.ticker.LogLocator(
        base=10.0, subs=np.linspace(0, 1.0, n_subticks + 2)[1:-1], numticks=1000
    )
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def plot_dark_background(ax):
    ax.set_facecolor("none")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.title.set_color("white")
    ax.legend().get_frame().set_edgecolor("white")
    ax.legend(loc="lower right", labelcolor="white", edgecolor="white")
    # set minor ticks to white
    ax.tick_params(axis="x", which="minor", colors="white")
    ax.tick_params(axis="y", which="minor", colors="white")


def errorbar_marker_style(color, marker, factor=0.5, **kwargs):
    """

    Parameters
    ----------

    color: str
        Color of the marker.

    marker: str
        Marker style.

    factor: float, optional
        Lighten factor. The default is 0.5.

    **kwargs: dict
        Additional keyword arguments.

    Returns
    -------
    dict
        Dictionary of marker style.

    """

    return dict(
        markerfacecolor=lighten_color(color, factor),
        markeredgecolor=color,
        markersize=kwargs.pop("markersize", 5),
        marker=marker,
        elinewidth=kwargs.pop("elinewidth", 0.5),
        capsize=kwargs.pop("capsize", 1.5),
        capthick=kwargs.pop("capthick", 0.5),
        color=color,
    )


def plot_marker_style(
    color, marker="o", ls="solid", ms=4.5, markeredgecolor=None, factor=0.5
):
    """

    Parameters
    ----------
    color : str
        Color of the marker.

    marker : str, optional

    ls : str, optional

    ms : float, optional

    markeredgecolor : str, optional

    factor : float, optional
        Lighten factor. The default is 0.5.

    """

    return dict(
        markerfacecolor=lighten_color(color, factor),
        markeredgecolor=color if markeredgecolor is None else markeredgecolor,
        markersize=ms,
        linestyle=ls,
        marker=marker,
    )


def save_fig(fig, filename, **kwargs):
    """
    Save figure with a specific filename.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure instance.
    filename : str
        Filename.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    None.

    """
    fig.savefig(filename, **kwargs)
    return
