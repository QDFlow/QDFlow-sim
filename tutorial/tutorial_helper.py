
from __future__ import annotations

import numpy as np
import matplotlib.figure
import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Any


def calc_mesh_grid(x_vals:NDArray[np.float64], y_vals:NDArray[np.float64]) \
                  -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''
    Calculates a mesh giving the coordinates of the corners of each pixel for
    use in functions like ``plt.pcolormesh()``.

    Parameters
    ----------
    x_vals, y_vals : NDArray[np.float64]
        Arrays with length ``width`` and ``height`` respectively, giving the
        values of the plunger voltages at each point along the x- and y-axes.
    
    Returns
    -------
    x_mesh, y_mesh : NDArray[np.float64]
        Meshgrid giving the corners of each of the pixels.
    '''    
    x_bound = np.zeros(len(x_vals)+1)
    x_bound[1:-1] = (x_vals[:-1]+x_vals[1:])/2
    x_bound[0] = (3*x_vals[0]-x_vals[1])/2
    x_bound[-1] = (3*x_vals[-1]-x_vals[-2])/2
    y_bound = np.zeros(len(y_vals)+1)
    y_bound[1:-1] = (y_vals[:-1]+y_vals[1:])/2
    y_bound[0] = (3*y_vals[0]-y_vals[1])/2
    y_bound[-1] = (3*y_vals[-1]-y_vals[-2])/2
    x_mesh, y_mesh = np.meshgrid(x_bound, y_bound, indexing="ij")
    return x_mesh, y_mesh


def plot_csd_data(fig:matplotlib.figure.Figure, ax:matplotlib.axes.Axes, data:NDArray[np.float64], *,
                  x_y_vals:tuple[NDArray[np.float64],NDArray[np.float64]]|None=None):
    '''
    Plots data from a charge stability diagram.

    Parameters
    ----------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes on which to plot the CSD.
    data : NDArray[np.float64]
        An array with shape ``(width, height)`` giving the data to plot.
    x_y_vals : tuple[NDArray[np.float64], NDArray[np.float64]]
        A tuple ``(x_vals, y_vals)``, where ``x_vals`` and ``y_vals`` are arrays
        with length ``width`` and ``height`` respectively, giving the values of
        the plunger voltages at each point along the x- and y-axes.
    '''
    if x_y_vals is None:
        x_vals = np.arange(data.shape[-2])
        y_vals = np.arange(data.shape[-1])
    else:
        x_vals, y_vals = x_y_vals
    x_mesh, y_mesh = calc_mesh_grid(x_vals, y_vals)
    ax.pcolormesh(x_mesh, y_mesh, data, cmap="grey_r")
    
    if x_y_vals is None:
        unit = "pixels"
    else:
        unit = "mV"
    ax.set_xlabel("Left Plunger (" + unit + ")")
    ax.set_ylabel("Right Plunger (" + unit + ")")
    ax.set_box_aspect(1)
    

def overlay_boolean_data(fig:matplotlib.figure.Figure, ax:matplotlib.axes.Axes,
                         data:NDArray[np.bool], *,
                         x_y_vals:tuple[NDArray[np.float64],NDArray[np.float64]]|None=None,
                         labels:None|tuple[str,str]=None,
                         colors:tuple[tuple[float,...],tuple[float,...]]=((1,0,0,.25),(0,0,1,.25))):
    '''
    Overlays boolean state data over top of a charge stability diagram.

    Parameters
    ----------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes on which to plot the data.
    data : NDArray[np.bool]
        An array with shape ``(width, height)`` giving the data to plot.
    x_y_vals : tuple[NDArray[np.float64], NDArray[np.float64]]
        A tuple ``(x_vals, y_vals)``, where ``x_vals`` and ``y_vals`` are arrays
        with length ``width`` and ``height`` respectively, giving the values of
        the plunger voltages at each point along the x- and y-axes.
    labels : tuple[NDArray[np.float64], NDArray[np.float64]]
        The labels of the states False and True respectively.
    colors : tuple[tuple[float,...], tuple[float,...]]
        The colors in RGBA format corresponding to False and True respectively.
    '''
    if x_y_vals is None:
        x_vals = np.arange(data.shape[-2])
        y_vals = np.arange(data.shape[-1])
    else:
        x_vals, y_vals = x_y_vals
    x_mesh, y_mesh = calc_mesh_grid(x_vals, y_vals)
    c = ax.pcolormesh(x_mesh, y_mesh, np.where(data,1,0),
                      cmap=matplotlib.colors.ListedColormap(colors))
    if labels is not None:
        cbar = fig.colorbar(c)
        cbar.set_ticks([0.4, 0.9])
        cbar.ax.tick_params(size=0)
        cbar.set_ticklabels(labels, rotation=90)


def plot_dist_data(fig:matplotlib.figure.Figure, ax:matplotlib.axes.Axes,
                   data:NDArray[np.float64], *, bins:int=15):
    '''
    Plots a histogram of ``data``.

    Parameters
    ----------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes on which to plot the data.
    data : NDArray[np.float64]
        A 1D array giving the data to plot.
    bins : int
        How many bins to use in the histogram.
    '''
    x0 = max(3 * np.percentile(data, 1) - 2 * np.percentile(data, 2), np.min(data))
    x1 = min(3 * np.percentile(data, 99) - 2 * np.percentile(data, 98), np.max(data))
    if x0 == x1:
        x0 = x0 - .5
        x1 = x1 + .5
    ax.hist(data, bins=bins, range=(x0, x1))
    ax.set_ylabel("Count")
    ax.set_xlabel("Value")


def overlay_rays(fig:matplotlib.figure.Figure, ax:matplotlib.axes.Axes,
                 point:NDArray[np.float64], rays:NDArray[np.float64], ray_num_to_highlight:int):
    '''
    Overlays rays over top of a charge stability diagram.

    Parameters
    ----------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes on which to plot the data.
    point : NDArray[np.float64]
        The central point ``[x, y]`` that all the rays extend out from.
    rays : NDArray[np.float64]
        An array with shape ``[num_rays, 2]`` giving the difference on the
        endpoint and initial point of each ray.
    ray_num_to_highlight : int
        This ray will be plotted in a different color
    '''
    for i, ray in enumerate(rays):
        if i == ray_num_to_highlight:
            ax.plot([point[0], point[0] + ray[0]], [point[1], point[1] + ray[1]], color="blue")
        else:
            ax.plot([point[0], point[0] + ray[0]], [point[1], point[1] + ray[1]], color="red")
    ax.scatter(point[0], point[1], color="blue", zorder=3)
    

def plot_potential(fig:matplotlib.figure.Figure, ax:matplotlib.axes.Axes,
                   x:NDArray[np.float64], qV:NDArray[np.float64], *,
                   color:str="blue", mu:float|None=None, ylabel:str="qV (meV)"):
    '''
    Plots the potential energy of a charge an the nanowire.

    Parameters
    ----------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes on which to plot the data.
    x : NDArray[np.float64]
        The x-values of the data to plot.
    qV : NDArray[np.float64]
        The potential energy at each point in ``x``.
    color : str
        The color to use to plot
    mu : float | None
        The Fermi level to plot
    ylabel : str
        The label of the y-axis
    '''
    ax.plot(x, qV, color=color)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel(ylabel)
    if mu is not None:
        ax.plot(x, np.full(x.shape, mu), color="orange")
        ax.text(x[0], mu - (np.max(qV) - np.min(qV))/12, "$\\mu$", color="orange")


def plot_n(fig:matplotlib.figure.Figure, ax:matplotlib.axes.Axes,
           x:NDArray[np.float64], n:NDArray[np.float64]):
    '''
    Plots the charge density along the nanowire.

    Parameters
    ----------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes on which to plot the data.
    x : NDArray[np.float64]
        The x-values of the data to plot.
    n : NDArray[np.float64]
        The charge density at each point in ``x``,
        or an array with shape (num_iterations, len(x)) giving several functions to plot.
    '''
    if len(n.shape) == 1 or n.shape[0] == 1:
        ax.plot(x, n, color="red")
    else:
        n_it = n.shape[0]
        for i in range(n_it):
            if i < (n_it-1)/3:
                clr = (3*i/(n_it-1), 0, 1)
            elif i < 2*(n_it-1)/3:
                clr = (.9+.1*(2*(n_it-1)-3*i)/(n_it-1), 0, (2*(n_it-1)-3*i)/(n_it-1))
            else:
                clr = (.9, .7*(3*i-2*(n_it-1))/(n_it-1), 0)
            ax.plot(x, n[i], color=clr)   
        ax.legend(["iteration " + str(i+1) for i in range(n_it)], bbox_to_anchor=(1.05, 1), loc='upper left')     
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("n (1/nm)")


def plot_islands_and_barriers(fig:matplotlib.figure.Figure, ax:matplotlib.axes.Axes,
           x:NDArray[np.float64], n:NDArray[np.float64], islands:NDArray[np.int_],
           barriers:NDArray[np.int_], cutoff:float):
    '''
    Plots the regions defined as islands and barriers on the nanowire.

    Parameters
    ----------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes on which to plot the data.
    x : NDArray[np.float64]
        The x-values of the data to plot.
    n : NDArray[np.float64]
        The charge density at each point in ``x``.
    islands : ndarray[int]
        An array with shape ``(n_islands, 2)`` giving the indeces of the
        endpoints ``[begin_index, end_index + 1]``.
    barriers : ndarray[int]
        An array with shape ``(n_barriers, 2)`` giving the indeces of the
        endpoints ``[begin_index, end_index + 1]``.
    cutoff : float
        The cutoff value between islands and barriers.
    '''
    isl_b = -1*np.max(n) / 8
    isl_a = -2*np.max(n) / 8
    bar_b = -1*np.max(n) / 8
    bar_a = -2*np.max(n) / 8
    ax.plot(x, np.full(len(x),cutoff), color="orange")
    ax.plot([], [], color="green")
    ax.plot([], [], color="blue")
    for isl in islands:
        ax.fill_between(x[isl[0]:isl[1]], isl_a, isl_b, color=(.6,.9,.6))
    for bar in barriers:
        ax.fill_between(x[bar[0]:bar[1]], bar_a, bar_b, color=(.6,.6,.9))



    