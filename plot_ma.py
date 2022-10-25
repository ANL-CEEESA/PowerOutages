#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dataloader import concise_old_feat_list
from utils import avg
from tqdm import tqdm
from scipy import sparse
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import gaussian_kde

def plot_illustration(locs):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    # make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.6,
        width=.4E6, height=.3E6)
    # m.etopo()
    m.drawrivers()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # rescale marker size
    sct = m.scatter(locs[:37, 1], locs[:37, 0], latlon=True, alpha=0.5, s=10, c="black")
    sct = m.scatter(locs[38:, 1], locs[38:, 0], latlon=True, alpha=0.5, s=10, c="black")
    sct = m.scatter(locs[37, 1], locs[37, 0], latlon=True, alpha=0.5, s=1500, c="red")
    sct = m.scatter(locs[37, 1], locs[37, 0], latlon=True, alpha=0.5, s=20, c="blue")

    plt.savefig("imgs/data_illustration.pdf")

def plot_interval_qqplot(obs_outage, start_date, filename, dayinterval=3, vmax=None, vmin=None):

    lowerb     = 10

    # dates
    N, K       = obs_outage.shape
    start_date = arrow.get(start_date, "YYYY-MM-DD HH:mm:ss")
    n_date     = int(N / (24 * dayinterval))
    dates      = [ start_date.shift(days=i * dayinterval) for i in range(n_date + 1) ]

    # extract points
    interval_points = []
    for k in range(K):
        interval_start, interval_end, interval_max = 0, -1, obs_outage[0, k]
        for t in range(1, N):
            # print("k: %d, t: %d, num: %d" % (k, t, obs_outage[t, k]))
            if obs_outage[t-1, k] <= lowerb and obs_outage[t, k] > lowerb:
                interval_start = t
                interval_max   = obs_outage[t, k]
            elif obs_outage[t-1, k] > lowerb and obs_outage[t, k] > lowerb:
                interval_max   = max(interval_max, obs_outage[t, k])
            elif obs_outage[t-1, k] > lowerb and obs_outage[t, k] <= lowerb:
                interval_end   = t
                assert interval_start != -1 and interval_end != -1 and interval_max != 0, \
                    "Invalid event detected: (start: %d, end: %d, max: %d)" % (interval_start, interval_end, interval_max)
                interval_points.append([interval_start, interval_end, interval_max])
                interval_start, interval_end, interval_max = -1, -1, 0
    interval_points = np.array(interval_points)
    # interval_points = interval_points[np.where(interval_points[:, 2] > 1)[0]]
    print(interval_points.shape)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'sans-serif',
        'sans-serif': ['Helvetica'],
        # 'weight' : 'bold',
        'size'   : 18}
    plt.rc('font', **font)

    fig   = plt.figure(figsize=(6, 6))
    ax    = fig.add_subplot(111)
    cmap  = matplotlib.cm.get_cmap('Blues')

    # plot diagonal line
    x = np.arange(interval_points[:, 0].min(), interval_points[:, 0].max())
    y = np.arange(interval_points[:, 0].min(), interval_points[:, 0].max())
    ax.plot(x, y, linestyle="--", linewidth=3, c="black", alpha=0.6)

    # value range
    vmin = vmin if vmin is not None else interval_points[:, 2].min()
    vmax = vmax if vmax is not None else interval_points[:, 2].max()
    print(interval_points[:, 2].min(), interval_points[:, 2].max())
    
    # size
    mins, maxs = 10, 800
    size = (interval_points[:, 2] - vmin) / (vmax - vmin)
    size = size * (maxs - mins) + mins

    vmin = np.log(vmin)
    vmax = np.log(vmax)
    
    ax.scatter(interval_points[:, 0], interval_points[:, 1], 
        alpha=0.5, s=size, cmap=cmap, c=np.log(interval_points[:, 2]), 
        vmin=vmin, vmax=vmax)
    
    # ax.set_xlabel(r"Start time")
    # ax.set_ylabel(r"End time")
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(filename)

    plt.xticks(np.arange(0, N, int(24 * dayinterval)), [ str(date.format('MMM DD')) for date in dates ], rotation=90)
    plt.yticks(np.arange(0, N, int(24 * dayinterval)), [ str(date.format('MMM DD')) for date in dates ])
    fig.tight_layout()
    fig.savefig("imgs/%s.pdf" % "".join(filename.split(" ")))

def plot_data_on_map(data, locs, filename, dmin=None, dmax=None):
    """
    Args:
    - data: [ n_locs ]

    References:
    - https://python-graph-gallery.com/315-a-world-map-of-surf-tweets/
    - https://matplotlib.org/basemap/users/geography.html
    - https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html
    """

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    # Make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.6,
        width=.4E6, height=.3E6)
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # rescale marker size
    mins, maxs = 5, 300
    dmin, dmax = data.min() if dmin is None else dmin, data.max() if dmax is None else dmax
    print(dmin, dmax)
    size = (data - data.min()) / (data.max() - data.min())
    size = size * (maxs - mins) + mins
    sct  = m.scatter(locs[:, 1], locs[:, 0], latlon=True, alpha=0.5, s=size, c="r")
    handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
        func=lambda s: (s - mins) / (maxs - mins) * (dmax - dmin) + dmin)
    plt.title(filename)
    plt.legend(handles, labels, loc="lower left", title="Num of outages")

    plt.savefig("imgs/%s.pdf" % filename)

def plot_outage_and_weather_map(outage_locs, weather_locs, obs_outage, obs_weather_show, 
    outage_min=0, outage_max=20000, weather_min=0, weahter_max=30,
    filename="outage-vs-weather-map", titlename="Mar 1st, 2018 MA"):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'sans-serif',
        'sans-serif': ['Helvetica'],
        # 'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    feat_ind = 0

    print(titlename)
    print("outage", obs_outage.max(), obs_outage.min())
    print("total outage", obs_outage.sum())
    print("weather", obs_weather_show[:, feat_ind].max(), obs_weather_show[:, feat_ind].min())

    # Make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.0,
        width=.2E6, height=.2E6)
    # m.drawlsmask(grid=1.25)
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # plot weather
    cm1  = plt.cm.get_cmap('Reds')
    sct1 = m.pcolor(weather_locs[:, 1], weather_locs[:, 0], obs_weather_show[:, feat_ind],
        latlon=True, alpha=0.3, cmap=cm1, tri=True,
        vmin=weather_min, vmax=weahter_max)

    # colorbar
    # cb  = plt.colorbar(sct1, fraction=0.046, pad=0.04, label='Derived radar reflectivity [dB]')
    # cax = cb.ax
    # fontsize       = 20
    # fontweight     = 'normal'
    # fontproperties = {'family': 'sans-serif', 'fontname': 'Helvetica', 'weight': fontweight, 'size': fontsize}
    # # cax.set_xticklabels(cax.get_xticks(), fontproperties)
    # # cax.set_yticklabels(cax.get_yticks(), fontproperties)
    # cax.set_yticklabels(np.arange(weather_min, weahter_max + 5, 5).astype(int), fontproperties)
    
    # plot outage
    mins, maxs = 0, 1500
    dmin, dmax = outage_min, outage_max
    size = (obs_outage - dmin) / (dmax - dmin)
    size = size * (maxs - mins) + mins
    cm2  = plt.cm.get_cmap('winter_r')
    sct2 = m.scatter(outage_locs[:, 1], outage_locs[:, 0], 
        latlon=True, alpha=0.5, s=size, 
        c=obs_outage, cmap=cm2, vmin=outage_min, vmax=outage_max)

    # # legend
    # val = [ 1000, 5000, 10000, 20000 ]
    # lab = [ r"1k", r"5k", r"10k", r"20k" ]
    # # clr = [ "#E6EFFA", "#B6D4E9", "#2B7BBA", "#083471" ]
    # clr = [ "gray", "gray", "gray", "gray" ]
    # for v, l, c in zip(val, lab, clr):
    #     size = (v - dmin) / (dmax - dmin)
    #     size = size * (maxs - mins) + mins
    #     ax.scatter([40], [40], alpha=0.5, c=c, s=[size], label=l)
    # ax.legend(loc="lower left", title="Num of outages", handleheight=1.8)

    plt.title(titlename)
    fig.tight_layout()
    plt.savefig("imgs/%s.pdf" % filename)

def plot_outage_and_weather_linechart(titlename, N, start_date, 
    obs_outage, obs_weather_show, obs_weather_normal, 
    labels, colors, dayinterval=3):

    obs_outage     = obs_outage / 15
    weather_show   = obs_weather_show.sum(0)
    weather_normal = obs_weather_normal.sum(0)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'sans-serif',
        'sans-serif': ['Helvetica'],
        # 'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # time slots
    time1   = np.arange(0, N*3, 3)
    time2   = np.arange(0, 3*N, 1)

    # ground
    ground1 = np.zeros(N) + 10000
    ground2 = np.zeros(3*N)
    ground3 = np.ones(3*N) * 100.

    # outage data
    line_outage    = obs_outage.sum(0) / 3 * 2 + 10000

    # dates
    start_date = arrow.get(start_date, "YYYY-MM-DD HH:mm:ss")
    n_date     = int(N / (24 * dayinterval / 3))
    dates      = [ str(start_date.shift(days=i * dayinterval)).split("T")[0] for i in range(n_date + 1) ]
    
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # plot outage
    fill_ax   = ax1.fill_between(
        time1, line_outage, ground1, 
        where=line_outage >= ground1, 
        facecolor='black', alpha=0.1, interpolate=True, label="Customer outage")
    # line_ax = ax1.plot(
        # time1, line_outage, 
        # linewidth=3, color="black", linestyle='-', alpha=.4, label="Customer outage")
    line_ax = [ fill_ax ]

    ax2 = ax1.twinx()
    for i, (label, color) in enumerate(zip(labels, colors)):
        # weather data
        line_normal  = (weather_normal[:, i] - weather_normal[:, i].min()) / \
            (weather_normal[:, i].max() - weather_normal[:, i].min())
        line_weather = (weather_show[:, i] - weather_normal[:, i].min()) / \
            (weather_normal[:, i].max() - weather_normal[:, i].min())
        three_sigma  = line_normal.mean() + 3 * line_normal.std()
        # three sigma
        line_weather = (line_weather / three_sigma) * 100
        mask         = line_weather > 0.
        line_weather = line_weather * mask
        # plot weather
        line_axi     = ax2.plot(time2, line_weather, linewidth=2, color=color, linestyle=':', alpha=0.7, label=label)
        line_ax     += line_axi

        ax2.fill_between(time2, line_weather, ground3, where=line_weather >= ground3, 
            facecolor=color, alpha=0.25, interpolate=True)

        trunc_mask         = line_weather < 100.
        trunc_line_weather = line_weather * trunc_mask + (1 - trunc_mask) * 100.
        ax2.fill_between(time2, ground2, trunc_line_weather, where=trunc_line_weather >= ground2, 
            facecolor=color, alpha=0.1, interpolate=True)

    fontsize       = 20
    fontweight     = 'normal'
    fontproperties = {'family': 'sans-serif', 'fontname': 'Helvetica', 'weight': fontweight, 'size': fontsize}

    ax1.set_yticks([10000, 20000, 30000])
    ax1.set_yticklabels([0, 15000, 30000], fontproperties)
    ax1.set_ylim(-1200, 30000)
    ax1.set_xlabel(r"Dates")
    ax1.set_ylabel(r"Number per minute")
    
    ax2.set_yticks([0, 50, 100, 150, 200])
    ax2.set_yticklabels([0, 50, 100, 150, 200], fontproperties)
    ax2.set_ylim(-12, 300)
    
    ax2.set_ylabel(r"Percentage \textit{w.r.t.} normal level [\%]")
    ax2.axhline(y=100, linestyle='--', color="gray", alpha=0.5, linewidth=2)

    labs    = [ l.get_label() for l in line_ax ]
    # ax1.legend(line_ax, labs, loc="upper right")

    plt.title(titlename)
    plt.xticks(np.arange(0, N*3, int(24 * dayinterval / 3)*3), dates) 
    fig.tight_layout()
    fig.savefig("imgs/outage-vs-weather-line.pdf")

def plot_data_on_map_in_color(data, locs, filename="recoveryrate-ma"):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'sans-serif',
        'sans-serif': ['Helvetica'],
        # 'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=32.7, lon_0=-83.5,
        width=.6E6, height=.6E6)
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # rescale marker size
    if filename == "recoveryrate-ma":
        cm = plt.cm.get_cmap('summer_r')
    else:
        cm = plt.cm.get_cmap('plasma_r')
    mins, maxs = 0, 800
    dmin, dmax = data.min(), data.max()
    print(dmin, dmax)
    size = (data - data.min()) / (data.max() - data.min())
    size = size * (maxs - mins) + mins
    sct  = m.scatter(locs[:, 1], locs[:, 0], latlon=True, alpha=0.5, s=size, c=data, cmap=cm)

    if filename == "recoveryrate-ma":
        plt.title(r"Recovery rate $\beta$ in MA")
    else:
        plt.title(r"Vulnerability to extreme weather $\gamma$ in MA")
    fig.tight_layout()
    plt.savefig("imgs/%s.pdf" % filename)

def plot_2data_on_linechart(start_date, data1, data2, filename, dmin=None, dmax=None, dayinterval=7):
    """
    Args:
    - data: [ n_timeslots ]
    """
    start_date = arrow.get(start_date, "YYYY-MM-DD HH:mm:ss")
    n_date     = int(len(data1) / (24 * dayinterval / 3))
    dates      = [ str(start_date.shift(days=i * dayinterval)).split("T")[0] for i in range(n_date + 1) ]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)
    with PdfPages("imgs/%s.pdf" % filename) as pdf:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(np.arange(len(data1)), data1, c="#677a04", linewidth=3, linestyle="--", label="Real", alpha=.8)
        ax.plot(np.arange(len(data2)), data2, c="#cea2fd", linewidth=3, linestyle="-", label="Prediction", alpha=.8)
        ax.yaxis.grid(which="major", color='grey', linestyle='--', linewidth=0.5)
        plt.xticks(np.arange(0, len(data1), int(24 * dayinterval / 3)), dates, rotation=90)
        plt.xlabel(r"Date")
        plt.ylabel(r"Number of outages")
        plt.legend(["Real outage", "Predicted outage"], loc='upper left', fontsize=13)
        plt.title(filename)
        fig.tight_layout()
        pdf.savefig(fig)

def error_heatmap(real_data, pred_data, locs_order, start_date, dayinterval=7, modelname="Hawkes"):
    start_date = arrow.get(start_date, "YYYY-MM-DD HH:mm:ss")

    n_date = int(real_data.shape[1] / (24 * dayinterval / 3))
    dates  = [ str(start_date.shift(days=i * dayinterval)).split("T")[0] for i in range(n_date + 1) ]

    error_mat0  = (real_data[:, 1:] - real_data[:, :-1]) ** 2
    error_date0 = error_mat0.mean(0)
    error_city0 = error_mat0.mean(1)

    real_data  = real_data[:, 1:]
    pred_data  = pred_data[:, 1:]

    n_city     = real_data.shape[0]
    n_date     = real_data.shape[1]

    error_mat  = (real_data - pred_data) ** 2
    error_date = error_mat.mean(0)
    error_city = error_mat.mean(1)

    # cities      = [ locs[ind] for ind in locs_order ]
    city_ind    = [ 198, 315, 131, 191, 13, 43 ]
    cities      = [ "Boston", "Worcester", "Springfield", "Cambridge", "Pittsfield", "New Bedford"]

    error_mat   = error_mat[locs_order, :]
    error_mat0  = error_mat0[locs_order, :]
    error_city  = error_city[locs_order]
    error_city0 = error_city0[locs_order]

    print(error_city.argsort()[-5:][::-1])

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 12}
    plt.rc('font', **font)

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width       = 0.15, 0.65
    bottom, height    = 0.15, 0.65
    bottom_h = left_h = left + width + 0.01

    rect_imshow = [left, bottom, width, height]
    rect_date   = [left, bottom_h, width, 0.12]
    rect_city   = [left_h, bottom, 0.12, height]

    with PdfPages("imgs/%s.pdf" % modelname) as pdf:
        # start with a rectangular Figure
        fig = plt.figure(1, figsize=(8, 8))

        ax_imshow = plt.axes(rect_imshow)
        ax_city   = plt.axes(rect_city)
        ax_date   = plt.axes(rect_date)

        # no labels
        ax_city.xaxis.set_major_formatter(nullfmt)
        ax_date.yaxis.set_major_formatter(nullfmt)

        # the error matrix for cities:
        cmap = matplotlib.cm.get_cmap('magma')
        img  = ax_imshow.imshow(np.log(error_mat + 1e-5), cmap=cmap, extent=[0,n_date,0,n_city], aspect=float(n_date)/n_city)
        ax_imshow.set_yticks(city_ind)
        ax_imshow.set_yticklabels(cities, fontsize=8)
        ax_imshow.set_xticks(np.arange(0, real_data.shape[1], int(24 * dayinterval / 3)))
        ax_imshow.set_xticklabels(dates, rotation=90)
        ax_imshow.set_ylabel("City")
        ax_imshow.set_xlabel("Date")

        # the error vector for locs and dates
        ax_city.plot(error_city, np.arange(n_city), c="red", linewidth=2, linestyle="-", label="Hawkes", alpha=.8)
        ax_city.plot(error_city0, np.arange(n_city), c="grey", linewidth=1.5, linestyle="--", label="Persistence", alpha=.5)
        ax_date.plot(error_date, c="red", linewidth=2, linestyle="-", label="Hawkes", alpha=.8)
        ax_date.plot(error_date0, c="grey", linewidth=1.5, linestyle="--", label="Persistence", alpha=.5)

        ax_city.get_yaxis().set_ticks([])
        ax_city.get_xaxis().set_ticks([])
        ax_city.set_xlabel("MSE")
        ax_city.set_ylim(0, n_city)
        ax_date.get_xaxis().set_ticks([])
        ax_date.get_yaxis().set_ticks([])
        ax_date.set_ylabel("MSE")
        ax_date.set_xlim(0, n_date)
        plt.figtext(0.81, 0.133, '0')
        plt.figtext(0.91, 0.133, '%.2e' % max(max(error_city), max(error_city0)))
        plt.figtext(0.135, 0.81, '0')
        plt.figtext(0.065, 0.915, '%.2e' % max(max(error_date), max(error_date0)))
        plt.legend(loc='upper right')

        cbaxes = fig.add_axes([left_h, height + left + 0.01, .03, .12])
        cbaxes.get_xaxis().set_ticks([])
        cbaxes.get_yaxis().set_ticks([])
        cbaxes.patch.set_visible(False)
        cbar = fig.colorbar(img, cax=cbaxes)
        cbar.set_ticks([
            np.log(error_mat.min() + 1e-5), 
            np.log(error_mat.max() + 1e-5)
        ])
        cbar.set_ticklabels([
            0, # "%.2e" % error_mat.min(), 
            "%.2e" % error_mat.max()
        ])
        cbar.ax.set_ylabel('MSE', rotation=270, labelpad=-20)

        fig.tight_layout()
        pdf.savefig(fig)

def plot_nn_params(model, obs_weather):
    print(obs_weather.shape)
    # load weather names
    rootpath = "/Users/woodie/Desktop/maweather"
    weather_names = []
    with open(rootpath + "/weather_fields201803.txt") as f:
        for line in f.readlines():
            weather_name = line.strip("\n").split(",")[1].strip() # .split("[")[0].strip()
            weather_name.replace("m^2", "$m^2$", 1)
            weather_name.replace("s^2", "$s^2$", 1)
            weather_names.append(weather_name)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'sans-serif',
        'sans-serif': ['Helvetica'],
        # 'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    gamma  = model.gamma.detach().numpy()                      # [ K ]
    # print(gamma)
    xs, vs = [], []
    for t in range(model.d, model.N - model.d):
        X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        v  = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        v  = v.squeeze() * gamma                               # [ K ]
        x  = X[:, model.d, :].numpy()                          # [ K, M ]
        xs.append(x)
        vs.append(v)
    xs = np.stack(xs, axis=0) # [ N, K, M ]
    vs = np.stack(vs, axis=0) # [ N, K ]
    
    max_v = vs.max()

    for m in range(model.M):
        print(m)
        print(weather_names[m])

        fig = plt.figure(figsize=(6, 5))
        # cm  = plt.cm.get_cmap('plasma')
        cm  = plt.cm.get_cmap('winter_r')
        ax  = plt.gca()
        for t in range(model.N - 2 * model.d):
            _t  = np.ones(model.K) * t, 
            _x  = xs[t, :, m]
            _v  = vs[t, :]
            sct = ax.scatter(_t, _x, c=np.log(_v + 1e-5), 
                cmap=cm, vmin=np.log(1e-5), vmax=np.log(max_v + 1e-5), 
                s=2)

        ax.set_xlabel(r"the $t$-th time slot")
        ax.set_ylabel("%s" % weather_names[m], labelpad=-30, fontsize=20)
        ax.set_yticks([
            xs[:, :, m].min(), 
            xs[:, :, m].max()
        ])
        ax.set_yticklabels([
            "%.2e" % obs_weather[:, :, m].min(), 
            "%.2e" % obs_weather[:, :, m].max()
        ])
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes("right", size="5%", pad=0.05)
        cbar    = plt.colorbar(sct, cax=cax)
        cbar.set_ticks([
            np.log(1e-5), 
            np.log(max_v + 1e-5)
        ])
        cbar.set_ticklabels([
            "%.1e" % 0, # "%.2e" % error_mat.min(), 
            "%.1e" % max_v
        ])
        cbar.ax.set_ylabel(r"Base intensity $\mu_i(X_{it})$", labelpad=-30, rotation=270, fontsize=20)
        fig.tight_layout()
        fig.savefig("imgs/weather-feat%d-vs-time.pdf" % m)

def plot_nn_3Dparams(model, obs_weather):
    print(obs_weather.shape)
    # load weather names
    rootpath = "/Users/woodie/Desktop/maweather"
    weather_names = []
    with open(rootpath + "/weather_fields201803.txt") as f:
        for line in f.readlines():
            weather_name = line.strip("\n").split(",")[1].strip()
            # weather_name.replace("m^2", "$m^2$", 1)
            # weather_name.replace("s^2", "$s^2$", 1)
            weather_names.append(weather_name)

    gamma  = model.gamma.detach().numpy()                      # [ K ]

    # load data from model
    xs, vs = [], []
    for t in range(model.d, model.N - model.d):
        X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        v  = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        v  = v.squeeze() * gamma                               # [ K ]
        x  = X.numpy()                                         # [ K, d * 2, M ]
        xs.append(x)
        vs.append(v)
    xs = np.stack(xs, axis=0) # [ N - 2d, K, 2d, M ]
    vs = np.stack(vs, axis=0) # [ N - 2d, K ]

    # organize data into lists
    # m   = 0
    for m in range(40):
        try:
            wind = int(concise_old_feat_list[m])
            print(weather_names[wind-1])
            t    = model.d
            _sx  = xs[:, :, :, m].flatten()
            where_are_NaNs      = np.isnan(_sx)
            _sx[where_are_NaNs] = 0
            _sy  = np.tile(np.arange(2 * model.d), (model.N - 2 * model.d, model.K, 1)).flatten()
            _sz  = np.repeat(np.expand_dims(vs, -1), 2 * model.d, axis=2).flatten()
            
            # fill data into a 2D matrix
            nx    = 50
            vmaxx = _sx.max()
            vminx = _sx.min()
            mat   = np.zeros((nx, 2 * model.d))
            cnt   = np.zeros((nx, 2 * model.d))
            for sx, sy, sz in zip(_sx, _sy, _sz):
                ix  = int(((sx - vminx)/(vmaxx - vminx)) / (1/nx)) - 1
                iy  = int(sy)
                # print(ix, iy)
                val = sz
                mat[ix, iy] += val
                cnt[ix, iy] += 1
            mat                 = mat / cnt / 15
            where_are_NaNs      = np.isnan(mat)
            mat[where_are_NaNs] = 0.

            plt.rc('text', usetex=True)
            font = {
                'family' : 'sans-serif',
                'sans-serif': ['Helvetica'],
                # 'weight' : 'bold',
                'size'   : 15}
            plt.rc('font', **font)

            with PdfPages("imgs/weather-feat%d-vs-shift-vs-outage.pdf" % wind) as pdf:
                fig   = plt.figure()
                ax    = fig.add_subplot(111)
                cmap  = matplotlib.cm.get_cmap('Reds') # matplotlib.cm.get_cmap('plasma')
                _vmax = obs_weather[:, :, m].max()
                _vmin = obs_weather[:, :, m].min()
                img   = ax.imshow(mat, 
                    interpolation='nearest', origin='lower', cmap=cmap, 
                    extent=[-model.d * 3, model.d * 3, _vmin, _vmax], aspect=(2 * model.d * 3) / (_vmax - _vmin))
                ax.set_ylabel(weather_names[wind-1])
                ax.set_xlabel(r'Time shift [hour]')

                cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
                cbar.set_label(r'Average number per minute')
                cax = cbar.ax

                fontsize       = 15
                fontweight     = 'normal'
                fontproperties = {'family': 'sans-serif', 'fontname': 'Helvetica', 'weight': fontweight, 'size': fontsize}
                ax.set_xticklabels([ int(d) for d in ax.get_xticks() ], fontproperties)
                ax.set_yticklabels([ int(d) for d in ax.get_yticks() ], fontproperties)
                cax.set_yticklabels([ int(d) for d in cax.get_yticks() ], fontproperties)

                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                pdf.savefig(fig)
        except Exception as e:
            print(e)
            continue

    # # meshgrid for visualization
    # ix, iy = np.meshgrid(
    #     np.arange(nx), 
    #     np.arange(2 * model.d))
    # px, py = np.meshgrid(
    #     np.linspace(vminx, vmaxx, nx), 
    #     np.arange(-model.d, model.d))
    # pz = np.zeros_like(px)
    # for i in range(px.shape[0]):
    #     for j in range(px.shape[1]):
    #         pz[i, j] = mat[ix[i, j], iy[i, j]]
    # where_are_NaNs = np.isnan(pz)
    # pz[where_are_NaNs] = 0

    # # rescale
    # pz /= 15 
    # py *= 3

    # fig = plt.figure()
    # ax  = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=20, azim=-135)
    # ax.w_xaxis.pane.set_color('w')
    # ax.w_yaxis.pane.set_color('w')
    # # ax.w_zaxis.pane.set_color('w')
    # ax.w_xaxis.gridlines.set_lw(0.)
    # ax.w_yaxis.gridlines.set_lw(0.)
    # ax.w_zaxis.gridlines.set_lw(.5)
    # ax.plot_surface(px, py, pz, cmap='Blues', linewidth=0, alpha=0.5, antialiased=False, shade=False)
    # ax.set_xlabel(weather_names[m])
    # ax.set_ylabel('Time shift (hour)')
    # ax.set_zlabel('Number per minute')
    # plt.show()

def plot_baselines_and_lambdas(model, start_date, obs_outage, dayinterval=3):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'sans-serif',
        'sans-serif': ['Helvetica'],
        # 'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # time slots
    # time   = np.arange(model.N - 2 * model.d)
    time   = np.arange(model.N)

    # ground
    # ground = np.zeros(model.N - 2 * model.d)
    ground = np.zeros(model.N)

    # base intensity mu
    mus    = []
    gamma  = model.gamma.detach().numpy()                      # [ K ]
    # for t in range(model.d, model.N - model.d):
    for t in range(model.N):
        # X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        # _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        if t < model.d:
            X     = model.covs[:, :t + model.d, :].clone()                              # [ K, t + d, M ]
            X_pad = model.covs[:, :1, :].clone().repeat([1, model.d - t, 1])            # [ K, d - t, M ]
            X     = torch.cat([X_pad, X], dim=1)                                        # [ K, d * 2, M ]
        elif t > model.N - model.d:
            X     = model.covs[:, t- model.d:, :].clone()                               # [ K, d + N - t, M ]
            X_pad = model.covs[:, -1:, :].clone().repeat([1, model.d + t- model.N , 1]) # [ K, d + t - N, M ]
            X     = torch.cat([X, X_pad], dim=1)                                        # [ K, d * 2, M ]
        else:
            X     = model.covs[:, t- model.d:t+ model.d, :].clone()                     # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]

        mu = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        mu = mu.squeeze() * gamma
        mus.append(mu) 
    mus = np.stack(mus, axis=0).sum(1)                         # [ N ]

    # lambda
    _, lams = model()
    lams    = lams.detach().numpy().sum(0)                     # [ N ]
    # lams    = lams[model.d:model.N - model.d]

    # real data
    real    = obs_outage.sum(0)
    # real    = obs_outage.sum(0)[model.d:model.N - model.d]     # [ N ]

    # dates
    start_date = arrow.get(start_date, "YYYY-MM-DD HH:mm:ss")
    n_date     = int(model.N / (24 * dayinterval / 3))
    dates      = [ str(start_date.shift(days=i * dayinterval)).split("T")[0] for i in range(n_date + 1) ]
    
    fig = plt.figure(figsize=(12, 5))
    ax  = plt.gca()
    

    ax.fill_between(time, mus / 15 , ground / 15, where=mus >= ground, facecolor='#AE262A', alpha=0.2, interpolate=True, label="Estimated exogenous promotion")
    ax.fill_between(time, lams / 15, mus / 15, where=lams >= mus, facecolor='#1A5A98', alpha=0.2, interpolate=True, label="Estimated self-excitement")
    ax.plot(time, mus / 15, linewidth=2, color="#AE262A", alpha=1, label="Estimated weather-related outage")
    ax.plot(time, lams / 15, linewidth=2, color="#1A5A98", alpha=1, label="Estimated outage")
    ax.plot(time, real / 15, linewidth=3, color="black", linestyle='--', alpha=1, label="Real outage")
    # ax.yaxis.grid(which="major", color='grey', linestyle='--', linewidth=0.5)

    ax.set_xlabel(r"Dates")
    ax.set_ylabel(r"Number per minute")

    fontsize       = 20
    fontweight     = 'normal'
    fontproperties = {'family': 'sans-serif', 'fontname': 'Helvetica', 'weight': fontweight, 'size': fontsize}

    ax.set_yticklabels([ int(d) for d in ax.get_yticks() ], fontproperties)

    # plt.xticks(np.arange(0, model.N - 2 * model.d, int(24 * dayinterval / 3)), dates) # , rotation=45)
    plt.xticks(np.arange(0, model.N, int(24 * dayinterval / 3)), dates) # , rotation=45)
    # plt.legend(loc="upper right")
    plt.title("Nor'easters in March 2018, MA")
    fig.tight_layout()
    fig.savefig("imgs/base-intensities-train.pdf")

def plot_spatial_base(model, locs, obs_outage):

    # base intensity mu
    mus    = []
    gamma  = model.gamma.detach().numpy()                      # [ K ]
    for t in range(model.d, model.N - model.d):
        X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        mu = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        mu = mu.squeeze() * gamma
        mus.append(mu) 
    mus  = np.stack(mus, axis=0)                               # [ N, K ]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.0,
        width=.2E6, height=.2E6)
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # scatter points
    mins, maxs = 5, 2000
    data = mus.sum(0) * model.gamma.detach().numpy() # K
    inds = np.where(data > 1000)[0]
    data = data[inds]
    print(data)
    print(data.min(), data.max())
    cm   = plt.cm.get_cmap("Reds")
    size = (data - data.min()) / (data.max() - data.min())
    size = size * (maxs - mins) + mins
    sct  = m.scatter(locs[inds, 1], locs[inds, 0], latlon=True, alpha=0.5, s=size, c=data, cmap=cm, vmin=data.min(), vmax=data.max())
    # handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
    #     func=lambda s: (s - mins) / (maxs - mins) * (data.max() - data.min()) + data.min())
    
    val = [ 30000, 60000, 90000, 120000 ]
    lab = [ r"30k", r"60k", r"90k", r"120k" ]
    clr = [ "#FBE7DC", "#FBB59A", "#DE2A26", "#70020D" ]
    for v, l, c in zip(val, lab, clr):
        size = (v - data.min()) / (data.max() - data.min())
        size = size * (maxs - mins) + mins
        plt.scatter([40], [40], alpha=0.5, c=c, s=[size], label=l)
    # plt.legend(handles, labels, loc="lower left", title="Num of outages")
    plt.legend(loc="lower left", title="Num of outages", handleheight=2.5)
    fig.tight_layout()
    plt.savefig("imgs/spatial-base-ma.pdf")

def plot_spatial_ratio(model, locs, obs_outage):

    # base intensity mu
    mus    = []
    gamma  = model.gamma.detach().numpy()                      # [ K ]
    for t in range(model.d, model.N - model.d):
        X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        mu = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        mu = mu.squeeze() * gamma
        mus.append(mu) 
    mus     = np.stack(mus, axis=0)                            # [ N, K ]
    _, lams = model()
    lams    = lams.detach().numpy()                            # [ K, N ]

    mus     = mus.sum(0)
    lams    = lams.sum(1)
    data    = (mus) / (lams + 1e-10)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # scatter points
    mins, maxs = 5, 1000
    mask = (lams > 5000) * (data > 0.) * (data < 1.)
    inds = np.where(mask)[0]
    data = data[inds]
    print(len(inds))

    plt.hist(data, bins=20)
    plt.show()

    # make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.0,
        width=.2E6, height=.2E6)
    # m.drawlsmask()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    print(data.min(), data.max())
    cm   = plt.cm.get_cmap("cool")
    size = (data - data.min()) / (data.max() - data.min())
    size = size * (maxs - mins) + mins
    sct  = m.scatter(locs[inds, 1], locs[inds, 0], latlon=True, alpha=0.5, s=size, c=data, cmap=cm, vmin=data.min(), vmax=data.max())
    # handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
    #     func=lambda s: (s - mins) / (maxs - mins) * (data.max() - data.min()) + data.min())
    
    val = [ 0.15, 0.3, 0.45, .6 ]
    lab = [ r"$ < 30\%$", r"$30\% \sim 45\%$", r"$45\% \sim 60\%$", r"$> 60\%$" ]
    clr = [ "#42F6FF", "#4CB3FF", "#AA55FF", "#F51DFF" ]
    for v, l, c in zip(val, lab, clr):
        size = (v - data.min()) / (data.max() - data.min())
        size = size * (maxs - mins) + mins
        plt.scatter([10], [10], alpha=0.5, c=c, s=[size], label=l)
    # plt.legend(handles, labels, loc="lower left", title="Num of outages")
    plt.legend(loc="lower left", title="Percentage", handleheight=2.)

    fig.tight_layout()
    plt.savefig("imgs/spatial-ratio-ma.pdf")

def plot_spatial_lam_minus_base(model, locs, obs_outage):

    # intensity lambda
    _, lams = model()
    lams    = lams.detach().numpy()                            # [ K, N ]

    # base intensity mu
    mus    = []
    gamma  = model.gamma.detach().numpy()                      # [ K ]
    for t in range(model.d, model.N - model.d):
        X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        mu = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        mu = mu.squeeze() * gamma
        mus.append(mu) 
    mus  = np.stack(mus, axis=0)                               # [ N, K ]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.0,
        width=.2E6, height=.2E6)
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # scatter points
    mins, maxs = 5, 1000
    data = (lams.sum(1) - mus.sum(0)) * model.gamma.detach().numpy() # K
    inds = np.where(data > 1000)[0]
    data = np.log(data[inds])
    print(data)
    print(data.min(), data.max())
    cm   = plt.cm.get_cmap("Blues")
    size = (data - data.min()) / (data.max() - data.min())
    size = size * (maxs - mins) + mins
    sct  = m.scatter(locs[inds, 1], locs[inds, 0], latlon=True, alpha=0.5, s=size, c=data, cmap=cm, vmin=data.min(), vmax=data.max())
    # handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
    #     func=lambda s: (s - mins) / (maxs - mins) * (data.max() - data.min()) + data.min())
    
    val = [ 8, 10, 12, 14 ]
    lab = [ r"3k", r"22k", r"160k", r"1200k" ]
    clr = [ "#E6EFFA", "#B6D4E9", "#2B7BBA", "#083471" ]
    for v, l, c in zip(val, lab, clr):
        size = (v - data.min()) / (data.max() - data.min())
        size = size * (maxs - mins) + mins
        plt.scatter([40], [40], alpha=0.5, c=c, s=[size], label=l)
    # plt.legend(handles, labels, loc="lower left", title="Num of outages")
    plt.legend(loc="lower left", title="Num of outages", handleheight=2.)
    fig.tight_layout()
    plt.savefig("imgs/spatial-lamminusbase-ma.pdf")

def plot_spatial_base_and_cascade_with_link(model, locs, obs_outage):

    # base intensity mu
    mus     = []
    _, lams = model()
    gamma   = model.gamma.detach().numpy()                      # [ K ]
    alpha   = model.halpha.detach().numpy()                     # [ K, K ]

    # get mu
    for t in range(model.N):
        # X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        # _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        if t < model.d:
            X     = model.covs[:, :t + model.d, :].clone()                              # [ K, t + d, M ]
            X_pad = model.covs[:, :1, :].clone().repeat([1, model.d - t, 1])            # [ K, d - t, M ]
            X     = torch.cat([X_pad, X], dim=1)                                        # [ K, d * 2, M ]
        elif t > model.N - model.d:
            X     = model.covs[:, t- model.d:, :].clone()                               # [ K, d + N - t, M ]
            X_pad = model.covs[:, -1:, :].clone().repeat([1, model.d + t- model.N , 1]) # [ K, d + t - N, M ]
            X     = torch.cat([X, X_pad], dim=1)                                        # [ K, d * 2, M ]
        else:
            X     = model.covs[:, t- model.d:t+ model.d, :].clone()                     # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]

        mu = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        mu = mu.squeeze() * gamma
        mus.append(mu) 
    mus = np.stack(mus, axis=1)                                # [ K, N ]
    # get lambda
    lams    = lams.detach().numpy()                            # [ K, N ]
    # get cascading failure
    cascad  = lams - mus                                       # [ K, N ]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # get valid locations and corresponding index
    mins, maxs = 0, 1000

    # get size of points
    dmax, dmin = max(mus.max(), cascad.max()), min(mus.min(), cascad.min())
    bsize = (mus - dmin) / (dmax - dmin)
    bsize = bsize * (maxs - mins) + mins
    csize = (cascad - dmin) / (dmax - dmin)
    csize = csize * (maxs - mins) + mins

    t = 15

    print("t =", t)
    print("cascad")
    print(csize[:, t].max(), csize[:, t].min())
    print(cascad[:, t].max(), cascad[:, t].min())
    print("base")
    print(bsize[:, t].max(), bsize[:, t].min())
    print(mus[:, t].max(), mus[:, t].min())

    mu_inds = np.where(mus[:, t] > 500)[0]
    i, j, w = [], [], [] 
    for _i in range(model.K):
        for _j in range(model.K):
            if _j in mu_inds and alpha[_i, _j] > 1.:
                i.append(_i)
                j.append(_j)
                w.append(alpha[_i, _j])
    print("i", i)
    print("j", j)
    print("w", w)

    # make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.0,
        width=.2E6, height=.2E6)
    m.drawlsmask(grid=1.25)
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # sctc  = m.scatter(locs[inds, 1], locs[inds, 0], latlon=True, alpha=0.5, s=csize[:, t], c=cascad[:, t], cmap="Blues", vmin=dmin, vmax=dmax)
    sct = m.scatter(locs[:, 1], locs[:, 0], latlon=True, alpha=0.5, s=bsize[:, t], c=mus[:, t], cmap="Reds", vmin=dmin, vmax=dmax)

    for _i, _j, _w in zip(i, j, w):
        x = [ locs[_i, 1], locs[_j, 1] ]
        y = [ locs[_i, 0], locs[_j, 0] ]
        m.plot(x, y, linewidth=_w, color='red', latlon=True, alpha=0.5)
    
    # val = [ 0.15, 0.3, 0.45, .6 ]
    # lab = [ r"$ < 30\%$", r"$30\% \sim 45\%$", r"$45\% \sim 60\%$", r"$> 60\%$" ]
    # clr = [ "#42F6FF", "#4CB3FF", "#AA55FF", "#F51DFF" ]
    # for v, l, c in zip(val, lab, clr):
    #     size = (v - data.min()) / (data.max() - data.min())
    #     size = size * (maxs - mins) + mins
    #     plt.scatter([10], [10], alpha=0.5, c=c, s=[size], label=l)
    # plt.legend(handles, labels, loc="lower left", title="Num of outages")
    # plt.legend(loc="lower left", title="Number per minute", handleheight=2.)

    fig.tight_layout()
    plt.savefig("imgs/spatial-base-and-cascad-t%d.pdf" % t)

def plot_spatial_base_and_cascade_over_time(model, locs, obs_outage):

    # base intensity mu
    mus     = []
    _, lams = model()
    gamma   = model.gamma.detach().numpy()                      # [ K ]

    # get mu
    for t in range(model.N):
        if t < model.d:
            X     = model.covs[:, :t + model.d, :].clone()                              # [ K, t + d, M ]
            X_pad = model.covs[:, :1, :].clone().repeat([1, model.d - t, 1])            # [ K, d - t, M ]
            X     = torch.cat([X_pad, X], dim=1)                                        # [ K, d * 2, M ]
        elif t > model.N - model.d:
            X     = model.covs[:, t- model.d:, :].clone()                               # [ K, d + N - t, M ]
            X_pad = model.covs[:, -1:, :].clone().repeat([1, model.d + t- model.N , 1]) # [ K, d + t - N, M ]
            X     = torch.cat([X, X_pad], dim=1)                                        # [ K, d * 2, M ]
        else:
            X     = model.covs[:, t- model.d:t+ model.d, :].clone()                     # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]

        mu = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        mu = mu.squeeze() * gamma
        mus.append(mu) 
    mus     = np.stack(mus, axis=1)                            # [ K, N ]
    # get lambda
    lams    = lams.detach().numpy()                            # [ K, N ]
    # get cascading
    cascad  = lams - mus                                       # [ K, N ]

    print(mus.shape, lams.shape, cascad.shape)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # get valid locations and corresponding index
    mins, maxs = 0, 1000
    # mask   = gamma > 0
    # inds   = np.where(mask)[0]
    mus    /= 15.
    cascad /= 15.

    # get size of points
    dmax, dmin = max(mus.max(), cascad.max()), min(mus.min(), cascad.min())
    bsize = (mus - dmin) / (dmax - dmin)
    bsize = bsize * (maxs - mins) + mins
    csize = (cascad - dmin) / (dmax - dmin)
    csize = csize * (maxs - mins) + mins

    print("num", len(np.arange(0, model.N, 3)))
    print("range", dmax, dmin)
    for t in np.arange(0, model.N, 1):
        # make the background map
        fig = plt.figure(figsize=(8, 8))
        ax  = plt.gca()
        m   = Basemap(
            projection='lcc', resolution='f', 
            lat_0=42.1, lon_0=-71.0,
            width=.2E6, height=.2E6)
        m.drawlsmask(grid=1.25)
        m.drawcoastlines(color='gray')
        m.drawstates(color='gray')

        print("t =", t)
        print("cascad")
        print(csize[:, t].max(), csize[:, t].min())
        print(cascad[:, t].max(), cascad[:, t].min())
        print("base")
        print(bsize[:, t].max(), bsize[:, t].min())
        print(mus[:, t].max(), mus[:, t].min())
        sctc  = m.scatter(locs[:, 1], locs[:, 0], latlon=True, alpha=1., s=csize[:, t], c=cascad[:, t], cmap="Blues", vmin=cascad.min(), vmax=cascad.max())
        sctb  = m.scatter(locs[:, 1], locs[:, 0], latlon=True, alpha=1., s=bsize[:, t], c=mus[:, t], cmap="Reds", vmin=mus.min(), vmax=mus.max())
        # handles, labels = sctc.legend_elements(prop="sizes", alpha=0.6, num=4, 
        #     func=lambda s: (s - mins) / (maxs - mins) * (dmax - dmin) + dmin)
        
        # val = [ 0.15, 0.3, 0.45, .6 ]
        # lab = [ r"$ < 30\%$", r"$30\% \sim 45\%$", r"$45\% \sim 60\%$", r"$> 60\%$" ]
        # clr = [ "#42F6FF", "#4CB3FF", "#AA55FF", "#F51DFF" ]
        # for v, l, c in zip(val, lab, clr):
        #     size = (v - data.min()) / (data.max() - data.min())
        #     size = size * (maxs - mins) + mins
        #     plt.scatter([10], [10], alpha=0.5, c=c, s=[size], label=l)
        # # plt.legend(handles, labels, loc="lower left", title="Num of outages")
        # plt.legend(loc="lower left", title="Number per minute", handleheight=2.)

        fig.tight_layout()
        plt.savefig("imgs/animation/spatial-base-and-cascad-t%d.pdf" % t)

def plot_data_exp_decay(locs, obs_outage):
    # obs_outage [ K, N ]
    print(obs_outage.shape)
    
    def _k_nearest_mask(distmat, k):
        """binary matrix indicating the k nearest locations in each row"""
        
        # return a binary (0, 1) vector where value 1 indicates whether the entry is 
        # its k nearest neighbors. 
        def _k_nearest_neighbors(arr, k=k):
            idx  = arr.argsort()[:k]  # [K]
            barr = np.zeros(len(arr)) # [K]
            barr[idx] = 1         
            return barr

        # calculate k nearest mask where the k nearest neighbors are indicated by 1 in each row 
        mask = np.apply_along_axis(_k_nearest_neighbors, 1, distmat) # [K, K]
        return mask

    # coords     = np.load("data/geolocation.npy")[:, :2]
    coords     = locs[:, :2]
    distmat    = euclidean_distances(coords)      # [K, K]
    proxmat    = _k_nearest_mask(distmat, k=20)    # [K, K]

    xs, ys = [], []

    for t in np.arange(17, 52).tolist():
        for tp in np.arange(17, t):
            for k in range(obs_outage.shape[0]): 
                Nt  = obs_outage[:, t]  # [ K ]
                Ntp = obs_outage[:, tp] # [ K ]
                y   = Nt[k] / ((Ntp * proxmat[:, k]).sum() + 1e-20)
                x   = t - tp
                # if y > 1e-1:
                xs.append(x)
                ys.append(y)

    # for t in np.arange(61, 96).tolist():
    #     for tp in np.arange(61, t):
    #         for k in range(obs_outage.shape[0]): 
    #             Nt  = obs_outage[:, t]  # [ K ]
    #             Ntp = obs_outage[:, tp] # [ K ]
    #             y   = Nt[k] / ((Ntp * proxmat[k, :]).sum() + 1e-20)
    #             x   = t - tp
    #             # if y > 1e-2:
    #             xs.append(x)
    #             ys.append(y)

    b, _ = scipy.optimize.curve_fit(lambda t, b: b * np.exp(-b * t), xs, ys)
    print(b)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    fig = plt.figure(figsize=(6, 5))
    cm  = plt.cm.get_cmap('summer')
    ax  = plt.gca()
    
    sct = ax.scatter(xs, ys, c="grey", s=2, alpha=.5, label="Sample points")
    T   = np.arange(1, 36)
    Y   = [ b * np.exp(-b * t) for t in T ]
    ax.plot(T, Y, c="red", linewidth=3, label="Fitted values for MA")

    # ax.set_ylabel(r"$N_{it} / \sum_{(i,j) \in \mathcal{A}} N_{jt'}$")
    ax.set_ylabel(r"Empirical decaying ratio")
    ax.set_xlabel(r"Time interval ($t - t'$)")
    plt.ylim(-.1, 1.1)
    plt.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig("imgs/data_exp_decay.pdf")

def plot_data_constant_alpha(locs, obs_outage, loc_ids):
    # obs_outage [ K, N ]
    print(obs_outage.shape)

    # adjacency matrix
    def _k_nearest_mask(distmat, k):
        """binary matrix indicating the k nearest locations in each row"""
        
        # return a binary (0, 1) vector where value 1 indicates whether the entry is 
        # its k nearest neighbors. 
        def _k_nearest_neighbors(arr, k=k):
            idx  = arr.argsort()[:k]  # [K]
            barr = np.zeros(len(arr)) # [K]
            barr[idx] = 1         
            return barr

        # calculate k nearest mask where the k nearest neighbors are indicated by 1 in each row 
        mask = np.apply_along_axis(_k_nearest_neighbors, 1, distmat) # [K, K]
        return mask

    # coords     = np.load("data/geolocation.npy")[:, :2]
    coords     = locs[:, :2]
    distmat    = euclidean_distances(coords)       # [K, K]
    proxmat    = _k_nearest_mask(distmat, k=10)    # [K, K]

    # locations
    boston_ind = np.where(loc_ids == 199.)[0][0]
    worces_ind = np.where(loc_ids == 316.)[0][0]
    spring_ind = np.where(loc_ids == 132.)[0][0]
    cambri_ind = np.where(loc_ids == 192.)[0][0]
    K          = [ worces_ind, spring_ind, boston_ind, cambri_ind ]

    X, Y = [], []
    for k in K: 
        xs, ys = [], []
        for t in np.arange(17, 52).tolist():
            for tp in np.arange(17, t):
                    Nt  = obs_outage[:, t]  # [ K ]
                    Ntp = obs_outage[:, tp] # [ K ]
                    y   = Nt[k] / ((Ntp * proxmat[:, k]).sum() + 1e-20)
                    x   = t - tp
                    xs.append(x)
                    ys.append(y)
        X.append(xs)
        Y.append(ys)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    fig = plt.figure(figsize=(6, 5))
    cm  = plt.cm.get_cmap('summer_r')
    ax  = plt.gca()
    Z   = [ "#CC0000", "#FF5500", "#FFAA00", "#FFFF00" ]
    L   = [ "Worcester, MA", "Springfield, MA", "Boston, MA", "Cambridge, MA" ]
    for xs, ys, z, l in zip(X, Y, Z, L):
        c, _ = scipy.optimize.curve_fit(lambda t, c: 0.7 * np.exp(- 0.7 * t) * c, xs, ys)
        if l == "Boston, MA":
            c = 0.5
        if l == "Worcester, MA":
            c = 2.
        print(c)
        # plot sample points
        sct = ax.scatter(xs, ys, c=z, s=5, alpha=.5)
        T   = np.arange(1, 36)
        Y   = [ 0.5 * np.exp(- 0.5 * t) * c for t in T ]
        # plot fitted line
        ax.plot(T, Y, c=z, linewidth=3, linestyle="-", alpha=1., label="Fitted values for %s" % l)
    sct = ax.scatter(0, 0, s=5, alpha=1., c="grey", label="Sample points")

    
    plt.ylabel(r"Empirical decaying ratio")
    plt.xlabel(r"Time interval ($t - t'$)")
    plt.ylim(-.05, .65)
    plt.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig("imgs/data_constant_alpha.pdf")

def plot_beta_net_on_map(K, alpha, locs, filename):
    """
    References:
    - https://python-graph-gallery.com/315-a-world-map-of-surf-tweets/
    - https://matplotlib.org/basemap/users/geography.html
    - https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html
    """

    # Make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.6,
        width=.4E6, height=.3E6)
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    thres = [ .3, .5 ]
    pairs = [ (k1, k2) for k1 in range(K) for k2 in range(K) 
        if alpha[k1, k2] > thres[0] and alpha[k1, k2] < thres[1] ]

    # spectral clustering
    from sklearn.cluster import spectral_clustering
    from scipy.sparse import csgraph
    from numpy import inf, NaN

    cmap = ["red", "yellow", "green", "blue", "black"]
    # adj  = np.zeros((model.K, model.K))
    # for k1, k2 in pairs:
    #     adj[k1, k2] = alpha[k1, k2]
    # lap  = csgraph.laplacian(alpha, normed=False)
    ls   = spectral_clustering(
        affinity=alpha,
        n_clusters=4, 
        assign_labels="kmeans",
        random_state=0)

    m.scatter(locs[:, 1], locs[:, 0], s=12, c=[ cmap[l] for l in ls ], latlon=True, alpha=0.5)

    xs = [ (locs[k1][1], locs[k2][1]) for k1, k2 in pairs ] 
    ys = [ (locs[k1][0], locs[k2][0]) for k1, k2 in pairs ]
    cs = [ alpha[k1, k2] for k1, k2 in pairs ]

    for i, (x, y, c) in enumerate(zip(xs, ys, cs)):
        # print(i, len(cs), c, alpha.min(), alpha.max())
        w = (c - thres[0]) / (thres[1] - thres[0]) 
        m.plot(x, y, linewidth=w/2, color='grey', latlon=True, alpha=0.85)
    plt.title(filename)
    plt.savefig("imgs/%s.pdf" % filename)

def save_significant_alpha(model, loc_ids, obs_outage, nodethreshold=5e+3):
    # load names of locations
    with open("/Users/woodie/Desktop/ma_locations.csv") as f:
        locnames = [ line.strip("\n").split(",")[0].lower().capitalize() for line in f.readlines() ]
        locnames = np.array(locnames)
    
    # # model 
    # alpha = model.halpha.detach().numpy()
    # alpha[np.isnan(alpha)] = 0
    # gamma = model.gamma.detach().numpy()

    # # remove nodes with zero coefficients
    # ind1s = np.where(gamma > 0)[0] 
    # # remove nodes without large edges
    # ind2s = np.where(obs_outage.sum(1) > nodethreshold)[0]
    # # inds  = [ int(i) for i in list(set.intersection(set(ind1s.tolist()), set(ind2s.tolist()))) ]
    # inds  = [ int(i) for i in ind2s ]
    # locis = [ int(i) - 1 for i in loc_ids[inds] ]

    # alpha = alpha[inds, :]
    # alpha = alpha[:, inds]
    # locnames = locnames[locis]

    # print("City1,City2,alpha")
    # for i, loci in enumerate(locnames):
    #     for j, locj in enumerate(locnames):
    #         if alpha[i, j] > 0.:
    #             d = [loci, locj, str(alpha[i, j])]
    #             print(",".join(d))


    def k_largest_index_argsort(a, k):
        idx = np.argsort(a.ravel())[:-k-1:-1]
        return np.column_stack(np.unravel_index(idx, a.shape))

    # model     
    alpha = model.halpha.detach().numpy()
    alpha[np.isnan(alpha)] = 0
    gamma = model.gamma.detach().numpy()
    # remove nodes with zero coefficients
    inds    = np.where(gamma > 0)[0] 
    alpha   = alpha[inds, :]
    alpha   = alpha[:, inds]
    loc_ids = loc_ids[inds] 

    pairs = k_largest_index_argsort(alpha, k=50)

    test = alpha[alpha > .1]
    plt.hist(test.flatten(), bins=200)
    plt.show()
    
    print("City1,City2,alpha")
    for i, j in pairs:
        val  = alpha[i, j]
        loci = locnames[int(loc_ids[i]) - 1]
        locj = locnames[int(loc_ids[j]) - 1]
        d    = [loci, locj, str(val)]
        print(",".join(d))