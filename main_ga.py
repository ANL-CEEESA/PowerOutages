#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt 
from plot_ga import *

from dataloader import load_outage, load_weather, dataloader, config
from hkstorch import TorchHawkesNNCovariates, train
from utils import avg

if __name__ == "__main__":

    torch.manual_seed(1)

    # TRAINING

    # # load data
    # obs_outage, obs_weather, geo_outage, _ = dataloader(
    #     config["GA Oct 2018"], standardization=True, outageN=3, weatherN=3, isproj=False)
    # obs_weather = obs_weather[:665, :, :]
    # print(obs_outage.shape)
    # print(obs_weather.shape)

    # # training
    # model = TorchHawkesNNCovariates(d=6, obs=obs_outage, covariates=obs_weather)
    # train(model, locs=geo_outage, niter=1000, lr=1., log_interval=10)
    # print("[%s] saving model..." % arrow.now())
    # torch.save(model.state_dict(), "saved_models/hawkes_covariates_vecbeta_ga_201810_d6_feat35.pt")
    # print(model.hbeta.detach().numpy())

    # # evaluation
    # _, lams = model()
    # lams    = lams.detach().numpy()

    # # visualization
    # plot_2data_on_linechart(config["GA Oct 2018"]["_startt"], lams.sum(0), obs_outage.sum(0), "Prediction of total outages in GA (Oct 2018)", dayinterval=1)


    # EVALUATION

    # obs_outage, obs_weather, geo_outage, _ = dataloader(
    #     config["GA Oct 2018"], standardization=True, outageN=3, weatherN=3, isproj=False)
    # obs_weather = obs_weather[:665, :, :]
    # loc_ids     = geo_outage[:, 2]
    # locs        = geo_outage[:, :2]

    # model = TorchHawkesNNCovariates(d=6, obs=obs_outage, covariates=obs_weather)
    # # model.load_state_dict(torch.load("saved_models/hawkes_covariates_varbeta_ga_201810full_d6_feat35.pt"))
    # model.load_state_dict(torch.load("saved_models/hawkes_covariates_vecbeta_ga_201810_d6_feat35.pt"))

    # ---------------------------------------------------
    #  Plot base intensity

    # plot_baselines_and_lambdas(model, config["GA Oct 2018"]["_startt"], obs_outage)
    # ---------------------------------------------------

    # ---------------------------------------------------
    #  Plot gamma

    # mask  = obs_outage.sum(1) > 2000.
    # gamma = model.gamma.detach().numpy()
    # gamma = gamma * mask
    # beta  = model.hbeta.detach().numpy()
    # beta  = np.exp(beta) * mask
    # plot_data_on_map_in_color(beta, geo_outage, "recoveryrate-ga")
    # plot_data_on_map_in_color(gamma, geo_outage, "weathervulnerability-ga")
    # ---------------------------------------------------



    # ---------------------------------------------------
    # OUTAGE AND WEATHER VISUALIZATION

    # startt = 100
    # endt   = 400
    # feats  = [6, -4]
    # colors = ["#DC143C", "#0165fc"]
    # labels = ["Derived radar reflectivity", "Wind speed"]
    # obs_outage, geo_outage = load_outage(config["Complete GA Oct 2018"])
    # obs_feats, geo_weather = load_weather(config["Complete GA Oct 2018"])
    # print(obs_outage.shape)
    # print(obs_feats.shape)
    # print(set(np.arange(1, 678)) - set(geo_outage[:, 2]))
    # obs_outage       = obs_outage[:361, :]
    # obs_feats_show   = obs_feats[feats, :361, :]
    # obs_feats_normal = obs_feats[feats, 361:, :]

    # plot_outage_and_weather_linechart(
    #     "Hurricane Michael in October 2018, GA", 
    #     config["GA Oct 2018"]["_startt"], 
    #     obs_outage, obs_feats_show, obs_feats_normal, labels, colors, 
    #     dayinterval=3)
    # ---------------------------------------------------

    # ---------------------------------------------------
    # OUTAGE QQPLOT

    # obs_outage, geo_outage = load_outage(config["Complete GA Oct 2018"], N=1)
    # # plt.plot(obs_outage.sum(1))
    # # plt.show()
    # print(obs_outage.shape)

    # startt  = 130 * 4
    # endt    = 330 * 4
    # _outage = obs_outage[startt:endt, :]
    # start_date = str(arrow.get(config["Complete GA Oct 2018"]["_startt"], "YYYY-MM-DD HH:mm:ss").shift(hours=startt/4).format("YYYY-MM-DD HH:mm:ss"))
    # plot_interval_qqplot(_outage, start_date, "Hurricane Michael in 2018, GA", dayinterval=2, vmin=10, vmax=8000)

    # startt  = 330 * 4
    # endt    = 530 * 4
    # _outage = obs_outage[startt:endt, :]
    # start_date = str(arrow.get(config["Complete GA Oct 2018"]["_startt"], "YYYY-MM-DD HH:mm:ss").shift(hours=startt/4).format("YYYY-MM-DD HH:mm:ss"))
    # plot_interval_qqplot(_outage, start_date, "Daily operation in 2018, GA", dayinterval=2, vmin=10, vmax=8000)
    # # ---------------------------------------------------


    # ---------------------------------------------------
    # Number of Customers

    # load data
    ncust = np.load("data/ncustomer_ga.npy")
    ncust = ncust[:665]
    obs_outage, obs_weather, loc, _ = dataloader(
        config["GA Oct 2018"], standardization=True, outageN=1, weatherN=1, isproj=False)
    obs_weather = obs_weather[:665, :, :]
    max_outage = obs_outage.max(1)

    print(loc)
    atl_coord = np.array([33.7490, -84.3880])
    dists     = loc[:, :2] - atl_coord
    dists     = np.sqrt(dists[:, 0] ** 2 + dists[:, 1] ** 2)
    cm        = plt.cm.get_cmap('Reds')
    plt.scatter(ncust, max_outage, c=dists, cmap=cm)
    plt.show()
    # ---------------------------------------------------


    # ---------------------------------------------------
    # Time Lag

    # # load data
    # ncust = np.load("data/ncustomer_ga.npy")
    # ncust = ncust[:665]
    # obs_outage, obs_weather, _, _ = dataloader(
    #     config["GA Oct 2018"], standardization=True, outageN=1, weatherN=1, isproj=False)
    # obs_weather = obs_weather[:665, :, :]
    # print(obs_outage.shape, obs_weather.shape)

    # # # remove city with few ouate
    # # mask        = obs_outage.max(1) > 100
    # # obs_outage  = obs_outage[mask]
    # # obs_weather = obs_weather[mask, :, :]
    # # ncust       = ncust[mask]

    # # find the time of extrem weather and most of the outage
    # t_outage_peak = obs_outage.argmax(1)
    # t_drr_peak    = obs_weather[:, :, 6].argmax(1)

    # # remove city where outage happen before extreme weather
    # mask          = t_drr_peak < t_outage_peak
    # t_outage_peak = t_outage_peak[mask]
    # t_drr_peak    = t_drr_peak[mask]
    # ncust         = ncust[mask]

    # # sort city by the time when the extreme weather hit
    # order         = t_drr_peak.argsort()
    # t_drr_peak    = t_drr_peak[order]
    # t_outage_peak = t_outage_peak[order]
    # print(ncust.shape, t_outage_peak.shape, t_drr_peak.shape)

    # plt.scatter(t_outage_peak, np.arange(len(ncust)), c="r")
    # plt.scatter(t_drr_peak, np.arange(len(ncust)), c="b")
    # plt.show()
    # ---------------------------------------------------