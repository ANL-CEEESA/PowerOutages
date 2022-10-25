#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt 
from plot_nc import *

from dataloader import load_outage, load_weather, dataloader, config
from hkstorch import TorchHawkesNNCovariates, train

if __name__ == "__main__":

    # training

    # obs_outage, obs_weather, geo_outage, _ = dataloader(config["NCSC Aug 2020"], outageN=3, weatherN=3, isproj=False)

    # # obs_outage  = obs_outage[:, 400:900]
    # # obs_weather = obs_weather[:, 400:900, :]
    # # print(obs_outage.shape)
    # # print(obs_weather.shape)

    # model = TorchHawkesNNCovariates(d=6, obs=obs_outage, covariates=obs_weather)
    # train(model, locs=geo_outage, niter=1000, lr=1., log_interval=10)
    # print("[%s] saving model..." % arrow.now())
    # torch.save(model.state_dict(), "saved_models/hawkes_covariates_vecbeta_ncsc_202008_d6_feat35.pt")
    # print(model.hbeta.detach().numpy())

    # evaluation
    # obs_outage, obs_weather, geo_outage, _ = dataloader(config["NCSC Aug 2020"], outageN=3, weatherN=3, isproj=False)

    # model = TorchHawkesNNCovariates(d=6, obs=obs_outage, covariates=obs_weather)
    # model.load_state_dict(torch.load("saved_models/hawkes_covariates_vecbeta_ncsc_202008_d6_feat35.pt"))

    # _, lams = model()
    # lams    = lams.detach().numpy()

    # visualization
    # plot_baselines_and_lambdas(model, config["NCSC Aug 2020"]["_startt"], obs_outage, dayinterval=2)


    # ---------------------------------------------------
    #  Plot gamma

    # mask  = obs_outage.sum(1) > 500.
    # gamma = model.gamma.detach().numpy()
    # gamma = gamma * mask
    # beta  = model.hbeta.detach().numpy()
    # beta  = np.exp(beta) * mask
    # plot_data_on_map_in_color(beta, geo_outage, "recoveryrate-ncsc")
    # plot_data_on_map_in_color(gamma, geo_outage, "weathervulnerability-ncsc")
    # ---------------------------------------------------



    # ---------------------------------------------------
    # OUTAGE AND WEATHER VISUALIZATION

    # startt = 100
    # endt   = 400
    # feats  = [6, -4]
    # colors = ["#DC143C", "#0165fc"]
    # labels = ["Derived radar reflectivity", "Wind speed"]
    # obs_outage, geo_outage        = load_outage(config["NCSC Aug 2020"])
    # obs_feats_show, geo_weather   = load_weather(config["NCSC Aug 2020"])
    # obs_feats_normal, geo_weather = load_weather(config["NCSC Summer 2020"])

    # # plot_outage_and_weather_linechart(
    # #     "Tropical Storm Arthur in May 2020, NC \& SC", 
    # #     config["NCSC May 2020"]["_startt"], 
    # #     obs_outage, obs_feats_show, obs_feats_normal, labels, colors, 
    # #     dayinterval=3)

    # plot_outage_and_weather_linechart(
    #     "Hurricane Isaias in August 2020, NC \& SC", 
    #     config["NCSC Aug 2020"]["_startt"], 
    #     obs_outage, obs_feats_show, obs_feats_normal, labels, colors, 
    #     dayinterval=2)
    # ---------------------------------------------------



    # ---------------------------------------------------
    # OUTAGE QQPLOT
    # obs_outage, geo_outage = load_outage(config["NCSC Summer 2020"], N=1)
    # print(obs_outage.shape)

    # startt = 9100
    # endt   = 9300
    # _outage = obs_outage[startt:endt, :]
    # start_date = str(arrow.get(config["NCSC Summer 2020"]["_startt"], "YYYY-MM-DD HH:mm:ss").shift(hours=startt/4).format("YYYY-MM-DD HH:mm:ss"))
    # plot_interval_qqplot(_outage, start_date, "Hurricane Isaias in 2020, NC \& SC", dayinterval=1, vmin=10, vmax=12000)

    # startt = 8800
    # endt   = 9000
    # _outage = obs_outage[startt:endt, :]
    # start_date = str(arrow.get(config["NCSC Summer 2020"]["_startt"], "YYYY-MM-DD HH:mm:ss").shift(hours=startt/4).format("YYYY-MM-DD HH:mm:ss"))
    # plot_interval_qqplot(_outage, start_date, "Daily operation in 2020, NC \& SC", dayinterval=1, vmin=10, vmax=12000)
    # ---------------------------------------------------


    # ---------------------------------------------------
    # Number of Customers

    # # gamma = model.gamma.detach().numpy()
    # # beta  = model.hbeta.detach().numpy()
    # # ncust = np.load("data/ncustomer_ncsc.npy")
    # # print(gamma.shape, beta.shape, ncust.shape)
    # # ncust = ncust[:-1]

    # # plt.scatter(ncust, gamma)
    # # plt.show()
    # ---------------------------------------------------


    # ---------------------------------------------------
    # Time Lag

    # # load data
    # ncust = np.load("data/ncustomer_ncsc.npy")
    # ncust = ncust[:-1]
    # obs_outage, obs_weather, _, _ = dataloader(
    #     config["NCSC Aug 2020"], standardization=True, outageN=1, weatherN=1, isproj=False)
    # obs_weather = obs_weather[:-1, :, :]
    # print(obs_outage.shape, obs_weather.shape)

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