#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt 
from plot_ma import *

from dataloader import load_outage, load_weather, dataloader, config
from hkstorch import TorchHawkes, TorchHawkesNNCovariates

if __name__ == "__main__":

    # obs_outage, obs_weather, locs, _ = dataloader(config["MA Mar 2018"])
    # loc_ids = locs[:, 2]

    # model1 = TorchHawkes(obs=obs_outage)
    model2 = TorchHawkesNNCovariates(d=6, obs=obs_outage, covariates=obs_weather)

    # model1.load_state_dict(torch.load("saved_models/hawkes.pt"))
    # model2.load_state_dict(torch.load("saved_models/hawkes_covariates_varbeta_ma_201803full_d6_feat35.pt"))
    model2.load_state_dict(torch.load("saved_models/hawkes_covariates_vecbeta_ma_201803_d6_feat35.pt"))

    # # _, lams1 = model1()
    # # lams1    = lams1.detach().numpy()

    # _, lams2 = model2()
    # lams2    = lams2.detach().numpy()

    # ---------------------------------------------------
    # #  Plot data

    # plot_illustration(locs)
    # plot_data_exp_decay(locs, obs_outage)
    # plot_data_constant_alpha(locs, obs_outage, loc_ids)
    # ---------------------------------------------------
    


    # # ---------------------------------------------------
    # #  Plot temporal predictions

    # boston_ind = np.where(loc_ids == 199.)[0][0]
    # worces_ind = np.where(loc_ids == 316.)[0][0]
    # spring_ind = np.where(loc_ids == 132.)[0][0]
    # cambri_ind = np.where(loc_ids == 192.)[0][0]
    # plot_2data_on_linechart(config["MA Oct 2019"]["_startt"], lams2.sum(0), obs_outage.sum(0), "Prediction of total outages in MA (Oct 2019)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Oct 2019"]["_startt"], lams2[boston_ind], obs_outage[boston_ind], "Prediction for Boston, MA (Oct 2019)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Oct 2019"]["_startt"], lams2[worces_ind], obs_outage[worces_ind], "Prediction for Worcester, MA (Oct 2019)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Oct 2019"]["_startt"], lams2[spring_ind], obs_outage[spring_ind], "Prediction for Springfield, MA (Oct 2019)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Oct 2019"]["_startt"], lams2[cambri_ind], obs_outage[cambri_ind], "Prediction for Cambridge, MA (Oct 2019)", dayinterval=1)
    # # ---------------------------------------------------



    # # ---------------------------------------------------
    # #  Plot error matrix

    # locs_order = np.argsort(loc_ids)
    # error_heatmap(real_data=obs_outage, pred_data=lams2, locs_order=locs_order, start_date=config["MA Mar 2018"]["_startt"], dayinterval=1, modelname="our model feat 43")
    # error_heatmap(real_data=obs_outage, pred_data=lams1, locs_order=locs_order, start_date=start_date, dayinterval=1, modelname="Hawkes without feat")
    # # ---------------------------------------------------



    # ---------------------------------------------------
    #  Plot gamma

    # mask  = obs_outage.sum(1) > 1000.
    # gamma = model.gamma.detach().numpy()
    # gamma = gamma * mask
    # beta  = model.hbeta.detach().numpy()
    # beta  = np.exp(beta) * mask
    # plot_data_on_map_in_color(beta, geo_outage, "recoveryrate-ma")
    # plot_data_on_map_in_color(gamma, geo_outage, "weathervulnerability-ma")
    # ---------------------------------------------------



    # # # ---------------------------------------------------
    # # #  Plot Alpha

    # alpha = model2.halpha.detach().numpy()
    # save_significant_alpha(model2, loc_ids, obs_outage)
    # # plot_data_on_map_in_color(alpha.sum(0), locs, "Critical cities")
    # # plot_data_on_map_in_color(alpha.sum(1), locs, "Vulnerable cities")
    # # # ---------------------------------------------------



    # # ---------------------------------------------------
    # #  Plot param space

    # _, obs_weather, _, _ = dataloader(config["MA Mar 2018"], standardization=False)
    # # plot_nn_params(model2, obs_weather)
    # plot_nn_3Dparams(model2, obs_weather)
    # # ---------------------------------------------------



    # ---------------------------------------------------
    #  Plot base intensity

    # plot_baselines_and_lambdas(model2, config["MA Mar 2018"]["_startt"], obs_outage)
    # plot_spatial_base(model2, locs, obs_outage)
    # plot_spatial_lam_minus_base(model2, locs, obs_outage)
    # plot_spatial_ratio(model2, locs, obs_outage)
    # plot_spatial_base_and_cascade(model2, locs, obs_outage)
    # plot_spatial_base_and_cascade_over_time(model2, locs, obs_outage)
    # ---------------------------------------------------

    # ---------------------------------------------------
    # Plot outage and weather on a line chart

    # N      = 129
    # feats  = [6, -4]
    # colors = ["#DC143C", "#0165fc"] #, "#3f9b0b"]
    # labels = ["Derived radar reflectivity", "Wind speed"]
    # obs_outage, obs_weather, _, _ = dataloader(config["Normal MA Mar 2018"], standardization=False, weatherN=1) 
    # obs_outage                 = obs_outage[:, :N]
    # obs_weather_show           = obs_weather[:, :N*3, feats]
    # obs_weather_normal         = obs_weather[:, N*3:220*3, feats]
    # plot_outage_and_weather_linechart(
    #     "Nor'easters in March 2018, MA", 
    #     N, config["MA Mar 2018"]["_startt"], 
    #     obs_outage, obs_weather_show, obs_weather_normal, labels, colors, 
    #     dayinterval=3)

    # N      = 129
    # feat   = 6
    # obs_outage, geo_outage = load_outage(config["Normal MA Mar 2018"], N=1)
    # obs_feats, geo_weather = load_weather(config["Normal MA Mar 2018"])
    # obs_outage             = obs_outage[:N, :].sum(0)
    # obs_weather_show       = obs_feats[feat, :N, :].mean(0)
    # obs_weather_normal     = obs_feats[feat, N*3:220*3, :].mean(0)
    # # obs_weather_show       = obs_feats[feat, 25, :]
    # # obs_weather_normal     = obs_feats[feat, 0, :]
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage, obs_weather_show, obs_weather_normal)

    # ---------------------------------------------------


    # ---------------------------------------------------
    # Plot outage and weather on the map

    # # N      = 129
    # # feats  = [6, -4]
    # # colors = ["#DC143C", "#0165fc"] #, "#3f9b0b"]
    # # labels = ["Derived radar reflectivity", "Wind speed"]
    # # obs_outage, obs_weather, _, _ = dataloader(config["Normal MA Mar 2018"], standardization=False, weatherN=1) 
    # # obs_outage                 = obs_outage[:, :N]
    # # obs_weather_show           = obs_weather[:, :N*3, feats]
    # # obs_weather_normal         = obs_weather[:, N*3:220*3, feats]
    # # plot_outage_and_weather_linechart(
    # #     "Nor'easters in March 2018, MA", 
    # #     N, config["MA Mar 2018"]["_startt"], 
    # #     obs_outage, obs_weather_show, obs_weather_normal, labels, colors, 
    # #     dayinterval=3)

    # N      = 129
    # feats  = [6, -4]
    # obs_outage, obs_weather, geo_outage, geo_weather = dataloader(
    #     config["Normal MA Mar 2018"], standardization=False, isproj=False) 
    # print(obs_outage.shape)
    # print(obs_weather.shape)
    
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage[:, 10], obs_weather[:, 10, feats], filename="outage-vs-radar-map-t10", titlename="06:00 Mar 2, 2018 MA")
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage[:, 11], obs_weather[:, 11, feats], filename="outage-vs-radar-map-t11", titlename="09:00 Mar 2, 2018 MA")
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage[:, 12], obs_weather[:, 12, feats], filename="outage-vs-radar-map-t12", titlename="12:00 Mar 2, 2018 MA")
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage[:, 13], obs_weather[:, 13, feats], filename="outage-vs-radar-map-t13", titlename="15:00 Mar 2, 2018 MA")
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage[:, 14], obs_weather[:, 14, feats], filename="outage-vs-radar-map-t14", titlename="18:00 Mar 2, 2018 MA")
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage[:, 15], obs_weather[:, 15, feats], filename="outage-vs-radar-map-t15", titlename="21:00 Mar 2, 2018 MA")
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage[:, 16], obs_weather[:, 16, feats], filename="outage-vs-radar-map-t16", titlename="00:00 Mar 3, 2018 MA")
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage[:, 17], obs_weather[:, 17, feats], filename="outage-vs-radar-map-t17", titlename="03:00 Mar 3, 2018 MA")
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage[:, 18], obs_weather[:, 18, feats], filename="outage-vs-radar-map-t18", titlename="06:00 Mar 3, 2018 MA")
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage[:, 19], obs_weather[:, 19, feats], filename="outage-vs-radar-map-t19", titlename="09:00 Mar 3, 2018 MA")
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage[:, 20], obs_weather[:, 20, feats], filename="outage-vs-radar-map-t20", titlename="12:00 Mar 3, 2018 MA")

    # ---------------------------------------------------

    # ---------------------------------------------------
    # OUTAGE QQPLOT

    # obs_outage, geo_outage = load_outage(config["Normal MA Mar 2018"], N=4)
    # print(obs_outage.shape)

    # plt.plot(obs_outage.sum(1))
    # plt.show()

    # startt     = 25
    # endt       = 125
    # _outage    = obs_outage[startt:endt, :]
    # start_date = str(arrow.get(config["Normal MA Mar 2018"]["_startt"], "YYYY-MM-DD HH:mm:ss").shift(hours=startt).format("YYYY-MM-DD HH:mm:ss"))
    # plot_interval_qqplot(_outage, start_date, "1st Nor'easter in 2018, MA", dayinterval=1, vmin=10, vmax=21000)

    # startt = 160
    # endt   = 260
    # _outage    = obs_outage[startt:endt, :]
    # start_date = str(arrow.get(config["Normal MA Mar 2018"]["_startt"], "YYYY-MM-DD HH:mm:ss").shift(hours=startt).format("YYYY-MM-DD HH:mm:ss"))
    # plot_interval_qqplot(_outage, start_date, "2nd Nor'easter in 2018, MA", dayinterval=1, vmin=10, vmax=21000)


    # startt = 290
    # endt   = 360
    # _outage    = obs_outage[startt:endt, :]
    # start_date = str(arrow.get(config["Normal MA Mar 2018"]["_startt"], "YYYY-MM-DD HH:mm:ss").shift(hours=startt).format("YYYY-MM-DD HH:mm:ss"))
    # plot_interval_qqplot(_outage, start_date, "3rd Nor'easter in 2018, MA", dayinterval=1, vmin=10, vmax=21000)

    # startt = 400
    # endt   = 500
    # _outage    = obs_outage[startt:endt, :]
    # start_date = str(arrow.get(config["Normal MA Mar 2018"]["_startt"], "YYYY-MM-DD HH:mm:ss").shift(hours=startt).format("YYYY-MM-DD HH:mm:ss"))
    # plot_interval_qqplot(_outage, start_date, "Daily operation in 2018, MA", dayinterval=1, vmin=10, vmax=21000)
    # ---------------------------------------------------


    # # ---------------------------------------------------
    # # Number of Customers

    # # load data
    # ncust = np.load("data/ncustomer_ma.npy")
    # obs_outage, obs_weather, loc, _ = dataloader(
    #     config["MA Mar 2018"], standardization=True, outageN=1, weatherN=1, isproj=True)
    # print(obs_outage.shape, obs_weather.shape)
    # max_outage = obs_outage.max(1)

    # cm  = plt.cm.get_cmap('Reds')
    # plt.scatter(ncust, max_outage, c=loc[:, 1], cmap=cm)
    # plt.show()
    # # ---------------------------------------------------



    # ---------------------------------------------------
    # Time Lag

    # # load data
    # ncust = np.load("data/ncustomer_ma.npy")
    # obs_outage, obs_weather, _, _ = dataloader(
    #     config["MA Mar 2018"], standardization=True, outageN=1, weatherN=1, isproj=True)
    # print(obs_outage.shape, obs_weather.shape)

    # # obs_outage   = obs_outage[:, :150]
    # # obs_weather  = obs_weather[:, :150]
    # # obs_outage   = obs_outage[:, 150:280]
    # # obs_weather  = obs_weather[:, 150:280]
    # obs_outage   = obs_outage[:, 280:]
    # obs_weather  = obs_weather[:, 280:]

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