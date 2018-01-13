#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 12:07:14 2018

@author: chavdar
"""
import experimenter
import os
import matplotlib.pyplot as plt
import numpy as np
####
env = 'ppong-v0'
run_params = {"run_interval":2000,
              "max_steps":100000,
              "log":1,
              "save":True,
              "num_runs":1,
              "load_model":"./results/exp_pong_smallball3/models/model1"
                }
nnetwork_params = {"architecture":[["conv",30,8,4],["conv",40,4,3],["conv",60,3,1],
                    ["gap"],["fc",256]],
                    "learn_rate":0.0002,
                    "optimiser":"RMSprop",
                    
                    }

####
epx = experimenter.Experiment("exp_pong_smallball3",env)
epx.run(run_params,nn_params=nnetwork_params)
#### compare results of different experiments
curr_path = os.path.dirname(os.path.realpath(__file__))
def plotMultipleExperiments(exp_names,current_path):
    colors=["r","b","g"]
    t=0
    for experiment_name in exp_names:
        directory = current_path+"/results/" + experiment_name + "/rewards/"
        runs = os.listdir(directory)
        tt=0
        for run_name in runs:
            r = np.load(directory+run_name)
            if tt==0:
                average_rews = np.ones((r.shape[0],len(runs)))
            average_rews[0:,tt] = r
            plt.plot(r,colors[t],alpha=0.25)
            tt+=1
        plt.plot(np.mean(average_rews,axis=1),colors[t])
        t+=1
    plt.show()
        