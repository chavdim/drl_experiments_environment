#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 12:05:35 2018

@author: chavdar
"""
#own
import main
#
import numpy as np
import keras
import time # calculate learning times
import os # folder creation for saving results
import matplotlib.pyplot as plt # plotting results

class Experiment():
    def __init__(self,name,env,param_dict0={},param_dict1={}):
        self.env = env
        self.name = name
        
    def run(self,param_dict={},nn_params={},num_run = -1):
        self.params=param_dict
        interval = param_dict["run_interval"]
        run_steps = param_dict["max_steps"]
        log=param_dict["log"]
        save=param_dict["save"]
        load_model = param_dict["load_model"]
        if num_run == -1:
            num_run = param_dict["num_runs"]
        #
        best_score = -100000
        self.simulation = main.Sim(self.env,nn_params)
        if load_model!=False:
            self.simulation.loadModel(load_model)
        #
        average_rewards = []
        r = self.simulation.run_iterations(iterations = interval,doUpdate = False)
        r = np.mean(r) #random agent baseline ( random dqn weights )
        average_rewards.append(r)
        while True:
            steps0 = self.simulation.agent.step_counter
            t0 = time.time()
            r = self.simulation.run_iterations(iterations=interval,doUpdate = True)
            steps1 = steps0 - self.simulation.agent.step_counter  # steps done
            step_per_sec = steps1 / (t0  - time.time())
            mr = np.mean(r)
            
            if log:
                print("steps: ", self.simulation.agent.step_counter,"mean reward: ",mr,
                    "epsilon: ",np.round(self.simulation.agent.exploreChance,2),
                    "steps/sec: ",np.round(step_per_sec,2),
                    "remaining min: ",np.round(((run_steps-self.simulation.agent.step_counter)/step_per_sec)/60,2))
            average_rewards.append(mr)
            ### save best
            if mr > best_score and save:
                if log:
                    print("new best score, saving model...")
                self.save_current_model(num_run=num_run)
                best_score = mr
            ##
            if self.simulation.agent.step_counter >= run_steps:
                if save:
                    self.save_rewards(average_rewards,num_run=num_run)
                break
        #
        if num_run > 1:
            self.run(param_dict,nn_params=nn_params,num_run=num_run-1)
        else:
            self.plot_results()
    def save_rewards(self,average_rewards,num_run):
        curr_path = os.path.dirname(os.path.realpath(__file__))
        ### save rewards
        directory =curr_path+"/results/"+self.name+"/rewards"
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory+"/rewards"+str(num_run),np.array(average_rewards))

    def save_current_model(self,num_run):
        curr_path = os.path.dirname(os.path.realpath(__file__))
        directory =curr_path+"/results/"+self.name+"/models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.simulation.agent.nn.nn.save(directory+"/model"+str(num_run),include_optimizer=True)
    def plot_results(self):
        curr_path = os.path.dirname(os.path.realpath(__file__))
        directory =curr_path+"/results/"+self.name+"/rewards"
        
        for i in range(1,self.params["num_runs"]+1):
            res = np.load(directory+"/rewards"+str(i)+".npy")
            if i == 1:
                average_res = np.zeros((len(res),self.params["num_runs"]))
            average_res[0:,i-1] = res[0:average_res.shape[0]]
            plt.plot(range(0,average_res.shape[0]*self.params["run_interval"],self.params["run_interval"]),res[0:average_res.shape[0]],alpha=0.225)
        plt.plot(range(0,average_res.shape[0]*self.params["run_interval"],self.params["run_interval"]),np.mean(average_res,axis=1))
        #plt.show()
        plt.savefig(directory+"_plot.png")
if __name__ == "__main__":
    exp = Experiment(env="PixelCopter-v0")
    exp.run()
