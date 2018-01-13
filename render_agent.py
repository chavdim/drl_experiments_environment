#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:27:36 2017

@author: chavdar
"""

import gym_ple,keras,gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import rescale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def drawShadow(im,flow,thresh=0.6):
    shadows = np.ones_like(im)/2
    fy = flow[0:,0:,1]
    shadows[fy>thresh] =1.0
    shadows[fy<-thresh] =0
    return shadows
env = gym.make('ppong-v0')
#env = gym.make('ppong-v0')
env.seed()
#13
#model1 = keras.models.load_model('_model_gap_mini_pixelcopter3')
model1 = keras.models.load_model("results/exp_pong_smallball/models/model1")
#model2 = keras.models.load_model("results/exp1/models/model2")
sim_len = 5000
sim_times = 3
rews=[]
save_obs = []
multi = False
opt_flow=True
rand_agent=False
obsShape = [50,50,3]# cropped , rescaled 
zoom = [1,1]
originalDim = list(env.observation_space.shape)
        #
t=0
for i in originalDim[0:-1]:
    zoom[t] = obsShape[t]/i
    t+=1
        #
zc = zoom[1]
zoom[1]=zoom[0]
zoom[0]=zc

for ttt in range( sim_times):
    observation = env.reset()
    observation = rescale(observation,zoom)
    obs2 = np.copy(observation)
    ep_rew = 0
    if opt_flow:
        obs2[0:,0:,2] = obs2[0:,0:,2]*0
    for tt in range(sim_len):
        if ttt==0: # which one to save obs of
            save_obs.append(obs2)
        env.render()
        if rand_agent:
            ac = env.action_space.sample()
        elif multi:
            ac = model1.predict_proba(np.reshape(obs2,[1,48,48,3]),verbose=0)
            ac1 = model2.predict_proba(np.reshape(obs2,[1,48,48,3]),verbose=0)
            #print(ac,np.argmax(ac[0]))
            ac = np.argmax((ac[0]+ac1[0])/2) 
        else:
            ac = model1.predict(np.reshape(obs2,[1,50,50,3]),verbose=0)
            ac = np.argmax(ac[0]) 
        #np.save("aa2"+str(tt), observation/255.0)
        #ac  = s.agent.getBestAction(observation/255.0)
        
        #ac = 1#env.action_space.sample()
        #ac = 0
        prev_state = np.copy(obs2)
        observation, reward, done, info  = env.step(ac)
        observation = rescale(observation,zoom)
        obs2 = np.copy(observation)
        if opt_flow:
            of_y = cv2.calcOpticalFlowFarneback(rgb2gray(prev_state*255),
                        rgb2gray(obs2*255),None,0.5, 3, 5, 3, 5, 1.2, 0)[0:,0:,1]
            obs2[0:,0:,2] =of_y/10
            #gray=rgb2gray(obs2*255)
            #of = cv2.calcOpticalFlowFarneback(rgb2gray(prev_state*255),
            #            gray,None,0.5, 3, 5, 3, 5, 1.2, 0)
            #obs2[0:,0:,2] = drawShadow(gray,of)
        
        ep_rew += reward
        
        #plt.imshow(observation)
        #print(np.max(obs2[0:,0:,1]),np.min(obs2[0:,0:,1]))
        #plt.imshow(obs2[0:,0:,2],alpha=0.22)
        #plt.show()
        
        if done:
            break
    rews.append(ep_rew)
print(rews)
print(np.mean(rews))
save_obs =np.array(save_obs)
#np.save("obs_for_rendering_noshadow",save_obs)