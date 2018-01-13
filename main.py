#!/usr/bin/env python3

import gym,random,time
from gym import wrappers
#
import qnn_agent
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import mpl_toolkits.mplot3d.axes3d as p3

import random
import numpy as np
from skimage.transform import rescale
import gym_ple
import cv2 
import keras
#np.dot(rs[...,:3], [0.299, 0.587, 0.114])
#random.seed(1)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def drawShadow(im,flow,thresh=0.6):
    shadows = np.ones_like(im)/2
    fy = flow[0:,0:,1]
    shadows[fy>thresh] =1.0
    shadows[fy<-thresh] =0
    return shadows
class Memory:
    def __init__(self,s_shape,a_size,r_size,maxSize = 100000):

        s_shape.insert(0,maxSize)
        self.colSize = (a_size+r_size+1) #+1 for done boolean
        #state storages
        self.stateStorage = np.empty(s_shape,dtype='float32')
        self.newStateStorage = np.empty(s_shape,dtype='float32')
        #
        self.storage = np.empty([maxSize,self.colSize],dtype='float32')
        self.currentRow = 0
        self.maxSize = maxSize
        self.s_size = s_shape
        self.a_size = a_size
        self.filledOnce = False
    def addData(self,s,a,s_new,r,done):
        #all_data = np.append(s,a)
        #all_data = np.append(all_data,s_new)
        self.stateStorage[self.currentRow][0:,0:,0:] = np.copy(s)
        self.newStateStorage[self.currentRow][0:,0:,0:] = np.copy(s_new)

        #print(all_data)
        all_data = np.append(a,r)
        all_data = np.append(all_data,done)
        
        self.storage[self.currentRow] = all_data
        self.currentRow += 1
        if self.currentRow == self.maxSize: # reset when full
            self.full()
    def full(self):
        self.currentRow = 0
        self.filledOnce = True
        print("memory full yo")
    def getBatch(self,batchSize=10):
        if self.filledOnce == False:
            choices = np.random.randint(0,self.currentRow , size=batchSize)
        else:
            choices = np.random.randint(0,self.maxSize , size=batchSize)
        
        return  {"state":self.stateStorage[choices],
                "action":self.storage[choices][0:,0:self.a_size],
                "new_state":self.newStateStorage[choices],
                "reward":self.storage[choices][0:,-2:-1],
                "done":self.storage[choices][0:,-1:]
                }
                
###
class Action:
    def __init__(self,name,action_range,isDiscrete):
        self.action_name = name
        self.action_range = action_range
        self.isDiscrete = isDiscrete
class Sim:
    def __init__(self,env_name,nn_params,max_iterations=100000,interval=20):
        #self.env = gym.make('PixelCopter-v0')
        self.env = gym.make(env_name)
        self.skip_frames = 1
        self.skip_frame_timer = self.skip_frames #actions repeated skip_frames -1 times
        self.episode_maxLength = 5000

        self.actions = [Action(0,[0,1],True), #left
                        Action(1,[0,1],True)  #right
                        #Action(2,[0,1],True)
                        ]
        self.lastAction = None
        ##self.obsShape = list(self.env.observation_space.shape)# cropped , rescaled 
        
        #
        self.obsShape = [50,50,3]# cropped , rescaled 
        self.zoom = [1,1]
        self.originalDim = list(self.env.observation_space.shape)
        #
        t=0
        for i in self.originalDim[0:-1]:
            self.zoom[t] = self.obsShape[t]/i
            t+=1
        zc = self.zoom[1]
        self.zoom[1]=self.zoom[0]
        self.zoom[0]=zc

        #
        
        self.agent = qnn_agent.Qnetwork_agent(self.obsShape[:],self.actions,nn_params)
        ####
        self.max_iterations = max_iterations
        self.interval = interval
        self.rewards = []
        self.temp_rews = 0
        self.done = 0
        ####
        self.maxExperienceSize = 50000 # Memory size
        reward_size = 1  # reward vector size
        action_size = 1  # actions take during one step
        
        self.experienceData = Memory(self.obsShape[:],
                                     action_size,
                                     reward_size,
                                     self.maxExperienceSize
                                     )
        self.times={"total":0.0,"get_action":0.0,"train":0.0,"create_batch":0.0}
    def loadModel(self,name):
        self.agent.nn.nn = keras.models.load_model(name)
        self.agent.target_train()
        self.agent.exploreChance =  self.agent.exploration_final_eps
    def runEpisode(self,testAgent=False, doUpdate=False):
        self.env.seed()
        observation = self.env.reset()
        
        #obs2 = observation
        
        obs2 = np.copy(observation/255.0)
        

        episodeReward = 0
        all_rewards = []
        episodeData = []
        for t in range(self.episode_maxLength):
            if self.skip_frame_timer == self.skip_frames:
                if testAgent==False:
                    if doUpdate:
                        self.agent.update(self)
                    t_before_action = time.time()
                    action = self.agent.getNextAction(obs2)
                    self.times["get_action"] += time.time() - t_before_action
                elif testAgent==True:
                    self.env.render()
                    action  = self.agent.getBestAction(obs2)
                self.skip_frame_timer = 0
            self.skip_frame_timer += 1
            self.lastAction = action
            prev_state = np.copy(obs2)
            observation, reward, done, info = self.env.step(action)
            episodeReward += reward 
            all_rewards.append(reward)
            #print(reward)
            obs2 = np.copy(observation/255.0)

            r = reward
            #### log rewards
            self.temp_rews += r
            if self.agent.step_counter % self.interval == 0:
                self.rewards.append(np.mean(self.temp_rews))
                self.temp_rews = 0
            if self.agent.step_counter > self.max_iterations:
                self.done=1
            ####
            r = np.clip(reward, -1, 1)
            
            if testAgent == False:
                if self.skip_frame_timer == 1:
                    self.experienceData.addData(prev_state,action,obs2,r,done)
            if done:
                self.skip_frame_timer = self.skip_frames

                break
            
            if t == self.episode_maxLength:# never
                print("episode max length reached")
                
        #self.env.close()
        if testAgent == False:
            if self.agent.exploreChance > self.agent.exploration_final_eps:
                if doUpdate:
                    self.agent.exploreChance *= 0.992
            return episodeReward
        if testAgent == True:
            return episodeReward
    def runIterations(self,testAgent=False, doUpdate=False,iterations=1000):
        self.env.seed()
        observation = self.env.reset()
        observation = rescale(observation,self.zoom)
        obs2 = np.copy(observation)
        #
        all_rewards = []  
        reseting = 0
        otp_flow=1
        if otp_flow:
            obs2[0:,0:,2] = obs2[0:,0:,2]*0
            #pass
        for t in range(iterations):
            if self.skip_frame_timer == self.skip_frames:
                if testAgent==False:
                    if doUpdate:
                        self.agent.update(self)
                    t_before_action = time.time()
                    action = self.agent.getNextAction(obs2)
                    self.times["get_action"] += time.time() - t_before_action
                elif testAgent==True:
                    self.env.render()
                    action  = self.agent.getBestAction(obs2)
                self.skip_frame_timer = 0
            self.skip_frame_timer += 1
            self.lastAction = action
            prev_state = np.copy(obs2)
            observation, reward, done, info = self.env.step(action)
            
            all_rewards.append(reward)
            #print(reward)
            observation = rescale(observation,self.zoom)
            obs2 = np.copy(observation)
            #plt.imshow(obs2)
            #plt.show()
            ####OPTICAL FLOW
            if reseting==0:
                gray=rgb2gray(obs2)
                of_y = cv2.calcOpticalFlowFarneback(rgb2gray(prev_state*255),
                        gray*255,None,0.5, 3, 5, 3, 5, 1.2, 0)[0:,0:,1]
                #of = cv2.calcOpticalFlowFarneback(rgb2gray(prev_state)*255,
                #        gray*255,None,0.5, 3, 5, 3, 5, 1.2, 0)
                obs2[0:,0:,2] =of_y/10
                #obs2[0:,0:,2] = drawShadow(gray,of)
                #obs2[0:,0:,0] = obs2[0:,0:,0]+ of_y*500
                #obs2[0:,0:,1] = obs2[0:,0:,1]+ of_y*500
                #obs2[0:,0:,2] = obs2[0:,0:,2]+ of_y*500
            else:
                obs2[0:,0:,2] = obs2[0:,0:,2]*0
                #pass
            reseting=0
            r = reward
            #### log rewards
            all_rewards.append(r)
            ####
            r = np.clip(reward, -1, 1)
            
            if testAgent == False:
                if self.skip_frame_timer == 1:
                    self.experienceData.addData(prev_state,action,obs2,r,done)
            if done:
                self.skip_frame_timer = self.skip_frames

                self.env.seed()
                observation = self.env.reset()
                observation = rescale(observation,self.zoom)
                obs2 = np.copy(observation)
                if otp_flow:
                    obs2[0:,0:,2] = obs2[0:,0:,2]*0
                reseting=1
            
            if t == self.episode_maxLength:# never
                print("episode max length reached")
                
        #self.env.close()
        if testAgent == False:
            if self.agent.exploreChance > self.agent.exploration_final_eps:
                if doUpdate:
                    self.agent.exploreChance *= 0.8
            return all_rewards
        if testAgent == True:
            return all_rewards
    def run(self,iterations=1000,doUpdate=True):
        #print("running...")
        results = []
        for i in range(iterations):
            #print("episode: ",i)
            results.append(self.runEpisode(testAgent=False,doUpdate=doUpdate))
        return results
    def run_iterations(self,iterations=1000,doUpdate=True):
        results=[]
        
        r = self.runIterations(testAgent=False,doUpdate=doUpdate,iterations=iterations)
        return r
            
    def testAgent(self,iterations=5):
        results = []
        for i in range(iterations):
            results.append(self.runEpisode(testAgent=True))
        return results
        
"""
s = Sim_Catcher()
while True:
    s.run(1)
    if s.experienceData.filledOnce:
        break
s.agent.learnFromAllExperience(s)
"""
"""
s = Sim_Catcher()
run_steps = 200000
interval = 50 # dont change 
average_rewards = []
while True:
    steps0 = s.agent.step_counter
    t0 = time.time()
    r = s.run(iterations = interval,doUpdate = True)
    steps1 = steps0 - s.agent.step_counter  # steps done
    step_per_sec = steps1 / (t0  - time.time())
    mr = np.mean(r)
    print("steps: ", s.agent.step_counter,"mean reward: ",mr,
                                          "epsilon: ",np.round(s.agent.exploreChance,2),
                                            "steps/sec: ",np.round(step_per_sec,2),
                                            "remaining min: ",np.round(((run_steps-s.agent.step_counter)/step_per_sec)/60,2))
    average_rewards.append(mr)
    if s.agent.step_counter >= run_steps:
        break
"""
#np.save("rewards/0_2mil_2063_4042_6031_lr0002rms_tail",average_rewards)
#s.agent.nn.nn.save("models/ppong/2mil_2063_3042_6031_lr0002rms_tail",include_optimizer=True)