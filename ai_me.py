# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:39:30 2021

@author: mody_
"""


import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating architexture of neural network
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action): # input_size representing num of sensors, orientation, -orientation
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
    def forward(self, state): # state reperesents the input of neural network 
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x) # getting q-values for each of our action if we used sotmax or argmax we will get the action only without q-values
        return q_values

# Implementing experience replay, 'instead of getting one previous step we get some n-steps in past and the current step'
# By that we get something like long short term memory
class ReplayMemory():
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event): # event or trainsition we adding to memory is a tuple of 4 events (last_state'st', new_state'st+1', last_action'at', last_reward'rt')
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del(self.memory[0])
    
    # get random samples from memory
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        # remember that we use torch variable which contais the variable and its gradient
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing DQN
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = [] # mean of the last n_rewards aka mean of rewards overtime
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_action = 0
        self.last_reward = 0
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #vector of 5 dimensions the 3 sensors, orientation and -orientation
        # why unsqueeze, cause any nn takes batch of observations so this adds first fake dimension to the data
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True)) * 100)
        action = probs.multinomial(num_samples=1) # draw random action
        return action.data[0,0] # because it's returned as fake batch with tensor variable
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            print('learing now...')
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")