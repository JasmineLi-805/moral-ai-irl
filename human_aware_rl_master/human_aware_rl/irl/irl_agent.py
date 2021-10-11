# Code adopted from https://github.com/jangirrishabh/toyCarIRL/blob/master/toy_car_IRL.py

# IRL algorith developed for the toy car obstacle avoidance problem for testing.
import numpy as np
import logging
import scipy
# from playing import play #get the RL Test agent, gives out feature expectations after 2000 frames
# from nn import neural_net #construct the nn and send to playing
from cvxopt import matrix, solvers  #convex optimization library
# from flat_game import carmunk # get the environment

NUM_STATES = 8 
BEHAVIOR = 'red' # yellow/brown/red/bumping
FRAMES = 100000 # number of RL training frames per iteration of IRL

class irlAppAgent:
    def __init__(self, randomFE, expertFE, epsilon, num_states, num_frames, behavior, reward_func):
        self.randomPolicy = randomFE
        self.expertPolicy = expertFE
        self.num_states = num_states
        self.num_frames = num_frames
        self.behavior = behavior
        self.epsilon = epsilon # termination when t<0.1
        self.randomT = np.linalg.norm(np.asarray(self.expertPolicy)-np.asarray(self.randomPolicy)) #norm of the diff in expert and random
        self.policiesFE = {self.randomT:self.randomPolicy} # storing the policies and their respective t values in a dictionary
        print ("Expert - Random at the Start (t) :: " , self.randomT) 
        self.currentT = self.randomT
        self.minimumT = self.randomT

        self.reward_func = reward_func

    def getRLAgentFE(self, train_config): #get the feature expectations of a new policy using RL agent
        '''
        Trains an RL agent with the current reward function. 
        Then rolls out one trial of the trained agent and calculate the feature expectation of the RL agent.
        - train_config: the configuration taken by the rllib trainer
        
        Returns the feature expectation.
        '''
        # TODO: implement
        pass

    def getExpertFE(self):
        '''
        Get the expert's feature expectation with the current reward function
        '''
        expertFE = self.reward_func.getFeatureExpectation(self.expertPolicy)
        return expertFE
    
    def policyListUpdater(self, tempFE):  #add the policyFE list and differences
        '''
        Calculate the hyper distance with the current reward function
        - tempFE: the feature expectation of the RL agent with the current reward func
        '''
        expertFE = self.getExpertFE()
        hyperDistance = np.abs(expertFE-tempFE) #hyperdistance = t
        self.policiesFE[hyperDistance] = tempFE
        return hyperDistance # t = (weights.tanspose)*(expert-newPolicy)
        
    def optimalWeightFinder(self):
        # f = open('weights-'+BEHAVIOR+'.txt', 'w')
        i = 1
        while True:
            W = self.optimization() # optimize to find new weights in the list of policies
            print ("weights ::", W )
            # f.write( str(W) )
            # f.write('\n')
            print ("the distances  ::", self.policiesFE.keys())
            self.currentT = self.policyListUpdater(W, i)
            print ("Current distance (t) is:: ", self.currentT )
            if self.currentT <= self.epsilon: # terminate if the point reached close enough
                break
            i += 1
        # f.close()
        return W
    
    def optimization(self): # implement the convex optimization, posed as an SVM problem
        m = len(self.expertPolicy)
        P = matrix(2.0*np.eye(m), tc='d') # min ||w||
        q = matrix(np.zeros(m), tc='d')
        policyList = [self.expertPolicy]
        h_list = [1]
        for i in self.policiesFE.keys():
            policyList.append(self.policiesFE[i])
            h_list.append(1)
        policyMat = np.matrix(policyList)
        policyMat[0] = -1*policyMat[0]
        G = matrix(policyMat, tc='d')
        h = matrix(-np.array(h_list), tc='d')
        sol = solvers.qp(P,q,G,h)

        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights = weights/norm
        return weights # return the normalized weights
