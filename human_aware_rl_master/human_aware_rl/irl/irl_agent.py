# Code adopted from https://github.com/jangirrishabh/toyCarIRL/blob/master/toy_car_IRL.py

# IRL algorith developed for the toy car obstacle avoidance problem for testing.
import numpy as np
# from nn import neural_net #construct the nn and send to playing
from cvxopt import matrix, solvers  #convex optimization library

class irlAppAgent:
    def __init__(self, expertFE):
        # self.randomPolicy = randomFE
        self.expertPolicy = expertFE
        # self.randomT = np.linalg.norm(np.asarray(self.expertPolicy)-np.asarray(self.randomPolicy)) #norm of the diff in expert and random
        self.policiesFE = {} # storing the policies and their respective t values in a dictionary
        # print ("Expert - Random at the Start (t) :: " , self.randomT) 
    
    def _policyListUpdater(self, tempFE, reward_func):  #add the policyFE list and differences
        '''
        Calculate the hyper distance with the current reward function
        - tempFE: the feature expectation of the RL agent with the current reward func
        '''
        expt_to_agent = np.asarray(self.expertPolicy)-np.asarray(tempFE)
        hyperDistance = np.abs(reward_func(expt_to_agent)) #hyperdistance = t
        hyperDistance = hyperDistance.item()
        self.policiesFE[hyperDistance] = tempFE
        return hyperDistance # t = (weights.tanspose)*(expert-newPolicy)
        
    def optimalWeightFinder(self, tempFE, reward_func):
        # while True:
        print ("the distances  ::", self.policiesFE.keys())
        currentT = self._policyListUpdater(tempFE, reward_func)
        print ("Current distance (t) is:: ", currentT )
        # if self.currentT <= self.epsilon: # terminate if the point reached close enough
        #     break
        # i += 1
        W = self._optimization() # optimize to find new weights in the list of policies
        # print ("new weights ::", W )
        return W, currentT
    
    def _optimization(self): # implement the convex optimization, posed as an SVM problem
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
