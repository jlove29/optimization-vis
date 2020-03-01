import numpy as np
import math
#import matplotlib.pyplot as plt
from algorithm import Algorithm

class GD_m(Algorithm):
    def __init__(self, f):
        super(GD_m, self).__init__(f)
        self.momentum = 0

    def single_trial(self, init, a=0.01, mu=0.85):
        self.mu = mu
        self.saved = np.zeros((len(list(init)), self.max_iters))
        self.convergence = 0

        # initialize changes, add init to array
        prev_sz = {}
        for p in range(self.num_vars):
            param = self.vars_ordered[p]
            prev_sz[param] = float(np.inf)
            self.saved[p][0] = init[param]

        i = 0
        while i < self.max_iters-1:
            if self.convergence == 0:
                tobreak = True
                for p in prev_sz:
                    if prev_sz[p] > self.t:
                        tobreak = False
                        break
                if tobreak == True:
                    self.convergence = i

            for p in range(self.num_vars):
                param = self.vars_ordered[p]
                new_grad = self.calc_gradient(param, i)
                if new_grad == 0:
                    for j in range(i, self.max_iters-1):
                        self.saved[p,j] = self.saved[p,i]
                        return self.saved, self.convergence
                '''
                print(self.saved[p,i])
                print(new_grad)
                print("---")
                '''
                self.momentum = (self.mu * self.momentum) - (self.a * new_grad)
                self.saved[p,i+1] = self.saved[p,i] + self.momentum
                prev_sz[param] = abs(self.saved[p,i] - self.saved[p,i+1])
            i += 1
        return self.saved, self.convergence

    def perform(self, init, a=0.01, validation=True, mu=0.85):
        self.a = a
        self.momentum = 0
        if validation == True:
            choices = np.arange(0, 1, 0.05)
        else:
            choices = [mu]
        vals = None
        min_t = self.max_iters + 1
        min_t_index = -1
        for m in range(len(choices)):
            minimum, t = self.single_trial(init, mu=choices[m])
            if t < min_t:
                min_t = t
                min_t_index = m
                vals = minimum
        return vals, min_t, choices[min_t_index]


'''
f = '2**x - y**2'
init = {'x': '4', 'y': '0.1'}
newGD = GD_m(f)
minimum, t, mu = newGD.perform(init, validation=True)
print(minimum[:,999], t, mu)
'''

