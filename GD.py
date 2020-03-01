import numpy as np
import math
from algorithm import Algorithm

class GD(Algorithm):
    def __init__(self, f):
        super(GD, self).__init__(f)

    def perform(self, init, a=0.01):
        self.a = a
        self.saved = np.zeros((len(list(init)), self.max_iters))

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
                self.saved[p,i+1] = self.saved[p,i] - (self.a * new_grad)
                prev_sz[param] = abs(self.saved[p, i] - self.saved[p,i+1])
            i += 1
        return self.saved, self.convergence

'''
f = 'y**3 - x'
init = {'x': '2.3499999999999996', 'y': '4.366666666666667'}
newGD = GD(f)
newGD.perform(init)
'''

