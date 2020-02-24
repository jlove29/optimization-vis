import numpy as np
import matplotlib.pyplot as plt
from algorithm import Algorithm

class RMSProp(Algorithm):
    def __init__(self, f):
        super(RMSProp, self).__init__(f)
        self.cache = [0 for p in range(self.num_vars)]

    def single_trial(self, init, a=0.01, decay=0.1):
        self.decay = decay
        self.saved = np.zeros((len(list(init)), self.max_iters))
        self.convergence = 0
        self.cache = [0 for p in range(self.num_vars)]

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
                self.cache[p] += (decay * self.cache[p]) + ((1-decay) * (new_grad**2))
                '''
                print(self.saved[p,i])
                print(new_grad)
                print("---")
                '''
                self.saved[p,i+1] = self.saved[p,i] - (new_grad * (1.0/np.sqrt(self.cache[p] + 1e-8)))
                prev_sz[param] = abs(self.saved[p, i] - self.saved[p,i+1])
            i += 1
        return self.saved, self.convergence

    def perform(self, init, a=0.01, validation=True, decay=0.1):
        self.a = a
        if validation == True:
            choices = np.arange(0, 1, 0.05)
        else:
            choices = [decay]
        vals = None
        min_t = self.max_iters + 1
        min_t_index = -1
        for m in range(len(choices)):
            minimum, t = self.single_trial(init, decay=choices[m])
            if t < min_t:
                min_t = t
                min_t_index = m
                vals = minimum
        return vals, min_t, choices[min_t_index]


'''
f = '(x+1)**2+(y**2)'
init = {'x': 3, 'y': 1}
newGD = RMSProp(f)
minimum, t, mu = newGD.perform(init, validation=True)
print(minimum[:,999], t, mu)
'''


