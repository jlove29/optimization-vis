import numpy as np
from algorithm import Algorithm

class AdaGrad(Algorithm):
    def __init__(self, f):
        super(AdaGrad, self).__init__(f)
        self.cache = [0 for p in range(self.num_vars)]

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
                self.cache[p] += new_grad**2
                '''
                print(self.saved[p,i])
                print(new_grad)
                print("---")
                '''
                self.saved[p,i+1] = self.saved[p,i] - (new_grad * (1.0/np.sqrt(self.cache[p] + 1e-8)))
                prev_sz[param] = abs(self.saved[p, i] - self.saved[p,i+1])
            i += 1
        return self.saved, self.convergence

