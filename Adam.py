import numpy as np
import math
#import matplotlib.pyplot as plt
from algorithm import Algorithm

class Adam(Algorithm):
    def __init__(self, f):
        super(Adam, self).__init__(f)
        self.momentum = 0
        self.velocity = 0

    def single_trial(self, init, beta1=0.9, beta2=0.99):
        self.saved = np.zeros((len(list(init)), self.max_iters))
        self.convergence = 0
        self.momentum = 0
        self.velocity = 0

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
                '''
                print(self.saved[p,i])
                print(new_grad)
                print("---")
                '''
                self.momentum = (beta1 * self.momentum) + ((1 - beta1)*new_grad)
                self.velocity = (beta2 * self.velocity) + ((1 - beta2)*(new_grad**2))
                self.saved[p,i+1] = self.saved[p,i] - (self.a * (self.momentum * (1.0/(np.sqrt(self.velocity)+1e-8))))
                prev_sz[param] = abs(self.saved[p, i] - self.saved[p,i+1])
            i += 1
        if self.convergence == 0:
            self.convergence = self.max_iters
        return self.saved, self.convergence

    def perform(self, init, a=0.01, validation=True, beta1=0.9, beta2=0.99):
        self.a = a
        self.beta1 = beta1
        self.beta2 = beta2
        if validation == True:
            beta1_choices = np.arange(0, 1, 0.2)
            beta2_choices = np.arange(0, 1, 0.2)
        else:
            beta1_choices = [self.beta1]
            beta2_choices = [self.beta2]
        vals = None
        min_t = self.max_iters + 1
        min_t_b1 = -1
        min_t_b2 = -1
        for m in range(len(beta1_choices)):
            for n in range(len(beta2_choices)):
                minimum, t = self.single_trial(init, beta1=beta1_choices[m], beta2=beta2_choices[n])
                if t < min_t:
                    min_t = t
                    min_t_b1 = m
                    min_t_b2 = n
                    vals = minimum
                #ys = minimum[1,:]
                #xs = range(len(ys))
                #plt.plot(xs, ys)
                #plt.show()
                #print(minimum[:,999], t, beta1_choices[m], beta2_choices[n])
        return vals, min_t, beta1_choices[min_t_b1], beta2_choices[min_t_b2]



'''
f = '(x+1)**2+(y**2)'
init = {'x': 3, 'y': 1}
newGD = Adam(f)
minimum, t, b1, b2 = newGD.perform(init, validation=True)
print(minimum[:,999], t, b1, b2)
'''


