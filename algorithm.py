import numpy as np

class Algorithm():
    def __init__(self, f):
        self.f = f
        self.convergence = 0
        self.vars = {'x','y','z','w','u','v'}
        found = set()
        for i in f:
            if i in self.vars:
                found.add(i)
        self.vars = found
        self.num_vars = len(list(self.vars))
        self.vars_ordered = sorted([i for i in self.vars])
        self.h = 0.0001
        self.max_iters = 1000
        self.t = 0.0001

    def calc_gradient(self, param, i):
        f = self.f
        for p in range(self.num_vars):
            if self.vars_ordered[p] == param:
                param_num = p
                continue
            f = f.replace(self.vars_ordered[p], str(self.saved[p,i]))
        param_val = self.saved[param_num, i]
        step = 0.01
        area = list(np.arange(param_val-0.1, param_val+0.1, step))
        vals = np.asarray([eval(f.replace(param, '(' + str(i) + ')')) for i in area])
        grad = np.gradient(vals)[10]*(1.0/step)
        return grad
