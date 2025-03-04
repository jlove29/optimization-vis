from flask import Flask, request, render_template, redirect, Response, url_for
from GD import GD
from GD_m import GD_m
from AdaGrad import AdaGrad
from RMSProp import RMSProp
from Adam import Adam
import csv
import json
import numpy as np
import math

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return render_template(('index.html'), val=2)

@app.route('/input', methods = ['GET', 'POST'])
def get_input():
    raw = str(request.get_data()).split(',')
    x = raw[0][2:]
    y = raw[1][:-1]
    inits = {'x': x, 'y': y}
    print(inits)
    mat = perform(f, inits)
    return json.dumps(mat)

@app.route('/input_fn', methods = ['GET', 'POST'])
def get_fn_input():
    raw = str(request.get_data())
    raw = raw[2:-1]
    raw = replace_math(raw)
    global f
    f = raw
    return 'OK'

def write_matrix(mat, algname):
    with open('outputs/' + algname + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        for i in range(999):
            writer.writerow(mat[:,i])

def replace_math(f):
    f = f.replace('^', '**')
    f = f.replace('cos', 'math.cos')
    f = f.replace('sin', 'math.sin')
    f = f.replace('sqrt', 'np.sqrt')
    f = f.replace('exp', 'math.e**')
    f = f.replace('pi', 'math.pi')
    print(f)
    return f

def perform(f, init):

    # 1. Vanilla Gradient Descent
    alg_GD = GD(f)
    min_GD, t_GD = alg_GD.perform(init, a=0.01)
    if t_GD == 0:
        t_GD = 10000
    #write_matrix(min_GD, 'GD')
    min_GD = min_GD.tolist()
    min_GD = {'x': min_GD[0], 'y': min_GD[1], 't': t_GD}

    # 2. Gradient Descent with Momentum
    alg_GDm = GD_m(f)
    min_GDm, t_GDm, mu_GDm = alg_GDm.perform(init, validation=False, a=0.01)
    if t_GDm == 0:
        t_GDm = 10000
    #write_matrix(min_GDm, 'GDm')
    min_GDm = min_GDm.tolist()
    min_GDm = {'x': min_GDm[0], 'y': min_GDm[1], 't': t_GDm}

    # 3. AdaGrad
    alg_AdaGrad = AdaGrad(f)
    min_AdaGrad, t_AdaGrad = alg_AdaGrad.perform(init, a=0.1)
    if t_AdaGrad == 0:
        t_AdaGrad = 10000
    #write_matrix(min_AdaGrad, 'AdaGrad')
    min_AdaGrad = min_AdaGrad.tolist()
    min_AdaGrad = {'x': min_AdaGrad[0], 'y': min_AdaGrad[1], 't': t_AdaGrad}

    # 4. RMSProp
    alg_RMSProp = RMSProp(f)
    min_RMSProp, t_RMSProp, param = alg_RMSProp.perform(init, validation=False, a=0.01)
    if t_RMSProp == 0:
        t_RMSProp = 10000
    #write_matrix(min_RMSProp, 'RMSProp')
    min_RMSProp = min_RMSProp.tolist()
    min_RMSProp = {'x': min_RMSProp[0], 'y': min_RMSProp[1], 't': t_RMSProp}

    # 5. Adam
    alg_Adam = Adam(f)
    min_Adam, t_Adam, a, b = alg_Adam.perform(init, validation=False, a=0.05)
    if t_Adam == 0:
        t_Adam = 10000
    #write_matrix(min_Adam, 'Adam')
    min_Adam = min_Adam.tolist()
    min_Adam = {'x': min_Adam[0], 'y': min_Adam[1], 't': t_Adam}

    print(t_GD, t_GDm, t_AdaGrad, t_RMSProp, t_Adam)

    return {
            'GD': {'steps': min_GD, 'time': t_GD},
            'GDm': {'steps': min_GDm, 'time': t_GDm},
            'AdaGrad': {'steps': min_AdaGrad, 'time': t_AdaGrad},
            'RMSProp': {'steps': min_RMSProp, 'time': t_RMSProp},
            'Adam': {'steps': min_Adam, 'time': t_Adam}
            }


if __name__ == '__main__':
    app.run()

