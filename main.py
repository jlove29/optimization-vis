from flask import Flask, request, render_template, redirect, Response, url_for
from GD import GD
from GD_m import GD_m
from AdaGrad import AdaGrad
from RMSProp import RMSProp
from Adam import Adam
import csv
import json

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return render_template(('index.html'), val=2)

@app.route('/input', methods = ['GET', 'POST'])
def get_input():
    raw = str(request.get_data())
    raw = raw.replace("'","").split('#')
    f = raw[0][1:]
    inits_raw = raw[1]
    inits_parsed = {}
    for var in inits_raw.split(','):
        parse = var.split(':')
        varname = parse[0]
        varval = float(parse[1])
        inits_parsed[varname] = varval
    mat = perform(f, inits_parsed)
    return json.dumps(mat)

@app.route('/input_fn', methods = ['GET', 'POST'])
def get_fn_input():
    raw = str(request.get_data())
    raw = raw[2:-1]
    f_values = []
    max_x = 100
    max_y = 100
    for i in range(max_x):
        x = i * 0.001
        for j in range(max_y):
            y = j * 0.001
            inst = raw.replace('x', str(x)).replace('y', str(y))
            val = eval(inst)
            f_values.append(val)
    return json.dumps(f_values)

def write_matrix(mat, algname):
    with open('outputs/' + algname + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        for i in range(999):
            writer.writerow(mat[:,i])

def perform(f, init):

    # 1. Vanilla Gradient Descent
    alg_GD = GD(f)
    min_GD, t_GD = alg_GD.perform(init)
    #write_matrix(min_GD, 'GD')
    min_GD = min_GD.tolist()

    # 2. Gradient Descent with Momentum
    alg_GDm = GD_m(f)
    min_GDm, t_GDm, mu_GDm = alg_GDm.perform(init, validation=False)
    #write_matrix(min_GDm, 'GDm')
    min_GDm = min_GDm.tolist()

    # 3. AdaGrad
    alg_AdaGrad = AdaGrad(f)
    min_AdaGrad, t_AdaGrad = alg_AdaGrad.perform(init)
    #write_matrix(min_AdaGrad, 'AdaGrad')
    min_AdaGrad = min_AdaGrad.tolist()

    # 4. RMSProp
    alg_RMSProp = RMSProp(f)
    min_RMSProp, t_RMSProp, param = alg_RMSProp.perform(init, validation=False)
    #write_matrix(min_RMSProp, 'RMSProp')
    min_RMSProp = min_RMSProp.tolist()

    # 5. Adam
    alg_Adam = Adam(f)
    min_Adam, t_Adam, a, b = alg_Adam.perform(init, validation=False)
    #write_matrix(min_Adam, 'Adam')
    min_Adam = min_Adam.tolist()

    return [min_GD, min_GDm, min_AdaGrad, min_RMSProp, min_Adam]


if __name__ == '__main__':
    app.run()

