from GD import GD
from GD_m import GD_m
from AdaGrad import AdaGrad
from RMSProp import RMSProp
from Adam import Adam
import csv


def write_matrix(mat, algname):
    with open('outputs/' + algname + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        for i in range(999):
            writer.writerow(mat[:,i])

def main():
    f = '(x+1)**2+(y**2)'
    init = {'x': 3, 'y': 1}

    # 1. Vanilla Gradient Descent
    alg_GD = GD(f)
    min_GD, t_GD = alg_GD.perform(init)
    write_matrix(min_GD, 'GD')

    # 2. Gradient Descent with Momentum
    alg_GDm = GD_m(f)
    min_GDm, t_GDm, mu_GDm = alg_GDm.perform(init, validation=True)
    write_matrix(min_GDm, 'GDm')

    # 3. AdaGrad
    alg_AdaGrad = AdaGrad(f)
    min_AdaGrad, t_AdaGrad = alg_AdaGrad.perform(init)
    write_matrix(min_AdaGrad, 'AdaGrad')

    # 4. RMSProp
    alg_RMSProp = RMSProp(f)
    min_RMSProp, t_RMSProp, _ = alg_RMSProp.perform(init, validation=True)
    write_matrix(min_RMSProp, 'RMSProp')

    # 5. Adam
    alg_Adam = Adam(f)
    min_Adam, t_Adam, _, _ = alg_Adam.perform(init, validation=True)
    write_matrix(min_Adam, 'Adam')

main()
