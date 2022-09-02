if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem
import pandas as pd


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")
    x_tra = df_train.drop('ViolentCrimesPerPop', axis=1).values
    y_tra = df_train['ViolentCrimesPerPop'].values
    x_test = df_test.drop('ViolentCrimesPerPop', axis=1).values
    y_test = df_test['ViolentCrimesPerPop'].values
    n,d = x_tra.shape

    w_initial = np.zeros(d)
    currentlam = np.max(2 * np.abs(np.dot((y_tra - np.mean(y_tra)), x_tra)))
    lambdas = []
    nonzeros = []
    nonzero = 0
    convergence_delta = 1e-4
    regpath = []
    errtra = []
    errtest = []


    while currentlam >= 0.01:
        lambdas.append(currentlam)
        (weight, bias) = train(x_tra, y_tra, currentlam, convergence_delta, w_initial)
        w_initial = weight
        currentlam = currentlam / 2
        print(currentlam)
        # part c
        nonzero = np.count_nonzero(weight)
        nonzeros.append(nonzero)
        # part d
        regpath.append(weight.copy())
        # part e
        errtra.append(np.mean((x_tra @ weight - y_tra) ** 2))
        errtest.append(np.mean((x_test @ weight - y_test) ** 2))

    # plot for part c
    plt.plot(lambdas, nonzeros)
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('the number of nonzeros')
    plt.title('The plot for 6c')
    plt.show()

    # plot for part d
    agePct12t29 = []
    pctWSocSec = []
    pctUrban = []
    agePct65up = []
    householdsize = []
    for i in range(len(regpath)):
        agePct12t29.append(regpath[i][3])
        pctWSocSec.append(regpath[i][12])
        pctUrban.append(regpath[i][7])
        agePct65up.append(regpath[i][5])
        householdsize.append(regpath[i][1])
    plt.figure()
    plt.plot(lambdas, agePct12t29)
    plt.plot(lambdas, pctWSocSec)
    plt.plot(lambdas, pctUrban)
    plt.plot(lambdas, agePct65up)
    plt.plot(lambdas, householdsize)
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('regularization path')
    plt.legend(('agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize'))
    plt.show()

    # part e
    plt.plot(lambdas, errtra)
    plt.plot(lambdas, errtest)
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('mean squared error')
    plt.legend(('train set', 'test set'))
    plt.show()

    # part f
    lam = 30
    # use the weight for \lamda = 33.84266266457681 as initial value
    test = regpath[3][:]
    print(test)
    (weight_30, bias_30) = train(x_tra, y_tra, lam, convergence_delta, test)
    print(weight_30)
    max_idx = np.argmax(weight_30)
    print('The index for largest (most positive) Lasso coefficient is:', max_idx)
    min_idx = np.argmin(weight_30)
    print('The index for smallest (most negative) Lasso coefficient is:', min_idx)


if __name__ == "__main__":
    main()
