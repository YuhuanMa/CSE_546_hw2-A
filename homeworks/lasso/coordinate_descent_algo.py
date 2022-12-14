from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """
    a = 2 * np.sum(X**2, axis=0)
    return a


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. Respresents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """
    n, d = X.shape
    b = np.mean(y - X @ weight)

    for i in range(d):
        w_i = weight.copy()
        w_i[i] = 0
        c = 2 * np.sum(X[:, i] * (y - (b + X @ w_i)))
        if c > _lambda:
            weight[i] = (c - _lambda) / a[i]
        elif c < -(_lambda):
            weight[i] = (c + _lambda) / a[i]
        else:
            weight[i] = 0
    return (weight, b)


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    output = np.linalg.norm(X @ weight+bias-y)**2 + np.linalg.norm(weight, ord=1) * _lambda
    return output


@problem.tag("hw2-A", start_line=4)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float .

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    a = precalculate_a(X)
    old_w: Optional[np.ndarray] = None

    n_prev = np.copy(start_weight)
    (n, bias) = step(X, y, n_prev, a, _lambda)
    while not convergence_criterion(n, n_prev, convergence_delta):
        n_prev = n
        (n, bias) = step(X, y, n_prev, a, _lambda)
    return (n, bias)



@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    diff = np.linalg.norm(weight - old_w, ord=np.inf)
    return diff < convergence_delta

@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    n = 500
    d = 1000
    k = 100
    sigma = 1
    convergence_delta = 1e-4
    np.random.seed(0)
    X = np.random.normal(0, 1, (n, d))
    w = np.arange(1, d + 1) / k
    w[k:] = 0
    eps = np.random.normal(0, 1, size=n)
    y = X @ w + eps
    currentlam = np.max(2 * np.abs(np.dot((y - np.mean(y)), X)))
    lambdas = []
    nonzeros = []
    nonzero = 0
    FDR = []
    TPR = []
    while nonzero < d:
        lambdas.append(currentlam)
        (weight, bias) = train(X, y, currentlam, convergence_delta, None)
        nonzero = np.count_nonzero(weight)
        nonzeros.append(nonzero)
        currentlam = currentlam/2
        if nonzero == 0:
            FDR.append(0)
        else:
            FDR.append(np.count_nonzero(weight[k:]) / nonzero)
        TPR.append(np.count_nonzero(weight[0: k]) / k)

    print(FDR)
    print(TPR)
    # plot the image for 5a
    plt.plot(lambdas, nonzeros)
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('the number of nonzeros')
    plt.title('The plot for 5a')
    plt.show()

    # plot the image for 5b
    plt.figure()
    plt.plot(FDR, TPR)
    plt.xlabel('FDR')
    plt.ylabel('TPR')
    plt.title('The plot for 5b')
    plt.show()


if __name__ == "__main__":
    main()
