import math
import numpy as np

def get_numpy_rng(seed):
    '''get numpy random state

    Parameters:
        seed (int,None,np.ranom.RandomState): if `int` or `None`, then `np.random.default_rng(seed)`,
                otherwise, return `seed` itself

    Returns:
        ret (np.ranom.RandomState): None
    '''
    if seed is None:
        np_rng = np.random.default_rng()
    elif isinstance(seed, int):
        np_rng = np.random.default_rng(seed)
    else:
        np_rng = seed
    return np_rng


def sample_D(Z, B, size=None, seed=None):
    np_rng = get_numpy_rng(seed)
    # Define the support of the distribution
    support = [x for x in range(-Z//2 + 1, Z//2 + 1) if np.abs(x) <= B]

    # Compute the unnormalized probabilities
    unnormalized_probs = [math.exp(-math.pi * x**2 / B**2) for x in support]

    # Normalize the probabilities
    sum_probs = sum(unnormalized_probs)
    probs = [p / sum_probs for p in unnormalized_probs]

    return np_rng.choice(support, size=size, p=probs)


def GenTrap(n, q, B, seed=None):
    np_rng = get_numpy_rng(seed)
    k = int(math.log2(q))
    m = 2 * n * k
    # R is a Gaussian matrix in Z_q^(nlogq x nlogq) use sample_D to sample each entry of R from D_Zq, B
    R = sample_D(q, B, size=(int(n * k), int(n * k)), seed=np_rng)
    # A_hat is a random matrix in Z_q^(n x nlogq)
    A_hat = np_rng.integers(-q//2, q//2, size=(n, int(n * k)))
    # g is the gadget matrix [1, 2, 4, ..., 2^(k-1)]

    g = np.array([2**i for i in range(k)], dtype=np.int64)
    # G in g \otimes I_n
    G = np.kron(np.eye(n, dtype=np.int64), g)
    # A is [A_hat , G - A_hat * R]
    A = np.hstack((A_hat, np.remainder(G - np.dot(A_hat, R), q)))
    # t_A = [R, I]
    t_A = np.vstack((R, np.eye(n * k)))
    # A should be in n x 2nlogq
    assert A.shape == (n, 2 * n * k)
    # t_A should be in 2nlogq x nlogq
    assert t_A.shape == (2 * n * k, n * k)
    # return transpose of A and t_A
    return A.T, t_A.T
