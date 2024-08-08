import numpy as np
import math

from ._common import get_numpy_rng, GenTrap, sample_D

# TODO: Implete class for simple HE scheme

def Dual_KeyGen(n, q, B, seed=None):
    np_rng = get_numpy_rng(seed)
    k = int(math.log2(q))
    m = 2 * n * k
    esk = np_rng.integers(0, 2, size=m)  # Choose esk in Z_q^m uniformly at random
    A, t_A = GenTrap(n, q, B, np_rng)  # Apply the GenTrap procedure to get A and t_A
    sk = np.concatenate((-esk, [1]))  # The secret key is sk = (-esk, 1)

    # Compute A' as A and A.T * esk (mod q)
    A_prime = np.vstack((A, np.remainder(np.dot(A.T, esk), q)))

    # Convert A_prime to range (-q/2, q/2]
    A_prime = np.where(A_prime > q//2, A_prime - q, A_prime)

    return sk, t_A, A_prime


# TODO different from DualHE.py/TrapRecov
def TrapRecov(t_A, c, q, n):
    k = int(math.log2(q))
    m = 2 * n * k
    g = np.array([2**i for i in range(k)])
    G = np.kron(np.eye(n), g)
    # extract the first m entries of c as c_1
    c_1 = c[:m]
    recov = np.remainder(np.dot(t_A, c_1), q)
    recov = np.where(recov < 0, recov + q, recov)
    s = []
    print(f'recov: {recov}')
    for i in range(n):
        s_i = 0
        for j in range(k-1, -1, -1):
            if (recov[i * k + j] - (2 ** (j)) * s_i) >= (q // 4) and (recov[i * k + j] - (2 ** j) * s_i) < (3 * q // 4):
            # if (recov[i * k + j] - (2 ** j) * s_i) >= (q // 4):
                s_i += 2**(k-1-j)

        s.append(s_i)
    s = np.array(s)
    s = np.where(s >= q//2, s - q, s)
    return s


def Dual_Enc(mu, A_prime, q, n, B, seed=None):
    np_rng = get_numpy_rng(seed)
    k = int(math.log2(q))
    m = 2 * n * k
    s = np_rng.integers(-q//2, q//2, size=n)  # Choose s in Z_q^n uniformly

    e = sample_D(q, B, size=m+1, seed=np_rng)  # Sample each entry of e from D_Zq, B

    # Compute A's * s + e + (0, ..., 0, mu * q / 2)
    encrypted = np.remainder(np.dot(A_prime, s) + e, q)
    encrypted[-1] += mu * (q // 2)

    # Convert to range (-q/2, q/2]
    encrypted = np.where(encrypted > q//2, encrypted - q, encrypted)

    return encrypted, s, e

def Dual_Dec(c, sk, q):
    # Compute b_prime = sk^T * c (mod q)
    b_prime = np.remainder(np.dot(sk, c), q)
    b_prime = b_prime - q if b_prime > q // 2 else b_prime
    # print(f'b_prime after adjustment: {b_prime}')


    # Output 0 if b_prime is closer to 0 than to q/2 mod q, otherwise output 1
    return 0 if abs(b_prime) < q // 4 else 1

def Dual_Add(c1, c2, q):
    return np.remainder(c1 + c2, q)

def Dual_Mult(c1, c2, q):
    return np.remainder(c1 * c2, q)
