import numpy as np
import math

from ._common import get_numpy_rng, GenTrap, sample_D


def DualHE_KeyGen(n, q, B, seed=None):
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


# TODO different from Dual.py/TrapRecov
def TrapRecov(t_A, c, q, n):
    k = int(math.log2(q))
    m = 2 * n * k
    g = np.array([2**i for i in range(k)])
    G = np.kron(np.eye(n), g)
    # extract the first m entries of c as c_1
    c_1 = c[:m]
    recov = np.remainder(np.dot(t_A, c_1), q)
    # recov = np.where(recov > q//2, recov - q, recov)
    s = []
    print(f'recov: {recov}')
    for i in range(n):
        s_i = 0
        for j in range(k-1, -1, -1):
            if (recov[i * k + j] - (2 ** j) * s_i) > (q // 4) and (recov[i * k + j] - (2 ** j) * s_i) <= (3 * q // 4):
                s_i += 2**(k-1-j)

        s.append(s_i)
    s = np.array(s)
    s = np.where(s > q//2, s - q, s)
    return s


def DualHE_Enc(mu, A_prime, q, n, B, seed=None):
    np_rng = get_numpy_rng(seed)
    k = int(math.log2(q))
    m = 2 * n * k
    N = (m + 1) * k

    g = np.array([2**i for i in range(k)])
    G = np.kron(np.eye(m+1), g)

    S = np_rng.integers(-q//2, q//2, size=[n, N])
    E = sample_D(q, B, size=[(m+1), N], seed=np_rng)

    encrypted = np.remainder(np.dot(A_prime, S) + E + mu * G, q)
    encrypted = np.where(encrypted > q//2, encrypted - q, encrypted)
    # force the encrypted be numpy integer
    encrypted = encrypted.astype(int)
    return encrypted, S, E

def DualHE_Convert(C, q, n):
    k = int(math.log2(q))
    m = 2 * n * k
    N = (m + 1) * k
    # output column N of C
    convert_c = C[:, N-1]
    return convert_c

def DualHE_Dec(c, sk, q):
    b_prime = np.remainder(np.dot(sk, c), q)
    b_prime = np.where(b_prime > q//2, b_prime - q, b_prime)


    # Output 0 if b_prime is closer to 0 than to q/2 mod q, otherwise output 1
    return 0 if abs(b_prime) < q // 4 else 1

def G_inv(matrix, q):
        # Ensure q is an integer greater than 1
    if not (isinstance(q, int) and q > 1):
        raise ValueError("q must be an integer greater than 1.")

    # Calculate the number of bits needed for binary representation
    num_bits = int(math.log2(q))

    # Get the dimensions of the input matrix
    m, n = matrix.shape

    # Initialize the output matrix with zeros - it will have m * log2(q) rows and n columns
    binary_matrix = np.zeros((m * num_bits, n), dtype=np.int64)

    # Process each element and place its binary representation in the correct position
    for i in range(m):
        for j in range(n):
            # Get the binary representation, with padding if necessary
            if matrix[i, j] < 0:
                  binary_repr = np.binary_repr(-matrix[i, j], width=num_bits)
                  # Assign the binary representation to the correct slice of the output matrix
                  binary_matrix[i * num_bits:(i + 1) * num_bits, j] = list(map(int, binary_repr))
                  binary_matrix[i * num_bits:(i + 1) * num_bits, j] = -binary_matrix[i * num_bits:(i + 1) * num_bits, j]
            else:
                  binary_repr = np.binary_repr(matrix[i, j], width=num_bits)
                  # Assign the binary representation to the correct slice of the output matrix
                  binary_matrix[i * num_bits:(i + 1) * num_bits, j] = list(map(int, binary_repr))

    return binary_matrix

def DualHE_Eval(C_0, C_1, q, n):
    # apply NANA(C_0, C_1)
    k = int(math.log2(q))
    m = 2 * n * k
    N = (m + 1) * k

    g = np.array([2**i for i in range(k)])
    G = np.kron(np.eye(m+1), g)

    # G - C_0 * G^{-1} * C_1
    C = np.remainder(G - np.dot(C_0, G_inv(C_1, q)), q)
    C = np.where(C > q//2, C - q, C)
    return C
