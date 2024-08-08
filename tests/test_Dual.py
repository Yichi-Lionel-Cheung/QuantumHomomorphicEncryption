import numpy as np
import math

import qhe


def test_trapdoor():
    np_rng = np.random.default_rng(233)
    n = 5
    q = 1024 * 32 # should be big enough
    k = int(math.log2(q))
    m = 2 * n * k
    B = 3
    for i in range(10):
        mu = 0
        sk, t_A, A_prime = qhe.Dual_KeyGen(n, q, B, np_rng)
        g = np.array([2**i for i in range(k)])
        G = np.kron(np.eye(n), g)
        c, s, e = qhe.Dual_Enc(mu, A_prime, q, n, B, np_rng)

        s_rec = qhe.Dual.TrapRecov(t_A, c, q, n)
        assert np.array_equal(s, s_rec), f'Trapdoor recovery failed: got {s_rec}, expected {s}'


def test_Dual_KeyGen_Enc_Dec():
    np_rng = np.random.default_rng(233)
    # Set parameters
    n = 5
    q = 1024
    k = int(math.log2(q))
    m = 2 * n * k
    B = 3

    for i in range(8):
        mu = np_rng.integers(0, 2)
        # Generate keys
        sk, t_A, A_prime = qhe.Dual_KeyGen(n, q, B, np_rng)
        print(f'sk * A_prime: {np.dot(sk, A_prime) % q}')
        print(f'sk: {sk}')
        print(f't_A: {t_A}')
        print(f'A_prime: {A_prime}')

        # Generate ciphertext
        c, s, e = qhe.Dual_Enc(mu, A_prime, q, n, B, np_rng)
        print(f'c: {c}')
        print(f'esk_mu: {np.remainder(np.dot(sk, c), q)}')

        # Decrypt ciphertext
        decrypted_mu = qhe.Dual_Dec(c, sk, q)
        print(f'decrypted_mu: {decrypted_mu}')

        # Check if decryption is correct
        assert decrypted_mu == mu, f'Decryption failed: got {decrypted_mu}, expected {mu}'
        print('Decryption successful!')

        c_1, _, _ = qhe.Dual_Enc(np_rng.integers(0, 2), A_prime, q, n, B, np_rng)
        print(f'c_1: {c_1}')
        c_2, _, _ = qhe.Dual_Enc(np_rng.integers(0, 2), A_prime, q, n, B, np_rng)
        print(f'c_2: {c_2}')
        c_3 = qhe.Dual_Add(c_1, c_2, q)
        print(f'c_3: {c_3}')
        print(f'c_3_decrypted: {qhe.Dual_Dec(c_3, sk, q)}')

        TrapRecov(t_A, c, q)
        print(f's: {s}')
        print(e)
        print(t_A)
        print(f't_A * e: {np.dot(t_A, e[:-1]) % q}')

test_Dual_KeyGen_Enc_Dec()