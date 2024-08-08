import math
import numpy as np

import qhe


def test_DualHE_eval():
    np_rng = np.random.default_rng(234) #fail when seed=233
    repeat_time = 5

    n = 5
    q = 2**16
    k = int(math.log2(q))
    m = 2 * n * k
    B = 3

    for _ in range(repeat_time):
        mu_0 = np_rng.integers(0, 2)
        mu_1 = np_rng.integers(0, 2)
        # mu_0 = 1
        # mu_1 = 1
        print(f'mu_0: {mu_0}')
        print(f'mu_1: {mu_1}')
        sk, t_A, A_prime = qhe.DualHE_KeyGen(n, q, B, np_rng)
        C_0, S_0, E_0 = qhe.DualHE_Enc(mu_0, A_prime, q, n, B, np_rng)
        C_1, S_1, E_1 = qhe.DualHE_Enc(mu_1, A_prime, q, n, B, np_rng)

        C = qhe.DualHE_Eval(C_0, C_1, q, n)

        decrypted_mu = qhe.DualHE_Dec(qhe.DualHE_Convert(C, q, n), sk, q)
        print(f'decrypted_mu: {decrypted_mu}')
        assert decrypted_mu == (mu_0 & mu_1) ^ 1, f'Decryption failed: got {decrypted_mu}, expected {(mu_0 & mu_1) ^ 1}'


def test_DualHE_KeyGen_Enc_Dec():
    np_rng = np.random.default_rng(234)
    repeat_time = 5

    n = 5
    q = 1024
    k = int(math.log2(q))
    m = 2 * n * k
    B = 3
    for _ in range(repeat_time):
        # randomly pick a mu
        mu = np_rng.integers(0, 2)
        sk, t_A, A_prime = qhe.DualHE_KeyGen(n, q, B, np_rng)
        # print(f'sk * A_prime: {np.dot(sk, A_prime) % q}')
        # print(f'sk: {sk}')
        # print(f't_A: {t_A}')
        # print(f'A_prime: {A_prime}')

        C, S, E = qhe.DualHE_Enc(mu, A_prime, q, n, B, np_rng)
        # print(f'C: {C}')
        # print(f'S: {S}')

        c = qhe.DualHE_Convert(C, q, n)
        # print shape of c
        # print(f'c.shape: {c.shape}')
        # print(f'c: {c}')

        decrypted_mu = qhe.DualHE_Dec(c, sk, q)
        # print(f'decrypted_mu: {decrypted_mu}')
        assert decrypted_mu == mu, f'Decryption failed: got {decrypted_mu}, expected {mu}'
        print('Decryption successful!')

