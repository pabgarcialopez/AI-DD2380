import sys

import numpy as np
from settings import MAX_ITERS, CONVERGENCE_CRITERIA

eps = 0 #sys.float_info.epsilon

def alpha_pass(A, B, pi, obs):
    N, T = len(A), len(obs)
    alpha = [[float(0) for _ in range(T)] for _ in range(N)]

    # Scaling vector
    c = [float(0) for _ in range(len(obs))]

    # Compute alpha[i][0] (time 0, state i)
    for i in range(N):
        alpha[i][0] = pi[0][i] * B[i][obs[0]]
        c[0] += alpha[i][0]

    # Scale alpha[i][0]
    c[0] = 1 / c[0]
    for i in range(N):
        alpha[i][0] *= c[0]

    # Compute a[i][t]
    for t in range(1, T):  # t = 1, ..., T - 1
        c[t] = 0
        for i in range(N):  # i = 0, ..., N - 1
            alpha[i][t] = 0
            for j in range(N):  # j = 0, ..., N - 1
                alpha[i][t] += alpha[j][t - 1] * A[j][i]

            alpha[i][t] *= B[i][obs[t]]
            c[t] += alpha[i][t]

        # Scale alpha[i][t]
        c[t] = 1 / c[t]
        for i in range(N):  # i = 0, ..., N - 1
            alpha[i][t] *= c[t]

    prob = 0
    for i in range(N):
        prob += alpha[i][T - 1]

    return alpha, prob, c


def beta_pass(A, B, obs, c):
    N, T = len(A), len(obs)
    beta = [[float(0) for _ in range(T)] for _ in range(N)]

    # Let beta[i][T-1] = 1, scaled by c[T-1]
    for i in range(N):
        beta[i][T - 1] = c[T - 1]

    # beta-pass
    for t in range(T - 2, -1, -1):  # t = T - 2, ..., 0
        for i in range(N):  # i = 0, ..., N - 1
            beta[i][t] = 0
            for j in range(N):  # j = 0, ..., N - 1
                beta[i][t] += A[i][j] * B[j][obs[t + 1]] * beta[j][t + 1]

            # Scale beta[i][t] with same scale factor as alpha[i][t]
            beta[i][t] *= c[t]

    return beta


def compute_gammas(A, B, alpha, beta, obs):
    N, T = len(A), len(obs)

    gamma = [[float(0) for _ in range(T)] for _ in range(N)]
    di_gamma = [[[float(0) for _ in range(T - 1)] for _ in range(N)] for _ in range(N)]

    # No need to normalize di_gamma, since alpha and beta are scaled
    for t in range(T - 1):  # t = 0, ..., T - 2
        for i in range(N):  # i = 0, ..., N - 1
            gamma[i][t] = 0
            for j in range(N):  # j = 0, ..., N - 1
                di_gamma[i][j][t] = alpha[i][t] * A[i][j] * B[j][obs[t + 1]] * beta[j][t + 1]
                gamma[i][t] += di_gamma[i][j][t]

    # Special case for gamma[i][T - 1] (no need to normalize)
    for i in range(N):  # i = 0, ..., N - 1
        gamma[i][T - 1] = alpha[i][T - 1]

    return di_gamma, gamma


def reestimate(A, B, pi, gamma, di_gamma, obs):
    N, M, T = len(A), len(B[0]), len(obs)

    convergence = True

    # Re-estimate pi
    for i in range(N):
        if abs(pi[0][i] - gamma[i][0] > CONVERGENCE_CRITERIA):
            convergence = False
        pi[0][i] = gamma[i][0]

    # Re-estimate A
    # numer = expected number of transitions from state qi to state qj
    # denom = expected number of transitions from qi to any state
    for i in range(N):
        denom = 0
        for t in range(T - 1):
            denom += gamma[i][t]
        for j in range(N):
            numer = 0
            for t in range(T - 1):
                numer += di_gamma[i][j][t]

            denom += eps

            if abs(A[i][j] - numer / denom > CONVERGENCE_CRITERIA):
                convergence = False

            A[i][j] = numer / denom

    # Re-estimate B
    for i in range(N):
        denom = 0
        for t in range(T):
            denom += gamma[i][t]
        for j in range(M):
            numer = 0
            for t in range(T):
                if obs[t] == j:
                    numer += gamma[i][t]

            denom += eps

            if abs(B[i][j] - numer / denom > CONVERGENCE_CRITERIA):
                convergence = False

            B[i][j] = numer / denom

    return convergence


def compute_logP(c):
    logProb = 0
    for i in range(len(c)):
        logProb += np.log(c[i])

    return -logProb


def baum_welch_alg(A, B, pi, obs):
    iters = 0
    log_prob = 0
    old_log_prob = float('-inf')

    convergence = True

    while iters < MAX_ITERS and log_prob > old_log_prob:

        if iters != 0:  # So that after first iteration, old_log_prob = -inf still.
            old_log_prob = log_prob

        # alpha-pass (c is mutable)
        alpha, prob, c = alpha_pass(A, B, pi, obs)

        # beta-pass (c is only used)
        beta = beta_pass(A, B, obs, c)

        # Compute di_gamma and gamma
        di_gamma, gamma = compute_gammas(A, B, alpha, beta, obs)

        # Re-estimate A, B and pi
        convergence = reestimate(A, B, pi, gamma, di_gamma, obs)

        # Compute log[P(O|lambda)]
        log_prob = compute_logP(c)

        iters += 1

    return A, B, pi, convergence
