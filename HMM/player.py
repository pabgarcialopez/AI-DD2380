#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import math
import sys
import matrix_operations

MAX_ITERS = 5
epsilon = sys.float_info.epsilon


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
    c[0] = 1 / (c[0] + epsilon)
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
        c[t] = 1 / (c[t] + epsilon)
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

    # Re-estimate pi
    for i in range(N):
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
            A[i][j] = numer / (denom + epsilon)

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

            B[i][j] = numer / (denom + epsilon)


def compute_logP(c):
    logProb = 0
    for i in range(len(c)):
        logProb += math.log(c[i])

    return -logProb


def baum_welch_alg(A, B, pi, obs):
    iters = 0
    log_prob = 0
    old_log_prob = float('-inf')

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
        reestimate(A, B, pi, gamma, di_gamma, obs)

        # Compute log[P(O|lambda)]
        log_prob = compute_logP(c)

        iters += 1

    return A, B, pi


def forward_alg(model, obs_fish):
    """
    :param model: HMM model of a given species
    :param obs_fish: observations of a given fish
    """

    # For any model, A and pi are 1x1 matrices and B is a 1 x NUM_EMISSIONS matrix

    # Transpose emission matrix because it is more convenient.
    # Now we have a matrix with observations as rows, and states as columns.
    # In our specific case, emission_aux is a column vector (NUM_EMISSIONS x 1)
    emission_aux = matrix_operations.transpose(model.B)

    # Initialize first time step. (alpha turns out to be 1x1 matrix)
    alpha = matrix_operations.dot_prod(model.pi, emission_aux[obs_fish[0]])

    # Remaining observations
    for o in obs_fish[1:]:
        # Transition from one time step to the next.
        # Multiplication well-defined: alpha and model.A are 1x1
        alpha = matrix_operations.multiply_matrices(alpha, model.A)
        alpha = matrix_operations.dot_prod(alpha, emission_aux[o])

    # Return total probability
    return sum(alpha[0])


class HMM_Model:

    # N is the number of species
    # M is the number of emissions
    def __init__(self, N, M):
        self.A = matrix_operations.create_stochastic_matrix(N, N, 1)
        self.B = matrix_operations.create_stochastic_matrix(N, M, 1)
        self.pi = matrix_operations.create_stochastic_matrix(1, N, 1)

    def set_A(self, A):
        self.A = A

    def set_B(self, B):
        self.B = B

    def set_PI(self, pi):
        self.pi = pi


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def __init__(self):
        super().__init__()
        self.obs = None
        self.fish = None
        self.models = None

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """

        # For each species, we create a model, and we store all of them in a vector.
        self.models = [HMM_Model(1, N_EMISSIONS) for _ in range(N_SPECIES)]

        # fish is a vector thar contains tuples of the form (fish_index, observations_i)
        self.fish = [(i, []) for i in range(N_FISH)]

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        # Give each fish its observation.
        for i in range(len(self.fish)):
            self.fish[i][1].append(observations[i])

        # Decide if we're guessing or not.
        if step < 110:  # If we have already done 70 guesses, don't do more.
            return None
        else:

            # Get the last fish
            fish_id, obs = self.fish.pop()

            # Guess of the fish type.
            fish_type = 0

            # Max probability
            maximum = 0

            # Save these observations for later
            self.obs = obs

            # Pair up models with species and see which one returns highest prob.
            for fish_species in range(N_SPECIES):
                model = self.models[fish_species]
                prob = forward_alg(model, obs)
                if prob > maximum:
                    maximum = prob
                    fish_type = fish_species

            return fish_id, fish_type

    def reveal(self, correct, fish_id, true_type):
        """
        This method gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """

        # Update model if guess is not correct
        if not correct:
            A, B, pi = baum_welch_alg(self.models[true_type].A,
                                      self.models[true_type].B,
                                      self.models[true_type].pi,
                                      self.obs)

            self.models[true_type].set_A(A)
            self.models[true_type].set_B(B)
            self.models[true_type].set_PI(pi)
