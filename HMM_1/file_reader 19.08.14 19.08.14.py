import matrix_operations


def forward_alg(A, B, pi, obs):
    # Number of states (N) and time steps (T)
    N, T = len(A), len(obs)

    # Dynamic programming table of size NxT.
    alpha = [[0 for _ in range(T)] for _ in range(N)]

    # Initialize dp (first column: state i, time 0)
    for i in range(N):
        alpha[i][0] = pi[0][i] * B[i][int(obs[0])]

    # Fill rest of the table
    for t in range(1, T):  # Time t = 1, ..., T - 1
        for i in range(N):  # State i = 0, ..., N - 1

            summation = 0
            for j in range(N):  # State j = 0, ..., N - 1
                summation += alpha[j][t - 1] * A[j][i]

            # Save value in our table at state i, time t
            alpha[i][t] = summation * B[i][int(obs[t])]

    # Do final sum to obtain probability
    solution = 0
    for i in range(N):
        solution += alpha[i][T - 1]

    return solution

try:

    with open('hmm1_01.in', 'r') as file:

        A = matrix_operations.read_matrix_from_file(file)
        B = matrix_operations.read_matrix_from_file(file)
        pi = matrix_operations.read_matrix_from_file(file)
        observations = matrix_operations.read_vector_from_file(file)
        print('%.6f' % forward_alg(A, B, pi, observations), end='')

except FileNotFoundError:
    raise FileNotFoundError("The specified file was not found.")
except ValueError as e:
    raise ValueError("Error while reading matrix data:", e)
