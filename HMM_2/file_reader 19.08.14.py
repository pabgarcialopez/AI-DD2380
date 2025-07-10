import matrix_operations


def trace_back(A, B, obs, delta, most_likely_step):
    # Path of states
    state_path = []

    N, T = len(A), len(obs)

    first_arg_max = 0
    elem = float('-inf')

    # Search for argmax{delta[j][T-1] : j in {0, ..., N-1}}
    for i in range(N):
        if delta[i][T - 1] > elem:
            elem = delta[i][T - 1]
            first_arg_max = i

    state = first_arg_max
    # Insert at the beginning of list
    state_path.insert(0, state)

    # Count time from T - 2 to 0
    for time in range(T - 2, -1, -1):
        state = most_likely_step[state][time + 1]
        state_path.insert(0, state)

    return state_path


def viterbi_alg(A, B, pi, obs):
    # Number of states (N) and time steps (T)
    N, T = len(A), len(obs)

    # Dynamic programming table of size NxT.
    delta = [[0 for _ in range(T)] for _ in range(N)]

    # Matrix to store which was the most likely state - k - to preceed state i at time t
    # most_likely_step[i][t] = k
    most_likely_step = [[0 for _ in range(T)] for _ in range(N)]

    # Initialize dp (first column: state i, time 0)
    for i in range(N):
        delta[i][0] = pi[0][i] * B[i][int(obs[0])]

    # Fill rest of the table
    for t in range(1, T):  # Time t = 1, ..., T - 1
        for i in range(N):  # State i = 0, ..., N - 1

            maximum = float('-inf')
            arg_max = None

            for j in range(N):  # State j = 0, ..., N - 1
                elem = A[j][i] * delta[j][t - 1] * B[i][int(obs[t])]
                if elem > maximum:
                    maximum = elem
                    arg_max = j

            # Save value in our table at state i, time t
            delta[i][t] = maximum
            most_likely_step[i][t] = arg_max

    # Do final sum to obtain probability
    solution = float('-inf')
    for i in range(N):
        solution = max(solution, delta[i][T - 1])

    return solution, trace_back(A, B, obs, delta, most_likely_step)


try:

    with open('hmm2_01.in', 'r') as file:

        A = matrix_operations.read_matrix(file)
        B = matrix_operations.read_matrix(file)
        pi = matrix_operations.read_matrix(file)
        observations = matrix_operations.read_vector(file)

        # Maximum probability state sequence and the value of the prob.
        probability, state_seq = viterbi_alg(A, B, pi, observations)

        for state in state_seq:
            print(state, end='')
            print(' ', end='')

except FileNotFoundError:
    raise FileNotFoundError("The specified file was not found.")
except ValueError as e:
    raise ValueError("Error while reading matrix data:", e)
