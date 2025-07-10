import matrix_operations
import learning


try:

    with open('tests/aux', 'r') as file:

        # N = 3
        # M = 4
        # T = 1000
        # rand = 0  # 0 if uniform, 1 if random
        #
        # print(f"Parameters: N = {N}, M = {M}, T = {T} [0, ..., {M - 1}]")

        # A = matrix_operations.create_random_stochastic_matrix(N, N, rand)
        # matrix_operations.print_matrix_debug(A)
        #
        # B = matrix_operations.create_random_stochastic_matrix(N, M, rand)
        # matrix_operations.print_matrix_debug(B)
        #
        # pi = matrix_operations.create_random_stochastic_matrix(1, N, rand)
        #
        # obs = matrix_operations.create_random_vector(T, M)

        A = matrix_operations.read_matrix_from_file(file)
        B = matrix_operations.read_matrix_from_file(file)
        pi = matrix_operations.read_matrix_from_file(file)
        obs = matrix_operations.read_vector_from_file(file)

        transition_estimation, emission_estimation, pi_estimation, convergence = learning.baum_welch_alg(A, B, pi, obs)

        if convergence:
            print(f"Algorithm converged with {learning.MAX_ITERS} iterations. (Convergence criteria: {learning.CONVERGENCE_CRITERIA})")
        else:
            print(f"Algorithm did not converge with {learning.MAX_ITERS} iterations. (Convergence criteria: {learning.CONVERGENCE_CRITERIA})")

        matrix_operations.print_matrix_debug(transition_estimation)
        matrix_operations.print_matrix_debug(emission_estimation)

        # Only for debugging purposes.
        matrix_operations.print_matrix_debug(pi_estimation)


except FileNotFoundError:
    raise FileNotFoundError("The specified file was not found.")
except ValueError as e:
    raise ValueError("Error while reading matrix data:", e)
