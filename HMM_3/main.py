import matrix_operations
import learning


A = matrix_operations.read_matrix()
B = matrix_operations.read_matrix()
pi = matrix_operations.read_matrix()
obs = matrix_operations.read_vector()

transition_estimation, emission_estimation, pi_estimation, convergence = learning.baum_welch_alg(A, B, pi, obs)

matrix_operations.print_matrix(transition_estimation)
matrix_operations.print_matrix(emission_estimation)