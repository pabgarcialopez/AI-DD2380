import matrix_operations
import forward


A = matrix_operations.read_matrix()
B = matrix_operations.read_matrix()
pi = matrix_operations.read_matrix()
observations = matrix_operations.read_vector()
print(forward.forward_alg(A, B, pi, observations), end='')
