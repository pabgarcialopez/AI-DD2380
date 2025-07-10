# Transition matrix
import matrix_operations

A = matrix_operations.read_matrix()

# Emission matrix
B = matrix_operations.read_matrix()

# Initial distribution vector
pi = matrix_operations.read_matrix()

pi_A = matrix_operations.multiply_matrices(pi, A)

answer = matrix_operations.multiply_matrices(pi_A, B)

print(len(answer), len(answer[0]), sep=' ', end='')

for x in answer[0]:
    print(' ', end='')
    print(x, end='')