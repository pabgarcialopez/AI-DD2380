import matrix_operations
import viterbi


A = matrix_operations.read_matrix()
B = matrix_operations.read_matrix()
pi = matrix_operations.read_matrix()
observations = matrix_operations.read_vector()

# Maximum probability state sequence and the value of the prob.
probability, state_seq = viterbi.viterbi_alg(A, B, pi, observations)

for state in state_seq:
    print(state, end='')
    print(' ', end='')
