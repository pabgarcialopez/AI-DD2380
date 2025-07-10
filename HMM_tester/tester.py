import matrix_operations
import algorithms
from settings import *

input_file = open("inputs/Q" + str(QUESTION_NUMBER) + "/test" + str(TEST_NUMBER), "r")
output_file = open("outputs/Q" + str(QUESTION_NUMBER) + "/test" + str(TEST_NUMBER), "w")

dist = ("Random distribution" if DISTRIBUTION == 1 else "Uniform distribution")
output_file.write(f"Parameters: N = {N}, M = {M}, T = {T}, {dist}\n\n")

A = matrix_operations.read_matrix(input_file)
B = matrix_operations.read_matrix(input_file)
pi = matrix_operations.read_matrix(input_file)
obs = matrix_operations.read_vector(input_file)

# A = matrix_operations.read_matrix(input_file)
# B = matrix_operations.read_matrix(input_file)
# pi = matrix_operations.read_matrix(input_file)
# obs = matrix_operations.read_vector(input_file)

output_file.write("Inputs:\n\n")

output_file.write(matrix_operations.matrix_to_string(A, "A"))
output_file.write(matrix_operations.matrix_to_string(B, "B"))
output_file.write(matrix_operations.matrix_to_string(pi, "pi"))
output_file.write(matrix_operations.vector_to_string(obs, "obs"))

output_file.write("\n\n")

error = False
convergence = None

try:

    A, B, pi, convergence = algorithms.baum_welch_alg(A, B, pi, obs)

except ZeroDivisionError:
    output_file.write("Division by zero error!")
    error = True

if not error:

    res = ""

    if convergence:
        res = "Algorithm converged with "
    else:
        res = "Algorithm did not converge with "

    output_file.write(
        res + str(MAX_ITERS) + " max iterations and convergence criteria of " + str(CONVERGENCE_CRITERIA) + "\n\n")

    output_file.write("Outputs:\n\n")

    output_file.write(matrix_operations.matrix_to_string(A, "A"))
    output_file.write(matrix_operations.matrix_to_string(B, "B"))
    output_file.write(matrix_operations.matrix_to_string(pi, "pi"))

    output_file.close()
