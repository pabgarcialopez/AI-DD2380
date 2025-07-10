import matrix_operations

try:

    # Open the file for reading
    with open('sample_00.in', 'r') as file:

        # Transition matrix
        A = matrix_operations.read_matrix(file)

        # Emission matrix
        B = matrix_operations.read_matrix_from_file(file)

        # Initial distribution vector
        pi = matrix_operations.read_matrix_from_file(file)

        pi_A = matrix_operations.multiply_matrices(pi, A)

        answer = matrix_operations.multiply_matrices(pi_A, B)

        print(len(answer), len(answer[0]), sep=' ', end='')

        for x in answer[0]:
            print(' ', end='')
            print('%.1f' % x, end='')

except FileNotFoundError:
    raise FileNotFoundError("The specified file was not found.")
except ValueError as e:
    raise ValueError("Error while reading matrix data:", e)