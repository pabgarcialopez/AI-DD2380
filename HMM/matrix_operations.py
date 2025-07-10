import random


def multiply_matrices(matrix1, matrix2):
    # Number of columns of matrix1 =? Number of rows of matrix2
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Matrix dimensions do not match for multiplication.")

    result = []

    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix2[0])):
            # Initialize the result element to 0
            element = 0
            for k in range(len(matrix2)):
                element += matrix1[i][k] * matrix2[k][j]
            row.append(element)
        result.append(row)

    return result


def read_matrix(file):
    if file is None:
        line = input()
    else:
        line = file.readline()

    elements = line.strip().split()

    num_rows = int(elements[0])
    num_cols = int(elements[1])

    # Check if there is enough data for the matrix
    if len(elements) != 2 + num_rows * num_cols:
        raise ValueError("Invalid input format: Data does not match specified dimensions.")

    # Extract the matrix data
    matrix_data = [float(elements[i]) for i in range(2, 2 + num_rows * num_cols)]

    # Create the matrix
    matrix = []
    for i in range(num_rows):
        row = matrix_data[i * num_cols: (i + 1) * num_cols]
        matrix.append(row)

    return matrix


def read_vector(file):
    if file is None:
        line = input()
    else:
        line = file.readline()

    # Split the line into individual elements
    elements = line.strip().split()

    # Extract the number of rows and columns
    num_cols = int(elements[0])

    # Extract the matrix data
    vector_data = [int(elements[i]) for i in range(1, num_cols + 1)]

    # Create the matrix
    vector = []
    for i in range(num_cols):
        vector.append(vector_data[i])

    return vector


def print_matrix(matrix):
    print(len(matrix), len(matrix[0]), sep=' ', end='')

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(' ', end='')
            print(matrix[i][j], end='')
    print()


def matrix_to_string(matrix, id):
    mat = id + "\n"

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            mat += str(round(matrix[i][j], 5)) + " "
        mat += "\n"

    return mat + "\n"


def vector_to_string(vect, id):
    return id + "\n" + str(vect)


def create_stochastic_matrix(N, M, dist):
    """
    param: N (Number of states)
    param: M (Number of observations)
    param: rand (0 if uniform, 1 if not)
    """

    matrix = []
    for _ in range(N):

        if dist:
            if N == M or N == 1:  # A and pi
                row = [(1 / N) + random.random() / 1000 for _ in range(M)]
            else:  # B
                row = [(1 / M) + random.random() / 1000 for _ in range(M)]

        else:  # If uniform dist
            if N == M or N == 1:  # Uniform dist for A and pi
                row = [1 / N for _ in range(M)]
            else:  # Uniform dist for B
                row = [1 / M for _ in range(M)]

        row_sum = sum(row)
        # Normalize the row so that it sums up to 1
        row = [x / row_sum for x in row]
        matrix.append(row)

    return matrix


def create_random_vector(T, M):
    return [random.randint(0, M - 1) for _ in range(T)]


def transpose(matrix):
    return [list(i) for i in zip(*matrix)]


def dot_prod(matrix_a, matrix_b):
    return [[a * b for a, b in zip(matrix_a[0], matrix_b)]]
