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


def read_matrix():
    line = input()
    elements = line.strip().split()

    num_rows = int(elements[0])
    num_cols = int(elements[1])

    # Extract the matrix data
    matrix_data = [float(elements[i]) for i in range(2, 2 + num_rows * num_cols)]

    # Create the matrix
    matrix = []
    for i in range(num_rows):
        row = matrix_data[i * num_cols: (i + 1) * num_cols]
        matrix.append(row)

    return matrix


def read_matrix_from_file(file):
    # Read line from the file
    line = file.readline()

    # Split the line into individual elements
    elements = line.strip().split()

    # Extract the number of rows and columns
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


def read_vector_from_file(file):
    # Read line from the file
    line = file.readline()

    # Split the line into individual elements
    elements = line.strip().split()

    # Extract the number of rows and columns
    num_cols = int(elements[0])

    # Extract the matrix data
    vector_data = [float(elements[i]) for i in range(1, num_cols + 1)]

    # Create the matrix
    vector = []
    for i in range(num_cols):
        vector.append(vector_data[i])

    return vector


def read_vector():
    # Read line
    line = input()

    # Split the line into individual elements
    elements = line.strip().split()

    # Extract the number of rows and columns
    num_cols = int(elements[0])

    # Extract the matrix data
    vector_data = [float(elements[i]) for i in range(1, num_cols + 1)]

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


def print_matrix_debug(matrix):
    print(len(matrix), len(matrix[0]), sep=' ')

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(round(matrix[i][j], 6), end='')
            print(" ", end='')
        print()


def create_random_stochastic_matrix(N, M, rand):
    if N <= 0 or M <= 0:
        return []

    # Initialize the matrix with random values
    matrix = []
    for _ in range(N):

        if rand:
            row = [random.uniform(0, 1) for _ in range(M)]
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
