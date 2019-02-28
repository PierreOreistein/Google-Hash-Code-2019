from munkres import Munkres, print_matrix

# Matrix of costs
matrix = [[5, 9, 1],
          [10, 3, 2],
          [8, 7, 4]]

# Instantiate a Munkres object
m = Munkres()
indexes = m.compute(matrix)

print_matrix(matrix, msg='Lowest cost through this matrix:')

total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print(f'({row}, {column}) -> {value}')

print(f'total cost: {total}')
