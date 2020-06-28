arr_a = np.column_stack((Points_a, [1]*Points_a.shape[0]))
arr_b = np.column_stack((Points_b, [1]*Points_b.shape[0]))

arr_a = np.tile(arr_a, 3)
arr_b = arr_b.repeat(3, axis=1)
A = np.multiply(arr_a, arr_b)

'''Solve f from Af=0'''
U, s, V = np.linalg.svd(A)
F_matrix = V[-1]
F_matrix = np.reshape(F_matrix, (3, 3))

'''Resolve det(F) = 0 constraint using SVD'''
U, S, Vh = np.linalg.svd(F_matrix)
S[-1] = 0
F_matrix = U @ np.diagflat(S) @ Vh