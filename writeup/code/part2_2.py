mean_a = Points_a.mean(axis=0)
mean_b = Points_b.mean(axis=0)
std_a = np.sqrt(np.mean(np.sum((Points_a-mean_a)**2, axis=1), axis=0))
std_b = np.sqrt(np.mean(np.sum((Points_b-mean_b)**2, axis=1), axis=0))

Ta1 = np.diagflat(np.array([np.sqrt(2)/std_a, np.sqrt(2)/std_a, 1]))
Ta2 = np.column_stack((np.row_stack((np.eye(2), [[0, 0]])), [-mean_a[0], -mean_a[1], 1]))

Tb1 = np.diagflat(np.array([np.sqrt(2)/std_b, np.sqrt(2)/std_b, 1]))
Tb2 = np.column_stack((np.row_stack((np.eye(2), [[0, 0]])), [-mean_b[0], -mean_b[1], 1]))

Ta = np.matmul(Ta1, Ta2)
Tb = np.matmul(Tb1, Tb2)

arr_a = np.column_stack((Points_a, [1]*Points_a.shape[0]))
arr_b = np.column_stack((Points_b, [1]*Points_b.shape[0]))

arr_a = np.matmul(Ta, arr_a.T)
arr_b = np.matmul(Tb, arr_b.T)

arr_a = arr_a.T
arr_b = arr_b.T

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

F_matrix = Tb.T @ F_matrix @ Ta