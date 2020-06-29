num_iterator = 1500
threshold = 0.005
best_F_matrix = np.zeros((3, 3))
max_inlier = 0
num_sample_rand = 8

xa = np.column_stack((matches_a, [1]*matches_a.shape[0]))
xb = np.column_stack((matches_b, [1]*matches_b.shape[0]))
xa = np.tile(xa, 3)
xb = xb.repeat(3, axis=1)
A = np.multiply(xa, xb)

for i in range(num_iterator):
    index_rand = np.random.randint(matches_a.shape[0], size=num_sample_rand)
    F_matrix = estimate_fundamental_matrix_with_normalize(matches_a[index_rand, :],
                 matches_b[index_rand, :])
    err = np.abs(np.matmul(A, F_matrix.reshape((-1))))
    current_inlier = np.sum(err <= threshold)
    if current_inlier > max_inlier:
        best_F_matrix = F_matrix
        max_inlier = current_inlier

err = np.abs(np.matmul(A, best_F_matrix.reshape((-1))))
index = np.argsort(err)
# print(best_F_matrix)
# print(np.sum(err <= threshold), "/", err.shape[0])
return best_F_matrix, matches_a[index[:34]], matches_b[index[:34]]