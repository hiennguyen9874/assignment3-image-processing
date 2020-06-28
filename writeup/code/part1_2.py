arr = np.column_stack((Points_3D, [1]*Points_3D.shape[0]))

A1 = np.concatenate((arr, np.zeros_like(arr)),
                    axis=1).reshape((-1, 4))
A2 = np.concatenate((np.zeros_like(arr), arr), 
                    axis=1).reshape((-1, 4))
A3 = -np.multiply(np.tile(Points_2D.reshape((-1, 1)), 4),
                  arr.repeat(2, axis=0))

A = np.concatenate((A1, A2, A3), axis=1)
U, s, V = np.linalg.svd(A)
M = V[-1]
M = M.reshape((3, 4))