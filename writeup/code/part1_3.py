M_ = np.split(M, [3], axis=1)
C = np.squeeze(-np.matmul(np.linalg.inv(M_[0]), M_[1]))