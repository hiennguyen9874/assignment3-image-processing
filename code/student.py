# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech

import numpy as np
from random import sample

# Returns the projection matrix for a given set of corresponding 2D and
# 3D points.
# 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
# 'Points_3D' is nx3 matrix of 3D coordinate of points in the world
# 'M' is the 3x4 projection matrix


def calculate_projection_matrix(Points_2D, Points_3D):
    # To solve for the projection matrix. You need to set up a system of
    # equations using the corresponding 2D and 3D points:
    #
    #                                                     [M11        [u1
    #                                                      M12         v1
    #                                                      M13         .
    #                                                      M14         .
    # [X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
    #  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
    #  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
    #  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn]         M31         .
    #                                                      M32         un
    #                                                      M33]        vn]
    #
    # Then you can solve this using least squares with the 'np.linalg.lstsq' operator.
    # Notice you obtain 2 equations for each corresponding 2D and 3D point
    # pair. To solve this, you need at least 6 point pairs. Note that we set
    # M34 = 1 in this scenario. If you instead choose to use SVD via np.linalg.svd, you should
    # not make this assumption and set up your matrices by following the
    # set of equations on the project page.
    #

    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    # Your total residual should be less than 1.

    arr = np.column_stack((Points_3D, [1]*Points_3D.shape[0]))
    A1 = np.concatenate((arr, np.zeros_like(arr)), axis=1).reshape((-1, 4))
    A2 = np.concatenate((np.zeros_like(arr), arr), axis=1).reshape((-1, 4))

    '''solution 1'''
    A3 = -np.multiply(Points_2D.reshape((-1, 1)).repeat(3,axis=1),Points_3D.repeat(2, axis=0))
    A = np.concatenate((A1, A2, A3), axis=1)
    b = Points_2D.reshape((-1, 1))
    M = np.append(np.linalg.lstsq(A, b)[0], [1]).reshape((3, 4))
    '''residual: 0.044534993949316135'''

    '''solution 2'''
    # A3 = -np.multiply(np.tile(Points_2D.reshape((-1, 1)), 4), arr.repeat(2, axis=0))
    # A = np.concatenate((A1, A2, A3), axis=1)
    # U, s, V = np.linalg.svd(A)
    # M = V[-1]
    # M = M.reshape((3, 4))
    '''residual: 0.044548941765576305'''

    return M

# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates

def compute_camera_center(M):
    # Replace this with the correct code
    # In the visualization you will see that this camera location is clearly
    # incorrect, placing it in the center of the room where it would not see all
    # of the points.
    # Center = np.array([1,1,1])
    Q = np.split(M, [3], axis=1)
    return np.squeeze(-np.matmul(np.linalg.inv(Q[0]), Q[1]))

# Returns the camera center matrix for a given projection matrix
# 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
# 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
# 'F_matrix' is 3x3 fundamental matrix

def estimate_fundamental_matrix(Points_a, Points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project
    #

    #                                              [f11
    # [u1u1' v1u1' u1' u1v1' v1v1' v1' u1 v1 1      f12     [0
    #  u2u2' v2v2' u2' u2v2' v2v2' v2' u2 v2 1      f13      0
    #  ...                                      *   ...  =  ...
    #  ...                                          ...     ...
    #  unun' vnun' un' unvn' vnvn' vn' un vn 1]     f32      0]
    #                                               f33]

    arr_a = np.column_stack((Points_a, [1]*Points_a.shape[0]))
    arr_b = np.column_stack((Points_b, [1]*Points_b.shape[0]))

    arr_a = np.tile(arr_a, 3)
    arr_b = arr_b.repeat(3, axis=1)
    A = np.multiply(arr_a, arr_b)

    '''Solve f from Af=0'''
    '''solution 1'''
    U, s, V = np.linalg.svd(A)
    F_matrix = V[-1]
    F_matrix = np.reshape(F_matrix, (3, 3))

    '''solution 2'''
    # b = A[:, 0].copy()
    # F_matrix = np.linalg.lstsq(A[:, 1:], -b)[0]
    # F_matrix = np.r_[1, F_matrix]
    # F_matrix = F_matrix.reshape((3, 3))

    '''Resolve det(F) = 0 constraint using SVD'''
    U, S, Vh = np.linalg.svd(F_matrix)
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh

    return F_matrix

# Returns the camera center matrix for a given projection matrix
# 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
# 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
# 'F_matrix' is 3x3 fundamental matrix

def estimate_fundamental_matrix_with_normalize(Points_a, Points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project
    #

    #                                              [f11
    # [u1u1' v1u1' u1' u1v1' v1v1' v1' u1 v1 1      f12     [0
    #  u2u2' v2v2' u2' u2v2' v2v2' v2' u2 v2 1      f13      0
    #  ...                                      *   ...  =  ...
    #  ...                                          ...     ...
    #  unun' vnun' un' unvn' vnvn' vn' un vn 1]     f32      0]
    #                                               f33]

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
    '''solution 1'''
    U, s, V = np.linalg.svd(A)
    F_matrix = V[-1]
    F_matrix = np.reshape(F_matrix, (3, 3))

    '''solution 2'''
    # b = A[:, 0].copy()
    # F_matrix = np.linalg.lstsq(A[:, 1:], -b)[0]
    # F_matrix = np.r_[1, F_matrix]
    # F_matrix = F_matrix.reshape((3, 3))

    '''Resolve det(F) = 0 constraint using SVD'''
    U, S, Vh = np.linalg.svd(F_matrix)
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh

    F_matrix = Tb.T @ F_matrix @ Ta

    return F_matrix


# Takes h, w to handle boundary conditions
def apply_positional_noise(points, h, w, interval=3, ratio=0.2):
    """ 
    The goal of this function to randomly perturbe the percentage of points given 
    by ratio. This can be done by using numpy functions. Essentially, the given 
    ratio of points should have some number from [-interval, interval] added to
    the point. Make sure to account for the points not going over the image 
    boundary by using np.clip and the (h,w) of the image. 

    Key functions include but are not limited to:
        - np.random.rand
        - np.clip

    Arugments:
        points :: numpy array 
            - shape: [num_points, 2] ( note that it is <x,y> )
            - desc: points for the image in an array
        h :: int 
            - desc: height of the image - for clipping the points between 0, h
        w :: int 
            - desc: width of the image - for clipping the points between 0, w
        interval :: int 
            - desc: this should be the range from which you decide how much to
            tweak each point. i.e if interval = 3, you should sample from [-3,3]
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will have some number from 
            [-interval, interval] added to the point. 
    """
    num_noise = int(points.shape[0] * ratio)
    noise = np.concatenate((np.random.rand(num_noise, 2)*2*interval - interval,
                            np.zeros((points.shape[0]-num_noise, 2))), axis=0)
    np.random.shuffle(noise)
    points = points + noise
    points[:, 0] = np.clip(points[:, 0], 0, w)
    points[:, 1] = np.clip(points[:, 1], 0, h)
    return points

# Apply noise to the matches.
def apply_matching_noise(points, ratio=0.2):
    """ 
    The goal of this function to randomly shuffle the percentage of points given 
    by ratio. This can be done by using numpy functions. 

    Key functions include but are not limited to:
        - np.random.rand
        - np.random.shuffle  

    Arugments:
        points :: numpy array 
            - shape: [num_points, 2] 
            - desc: points for the image in an array
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will be randomly shuffled.
    """
    num_noise = int(points.shape[0] * ratio)
    temp = np.concatenate((np.ones(num_noise), np.zeros(points.shape[0]-num_noise)), axis=0)
    np.random.shuffle(temp)

    temp1 = points.copy()[temp == 1]
    np.random.shuffle(temp1)

    for i in range(num_noise):
        points[np.argwhere(temp == 1)[i]] = temp1[i]
    return points


# Find the best fundamental matrix using RANSAC on potentially matching
# points
# 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
# matching points from pic_a and pic_b. Each row is a correspondence (e.g.
# row 42 of matches_a is a point that corresponds to row 42 of matches_b.
# 'Best_Fmatrix' is the 3x3 fundamental matrix
# 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
# of 'matches_a' and 'matches_b') that are inliers with respect to
# Best_Fmatrix.
def ransac_fundamental_matrix(matches_a, matches_b):
    # For this section, use RANSAC to find the best fundamental matrix by
    # randomly sampling interest points. You would reuse
    # estimate_fundamental_matrix() from part 2 of this assignment.
    # If you are trying to produce an uncluttered visualization of epipolar
    # lines, you may want to return no more than 30 points for either left or
    # right images.

    # Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.

    num_iterator = 1500
    threshold = 0.005
    best_F_matrix = np.zeros((3, 3))
    max_inlier = 0
    num_sample_rand = 9

    xa = np.column_stack((matches_a, [1]*matches_a.shape[0]))
    xb = np.column_stack((matches_b, [1]*matches_b.shape[0]))
    xa = np.tile(xa, 3)
    xb = xb.repeat(3, axis=1)
    A = np.multiply(xa, xb)

    for i in range(num_iterator):
        index_rand = np.random.randint(matches_a.shape[0], size=num_sample_rand)
        F_matrix = estimate_fundamental_matrix_with_normalize(matches_a[index_rand, :], matches_b[index_rand, :])
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
