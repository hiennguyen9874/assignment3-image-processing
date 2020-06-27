# Camera Calibration Stencil Code
# Transferred to python by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
#
# This script
# (1) Loads 2D and 3D data points and images
# (2) Calculates the projection matrix from those points    (you code this)
# (3) Computes the camera center from the projection matrix (you code this)
# (4) Estimates the fundamental matrix                      (you code this)
# (5) Adds noise to the points if asked                     (you code this)
# (6) Estimates the fundamental matrix using RANSAC         (you code this)
#     and filters away spurious matches                                    
# (7) Visualizes the F Matrix with homography rectification
#
# The relationship between coordinates in the world and coordinates in the
# image defines the camera calibration. See Szeliski 6.2, 6.3 for reference.
#
# 2 pairs of corresponding points files are provided
# Ground truth is provided for pts2d-norm-pic_a and pts3d-norm pair
# You need to report the values calculated from pts2d-pic_b and pts3d
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import os
import cv2
import argparse
from skimage import io 
from scipy import misc
from student import (calculate_projection_matrix, compute_camera_center,
                     estimate_fundamental_matrix, ransac_fundamental_matrix,
                     apply_positional_noise, apply_matching_noise)
from helpers import (evaluate_points, visualize_points, plot3dview,
                     draw_epipolar_lines, matchAndShowCorrespondence,
                     showCorrespondence, get_ground_truth)

from student import estimate_fundamental_matrix_with_normalize

def main(args):

    data_dir = os.path.dirname(__file__) + '/../data/'

    ########## Parts (1) through (2)
    Points_2D = np.loadtxt(data_dir + 'pts2d-norm-pic_a.txt')
    Points_3D = np.loadtxt(data_dir + 'pts3d-norm.txt')

    # (Optional) Run this once you have your code working
    # with the easier, normalized points above.
    if args.hard_points:
        Points_2D = np.loadtxt(data_dir + 'pts2d-pic_b.txt')
        Points_3D = np.loadtxt(data_dir + 'pts3d.txt')

    # Calculate the projection matrix given corresponding 2D and 3D points
    # !!! You will need to implement calculate_projection_matrix. !!!
    M = calculate_projection_matrix(Points_2D, Points_3D)
    print('The projection matrix is:\n {0}\n'.format(M))

    Projected_2D_Pts, Residual = evaluate_points(M, Points_2D, Points_3D)
    print('The total residual is:\n {0}\n'.format(Residual))

    if not args.no_vis:
        visualize_points(Points_2D, Projected_2D_Pts)

    # Calculate the camera center using the M found from previous step
    # !!! You will need to implement compute_camera_center. !!!
    Center = compute_camera_center(M)
    print('The estimated location of the camera is:\n {0}\n'.format(Center))

    if not args.no_vis:
        plot3dview(Points_3D, Center)

    ########## Part (3)
    Points_2D_pic_a = np.loadtxt(data_dir + 'pts2d-pic_a.txt')
    Points_2D_pic_b = np.loadtxt(data_dir + 'pts2d-pic_b.txt')

    ImgLeft = io.imread(data_dir + 'pic_a.jpg')
    ImgRight = io.imread(data_dir + 'pic_b.jpg')

    # Calculate the fundamental matrix given corresponding point pairs
    # !!! You will need to implement estimate_fundamental_matrix. !!!
    F_matrix = estimate_fundamental_matrix(Points_2D_pic_a, Points_2D_pic_b)

    # Draw the epipolar lines on the images
    if not args.no_vis:
        draw_epipolar_lines(F_matrix, ImgLeft, ImgRight, Points_2D_pic_a,
                            Points_2D_pic_b)

    ########## Parts (6) through (8)
    # This Mount Rushmore pair is easy. Most of the initial matches are
    # correct. The base fundamental matrix estimation without coordinate
    # normalization will work fine with RANSAC.
    print("Using image: ", args.image)
    if args.image == "mt_rushmore":
        pic_a = io.imread(data_dir + 'MountRushmore/Mount_Rushmore1.jpg')
        pic_b = io.imread(data_dir + 'MountRushmore/Mount_Rushmore2.jpg')
        sf = pic_b.shape[0] / pic_a.shape[0]
        [Points_2D_pic_a, Points_2D_pic_b
         ] = get_ground_truth(data_dir + "MountRushmore/mt_rushmore.mat",
                              scale_factor_A=sf)
        pic_a = misc.imresize(pic_a, sf, interp='bilinear')

    # The Notre Dame pair is difficult because the keypoints are largely on the
    # same plane. Still, even an inaccurate fundamental matrix can do a pretty
    # good job of filtering spurious matches.
    elif args.image == "notre_dame":
        pic_a = io.imread(data_dir + 'NotreDame/NotreDame1.jpg')
        pic_b = io.imread(data_dir + 'NotreDame/NotreDame2.jpg')
        sf = pic_b.shape[0] / pic_a.shape[0]
        [Points_2D_pic_a, Points_2D_pic_b
         ] = get_ground_truth(data_dir + "NotreDame/notre_dame.mat",
                              scale_factor_A=sf)
        pic_a = misc.imresize(pic_a, sf, interp='bilinear')

    # The Gaudi pair doesn't find many correct matches unless you run at high
    # resolution, but that will lead to tens of thousands of ORB features
    # which will be somewhat slow to process. Normalizing the coordinates
    # seems to make this pair work much better.
    elif args.image == "gaudi":
        pic_a = io.imread(data_dir + 'EpiscopalGaudi/EGaudi_1.jpg')
        pic_b = io.imread(data_dir + 'EpiscopalGaudi/EGaudi_2.jpg')
        sf = pic_b.shape[0] / pic_a.shape[0]
        [Points_2D_pic_a, Points_2D_pic_b
         ] = get_ground_truth(data_dir + "EpiscopalGaudi/gaudi.mat",
                              scale_factor_A=sf)
        pic_a = misc.imresize(pic_a, sf, interp='bilinear')

    else:
        print("Error in argument passed for image: ", args.image)
        return

    if args.positional_ratio:
        print('Applying noise on position')
        Points_2D_pic_a = apply_positional_noise(Points_2D_pic_a,
                                                 pic_a.shape[0] * sf,
                                                 pic_a.shape[1] * sf,
                                                 args.positional_interval,
                                                 args.positional_ratio)
        Points_2D_pic_b = apply_positional_noise(Points_2D_pic_b,
                                                 pic_b.shape[0],
                                                 pic_b.shape[1],
                                                 args.positional_interval,
                                                 args.positional_ratio)
    if args.use_orb:
        print('Using ORB')
        # Finds matching points in the two images using opencv's implementation of
        # ORB. There can still be many spurious matches, though.
        [Points_2D_pic_a,
         Points_2D_pic_b] = matchAndShowCorrespondence(pic_a, pic_b)
        print('Found {0} possibly matching features using ORB\n'.format(
            Points_2D_pic_a.shape[0]))
    # Calculate the fundamental matrix using RANSAC
    # !!! You will need to implement ransac_fundamental_matrix. !!!

    if args.matching_ratio:
        print('Applying noise on matches')
        Points_2D_pic_a = apply_matching_noise(Points_2D_pic_a,
                                                args.matching_ratio)

    if args.no_ransac:
        print('Not using RANSAC, estimation using estimate_fundamental_matrix')
        F_matrix = estimate_fundamental_matrix(Points_2D_pic_a, Points_2D_pic_b)
        matched_points_a = Points_2D_pic_a
        matched_points_b = Points_2D_pic_b
    else:
        print('Running through RANSAC')
        [F_matrix, matched_points_a,
         matched_points_b] = ransac_fundamental_matrix(Points_2D_pic_a,
                                                       Points_2D_pic_b)

    # Visualizing the F matrix using homography rectification
    if not args.no_vis:
        H, _ = cv2.findHomography(matched_points_a, matched_points_b)
        pic_a = cv2.warpPerspective(pic_a, H, (pic_b.shape[1], pic_b.shape[0]))
        transformed_points_a = cv2.perspectiveTransform(
            matched_points_a.reshape(-1, 1, 2), H).squeeze(axis=1)
        showCorrespondence(pic_a, pic_b, transformed_points_a, matched_points_b)
        draw_epipolar_lines(F_matrix, pic_a, pic_b, transformed_points_a, matched_points_b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        help="Choose what image you'd like to run on: one of listed above",
        type=str,
        choices=['mt_rushmore', 'notre_dame', 'gaudi'],
        default='mt_rushmore')
    parser.add_argument("--positional-interval",
                        type=int,
                        default=3,
                        help="Add positional noise with the given interval, \
                                3 means [pos-3,pos+3]")
    parser.add_argument(
        "--positional-ratio",
        type=float,
        default=0.2,
        help="Add positional noise with the given ratio of the \
                                matches to perturbe. 0.2 means 20 percent are perturbed"
    )
    parser.add_argument("--matching-ratio",
                        type=float,
                        default=0.2,
                        help="Add matching noise with the given ratio of the \
                                matches to shuffle. 0.2 means 20 percent are shuffled"
                        )
    parser.add_argument(
        "--hard_points",
        help="Use the harder, unnormalized points, to be used after you have it \
                working with normalized points (for parts 1 and 2)",
        action="store_true")
    parser.add_argument("--no-ransac",
                        help="Disable using ransac for part 3",
                        action="store_true")
    parser.add_argument("--no-vis",
                        help="Disable visualization for parts 1 and 2",
                        action="store_true")
    parser.add_argument("--use-orb",
                        help="Use ORB from OpenCV to find points instead of \
                        ground truth",
                        action="store_true")
    arguments = parser.parse_args()
    main(arguments)
