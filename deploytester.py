#!/usr/bin/env python

import time
import glob
# working VM library
from VMheader import VectorFormatter, BinCompletion

true_positives = []
false_positives = []
true_negatives = []
false_negatives = []


def evaluate_match(truth, guess):
    if truth != guess:
        return 1
    else:
        return 0


def deg_to_dms(deg):  # degrees, minutes, seconds. Returns string
    d = int(deg)
    md = abs(deg - d) * 60
    m = float(md)
    return str(d) + str(m)


def flatten(l):
    return [item for sublist in l for item in sublist]


def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    # assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def main():
    sat_crop = '..\Vermont_sim_data3\ cropped_600\ '

    # Set up for Vector Construction

    cropped_image_vector = []
    cropped_true_match = []
    cropped_imdist = []
    cropped_imangle = []
    cropped_feature_class = []
    cropped_feature_closest = []
    for frame_name in glob.glob(sat_crop + '*.jpg', recursive=True):

        label_name = frame_name[:-4] + '.txt'
        satellite = VectorFormatter('satellite')

        satellite.yolo_initialize(label_name, frame_name)

        if len(satellite.p_x) > neighbor:
            satellite.k_d_tree_test(neighbor)
            cropped_true_match.append(satellite.true_match)
            cropped_imdist.append(satellite.im_dist)
            cropped_imangle.append(satellite.im_angle)
            cropped_feature_closest.append(satellite.feature_closest)
            cropped_feature_class.append(satellite.feature_class)

    binn = BinCompletion('binn')
    binn.bin_initialize(flatten(cropped_imdist), flatten(cropped_imangle), .8, .8)
    for i in range(len(cropped_true_match)):
        vector_sat_all = []
        for j in range(len(cropped_feature_class[i])):
            vector_sat_all.append(
                binn.vector_def(j, cropped_feature_class[i], cropped_imdist[i], cropped_imangle[i],
                                cropped_feature_closest[i])[0])
        cropped_image_vector.append(vector_sat_all)

    test_image_dir = '..\Vermont_sim_data2\ cropped_800\ '
    distance_ratings = []
    hammingtotal = []
    features_detected = []
    times = []
    for frame_name in glob.glob(test_image_dir + '*.jpg', recursive=True):
        start = time.time()
        test_label_dir = frame_name[:-4] + '.txt'
        drone = VectorFormatter('drone')
        drone.yolo_initialize(test_label_dir, frame_name)
        features_detected.append(drone.true_match)
        if len(drone.p_x) > neighbor + 1:
            drone.k_d_tree_test(neighbor)
            vector_drone = []

            for i in range(len(drone.feature_class)):
                vector_drone.append(
                    binn.vector_def(i, drone.feature_class, drone.im_dist, drone.im_angle,
                                    drone.feature_closest)[0])

            cropped_truth = flatten(cropped_true_match)
            vector_sat_all = flatten(cropped_image_vector)

            distance = []
            for i in range(len(vector_drone)):

                last_rating = threshold

                for j in range(len(vector_sat_all)):
                    # if vector_drone[i][0:12] == vector_sat_all[j][0:12]:
                    # compair drone to sat, keep only features of the same classes
                    rating = hamming2(vector_drone[i], vector_sat_all[j])

                    if rating <= last_rating:
                        distance = [rating, cropped_truth[j], drone.true_match[i]]
                        last_rating = rating
                hammingtotal.append(last_rating)
                distance_ratings.append(distance)

            times.append(time.time() - start)
        # print(distance_ratings)
        # print(len(binn.min_ind_dist), len(binn.min_ind_angle))
    # print(len(distance_ratings))
    if len(distance_ratings) >= 1:
        tp = 0
        fp = 0

        for i in distance_ratings:
            if len(i) > 2:
                if evaluate_match(i[1], i[2]) == 0:
                    tp = tp + 1
                else:
                    fp = fp + 1
        fn = len(flatten(features_detected)) - tp - fp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)  # also known as senitivity
        # false_positive_rate = fp/(0+fp)

        average_time = sum(times) / len(times)
        fps = 1 / average_time


        if (tp != 0 or fp != 0):
            print(neighbor, threshold, precision, recall, tp, fp, fps)


if __name__ == "__main__":
    main()
