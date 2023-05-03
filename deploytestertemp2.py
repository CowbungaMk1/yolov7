#!/usr/bin/env python

# Libraries for deploying CNN
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import glob
# working VM library
from VMheader import VectorFormatter, BinCompletion


def aflattenerthingy(px, py):
    x = [px, py]
    x = flatten(x)
    x = np.array(x)
    x = x.reshape((int(len(x) / 2), 2), order='F')
    return x


def acrop_2_sat_pos(positions, path_image, shift):
    path_image = path_image.split('/')
    path_image = path_image[-1].split('_')
    y_change = int(path_image[2])
    x_change = int(path_image[-1].split('.')[0])

    positions = positions + [x_change * shift, y_change * shift]

    return positions


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


def pred_2_list(prediction):
    myorder = [5, 0, 1, 2, 3]
    pred = prediction.tolist()
    idx = 0
    for vector in pred:
        pred[idx] = [vector[i] for i in myorder]
        idx += 1
    # print(pred)
    return pred


def main():
    # Paths for satellite weights and images
    weights = './satwts/yolov712/weights/best.pt'
    source = './satellite/cropped_300_scres/'

    # Storing values across all reviewed images (probably can go) -IP
    # These are used to perform the binning process across the whole thing
    vector_sat_total = []
    pos_in_sat = []
    cropped_true_match = []
    cropped_imdist = []
    cropped_imangle = []
    cropped_feature_class = []
    cropped_feature_closest = []

    # Initialize
    set_logging()
    device = select_device(device='0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = 640
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None

    webcam = False
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    old_img_w = old_img_h = imgsz

    # Generating dictionary of satellite image crops so that each classes information is saved for each image
    satellite_obj = {}
    for path, img, im0s, vid_cap in dataset:
        # print(path)
        sat_name =  'sat_{}'.format(path)
        satellite_obj[sat_name] = VectorFormatter('satellite')

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=True)[0]

        # Apply NMS, selecting only the classes that we want to use.
        pred = non_max_suppression(pred, .5, .45, classes=(0, 1, 2, 3), agnostic='store_true')
        # print(pred.size)
        pred = pred[0]

        pred = pred.detach().cpu().numpy()
        pred = pred_2_list(pred)
        satellite_obj[sat_name].cnn_initialization(pred, [old_img_h, old_img_w])
        neighbor = 5

        if len(satellite_obj[sat_name].p_x) > neighbor:
            satellite_obj[sat_name].k_d_tree_test(neighbor)
            cropped_imdist.append(satellite_obj[sat_name].im_dist)
            cropped_imangle.append(satellite_obj[sat_name].im_angle)
            cropped_feature_closest.append(satellite_obj[sat_name].feature_closest)
            cropped_feature_class.append(satellite_obj[sat_name].feature_class)
            pos_temp = aflattenerthingy(satellite_obj[sat_name].p_x, satellite_obj[sat_name].p_y)
            pos_in_sat.append(acrop_2_sat_pos(pos_temp, path, shift=100))

    # Performing binning process. Creates histogram of distribution for max vector definition effectiveness and stuff
    binn = BinCompletion('binn')
    binn.bin_initialize(flatten(cropped_imdist), flatten(cropped_imangle), .8, .8)
    for i in range(len(cropped_feature_class)):
        vector_sat_all = []
        for j in range(len(cropped_feature_class[i])):
            vector_sat_all.append(
                binn.vector_def(j, cropped_feature_class[i], cropped_imdist[i], cropped_imangle[i],
                                cropped_feature_closest[i])[0])
        vector_sat_total.append(vector_sat_all)
    vector_sat_total = flatten(vector_sat_total)
    pos_in_sat = np.array(pos_in_sat, dtype=object)
    pos_in_sat = flatten(pos_in_sat)
    pos_in_sat = np.array(pos_in_sat, dtype=object)

    print(pos_in_sat)

    # print(vector_sat_total)

    # start Drone Nural network

    # Paths for satellite weights and images
    weights = './satwts/yolov712/weights/best.pt'
    source = './satellite/cropped_300_scres/'

    # Initialize
    set_logging()
    device = select_device(device='0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = 640
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None

    webcam = False
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    old_img_w = old_img_h = imgsz

    hammingtotal = []

    for path, img, im0s, vid_cap in dataset:

        drone = VectorFormatter('drone')

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, .5, .45, agnostic='store_true')
        # print(pred.size)
        pred = pred[0]

        pred = pred.detach().cpu().numpy()
        pred = pred_2_list(pred)
        drone.cnn_initialization(pred, [old_img_h, old_img_w])

        vector_drone = []

        # Checks if enough features are even detected to create a correspondence, the plus 1 is for the humvee
        if len(drone.p_x) > neighbor + 1:

            # if drone.humvee_detected:  # If humvee is detected move on, otherwise go back

            drone.k_d_tree_test(neighbor)

            for i in range(len(drone.feature_class)):
                vector_drone.append(
                    binn.vector_def(i, drone.feature_class, drone.im_dist, drone.im_angle,
                                    drone.feature_closest)[0])

        threshold = 2
        matched_idx = np.zeros(shape=(len(vector_drone), 2))  # stored indexes [[idx sat, idx drone],.....,]

        for i in range(len(vector_drone)):
            best_rating = threshold

            for name in satellite_obj:

                for k in satellite_obj[name].feature_vectors:

                    # if vector_drone[i][0:12] == vector_sat_total[j][0:12]:
                    # compair drone to sat, keep only features of the same classes
                    rating = hamming2(vector_drone[i], vector_sat_total[j])

                    if rating <= best_rating:
                        # print(rating,vector_drone[i],vector_sat_total[j])
                        best_rating = rating
                        matched_idx[i] = [i, j]
            print(matched_idx[i])

        # print(matched_idx)


if __name__ == "__main__":
    with torch.no_grad():
        main()
