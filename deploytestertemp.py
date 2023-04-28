#!/usr/bin/env python

# Libraries for deploying CNN
import time
from pathlib import Path

import cv2
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
        idx +=1
    print(pred)
    return pred

def main():

    # Paths for satellite weights and images
    weights = './satwts/yolov712/weights/best.pt'
    source = './satellite/cropped_300_scres/'
    
    # Storing values across all reviewed images (porbably can go) -IP
    cropped_image_vector = []
    cropped_true_match = []
    cropped_imdist = []
    cropped_imangle = []
    cropped_feature_class = []
    cropped_feature_closest = []
    
    # Initialize
    set_logging()
    device = select_device(device = '0')
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

    for path, img, im0s, vid_cap in dataset:


        satellite = VectorFormatter('satellite')
        


        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, .5, .45, agnostic='store_true')
        # print(pred.size)
        pred = pred[0]
        
        pred=pred.detach().cpu().numpy()
        pred = pred_2_list(pred)
        satellite.cnn_initialization(pred, [old_img_h,old_img_w])
        neighbor = 4

        if len(satellite.p_x) > neighbor :
            satellite.k_d_tree_test(neighbor)
            cropped_true_match.append(satellite.true_match)
            cropped_imdist.append(satellite.im_dist)
            cropped_imangle.append(satellite.im_angle)
            cropped_feature_closest.append(satellite.feature_closest)
            cropped_feature_class.append(satellite.feature_class)

    # Performing binning process. Creates histogram of distribution for max vector definition effectiveness and stuff
    binn = BinCompletion('binn')
    binn.bin_initialize(flatten(cropped_imdist), flatten(cropped_imangle), .8, .8)
    for i in range(len(cropped_true_match)):
        vector_sat_all = []
        for j in range(len(cropped_feature_class[i])):
            vector_sat_all.append(
                binn.vector_def(j, cropped_feature_class[i], cropped_imdist[i], cropped_imangle[i],
                                cropped_feature_closest[i])[0])
        cropped_image_vector.append(vector_sat_all)

    # print(cropped_image_vector)
    
    
    # #start Drone Nural network
    #
    # # Paths for satellite weights and images
    # weights = './satwts/yolov712/weights/best.pt'
    # source = './satellite/cropped_300_scres/'
    #
    # # Initialize
    # set_logging()
    # device = select_device(device='0')
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    #
    # # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # stride = int(model.stride.max())  # model stride
    # imgsz = 640
    # imgsz = check_img_size(imgsz, s=stride)  # check img_size
    #
    #
    # if half:
    #     model.half()  # to FP16
    #
    # # Set Dataloader
    # vid_path, vid_writer = None, None
    #
    # webcam = False
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    #     dataset = LoadImages(source, img_size=imgsz, stride=stride)
    #
    #
    # # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    #
    # old_img_w = old_img_h = imgsz
    #
    # distance_ratings = []
    # hammingtotal = []
    # features_detected = []
    # times = []
    #
    # for path, img, im0s, vid_cap in dataset:
    #
    #     drone = VectorFormatter('drone')
    #
    #     img = torch.from_numpy(img).to(device)
    #     img = img.half() if half else img.float()  # uint8 to fp16/32
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    #
    #     # Inference
    #     with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
    #         pred = model(img, augment=True)[0]
    #
    #     # Apply NMS
    #     pred = non_max_suppression(pred, .5, .45, agnostic='store_true')
    #     # print(pred.size)
    #     pred = pred[0]
    #
    #     pred = pred.detach().cpu().numpy()
    #     pred = pred_2_list(pred)
    #     drone.cnn_initialization(pred, [old_img_h, old_img_w])
    #     neighbor = 4
    #
    #     # Checks if enough features are even detected to create a correspondence
    #     if len(drone.p_x) > neighbor:
    #         drone.k_d_tree_test(neighbor)
    #         cropped_true_match.append(drone.true_match)
    #         cropped_imdist.append(drone.im_dist)
    #         cropped_imangle.append(drone.im_angle)
    #         cropped_feature_closest.append(drone.feature_closest)
    #         cropped_feature_class.append(drone.feature_class)


    # for frame_name in glob.glob(test_image_dir + '*.jpg', recursive=True):
    #     start = time.time()
    #     test_label_dir = frame_name[:-4] + '.txt'
    #     drone = VectorFormatter('drone')
    #     drone.yolo_initialize(test_label_dir, frame_name)
    #     features_detected.append(drone.true_match)
    #     if len(drone.p_x) > neighbor + 1:
    #         drone.k_d_tree_test(neighbor)
    #         vector_drone = []
    #
    #         for i in range(len(drone.feature_class)):
    #             vector_drone.append(
    #                 binn.vector_def(i, drone.feature_class, drone.im_dist, drone.im_angle,
    #                                 drone.feature_closest)[0])
    #
    #         cropped_truth = flatten(cropped_true_match)
    #         vector_sat_all = flatten(cropped_image_vector)
    #
    #         distance = []
    #         for i in range(len(vector_drone)):
    #
    #             last_rating = threshold
    #
    #             for j in range(len(vector_sat_all)):
    #                 # if vector_drone[i][0:12] == vector_sat_all[j][0:12]:
    #                 # compair drone to sat, keep only features of the same classes
    #                 rating = hamming2(vector_drone[i], vector_sat_all[j])
    #
    #                 if rating <= last_rating:
    #                     distance = [rating, cropped_truth[j], drone.true_match[i]]
    #                     last_rating = rating
    #             hammingtotal.append(last_rating)
    #             distance_ratings.append(distance)
    #
    #         times.append(time.time() - start)
 


if __name__ == "__main__":

    with torch.no_grad():

        main()
