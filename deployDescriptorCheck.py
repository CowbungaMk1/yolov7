#!/usr/bin/env python

# Libraries for deploying CNN
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# working VM library
from VMheader import VectorFormatter, BinCompletion
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, set_logging, clip_coords
from utils.torch_utils import select_device
import cv2

"""

"""


# TODO try other sat nets and see if any look like humvyy results. Also try tiny humvee and see if that one works better, add more presepectives to sat to make up for mismatching bits
# function stolen from utils general

def resizeing_corrector(locations, origonal_dim, network_dim):
    return locations


def scale_coords(img1_shape, pred_list, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    # print(img1_shape)
    # print(img0_shape)
    # print(pad)
    # print((gain))
    pred = []
    for coords in pred_list:
        coords[1] = round((coords[1] - pad[0]) / gain)  # x padding
        coords[3] = round((coords[3] - pad[0]) / gain)  # x padding
        coords[2] = round((coords[2] - pad[1]) / gain)  # y padding
        coords[4] = round((coords[4] - pad[1]) / gain)  # y padding
        pred.append(coords)

    return pred


def aflattenerthingy(px, py):
    # x = list(zip(px,py))
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


def pred_2_list(pred):
    myorder = [5, 0, 1, 2, 3]
    pred = pred.tolist()
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

    # source = './satellite/cropped_300_sbts/'
    # source = './vermontsim_5_translation/cropped_600'

    # Storing values across all reviewed images (probably can go) -IP
    # These are used to perform the binning process across the whole thing
    vector_sat_total = []
    # pos_in_sat = [] # Moved to individialy for each sat crop image
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
        sat_img = img
        sat_imgz_size = im0s.shape
        # print(path)
        sat_name = 'sat_{}'.format(path)
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
        pred = non_max_suppression(pred, .5, .45, classes=(0, 2), agnostic='store_true')

        pred = pred[0]

        pred = pred.detach().cpu().numpy()
        pred = pred_2_list(pred)
        pred = scale_coords(img.shape[2:], pred, im0s.shape)

        satellite_obj[sat_name].cnn_initialization(pred, [old_img_h, old_img_w])
        neighbor = 5

        if len(satellite_obj[sat_name].p_x) > neighbor:
            satellite_obj[sat_name].k_d_tree_test(neighbor)
            cropped_imdist.append(satellite_obj[sat_name].im_dist)
            cropped_imangle.append(satellite_obj[sat_name].im_angle)
            cropped_feature_closest.append(satellite_obj[sat_name].feature_closest)
            cropped_feature_class.append(satellite_obj[sat_name].feature_class)

    # Performing binning process. Creates histogram of distribution for max vector definition effectiveness and stuff
    binn = BinCompletion('binn')
    binn.bin_initialize(flatten(cropped_imdist), flatten(cropped_imangle), .8, .8)

    # Generating the feature descriptors/vectors for each sat image
    for name in satellite_obj:

        satellite_obj[name].feature_vectors = []

        # Check to see if the satellite image has any feature to have descriptors made for

        if len(satellite_obj[name].feature_class) > neighbor:

            for j in range(len(satellite_obj[name].feature_class)):
                satellite_obj[name].feature_vectors.append(
                    binn.vector_def(j, satellite_obj[name].feature_class, satellite_obj[name].im_dist,
                                    satellite_obj[name].im_angle, satellite_obj[name].feature_closest)[0])

        vector_sat_total.append(satellite_obj[name].feature_vectors)
    vector_sat_total = flatten(vector_sat_total)

    # start Drone Nural network

    # Paths for satellite weights and images
    # weights = './HumWts/yolov7_humvee6403/weights/best.pt'

    # source = './sbtestimages/' # sb tester
    # source = './TrialRunIPDVermont/images_from_drone/'

    weights = './satwts/yolov712/weights/best.pt'
    # source = './vermontsim_5_translation/cropped_800'
    source = './satellite/cropped_300_scres/'

    # Initialize
    set_logging()
    device = select_device(device='0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = 640  # Humvee
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

    old_img_h = old_img_w = imgsz

    hammingtotal = []

    drone_file_idx = 0
    for path, img, im0s, vid_cap in dataset:
        drone_imgz_size = im0s.shape
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
        # First Check for humvee, if not there don't even continue

        humvee_pred = non_max_suppression(pred, .2, .45, classes=6, agnostic='store_true')
        # print(pred.size)
        humvee_pred = humvee_pred[0]

        humvee_pred = humvee_pred.detach().cpu().numpy()
        humvee_pred = pred_2_list(humvee_pred)
        humvee_pred = scale_coords(img.shape[2:], humvee_pred, im0s.shape)
        ##############insert if statement for if a humvee is detected
        if not len(humvee_pred):
            pred = non_max_suppression(pred, .5, .45, classes=(0, 1, 2), agnostic='store_true')
            # print(pred.size)
            pred = pred[0]

            pred = pred.detach().cpu().numpy()
            pred = pred_2_list(pred)

            pred = scale_coords(img.shape[2:], pred, im0s.shape)

            drone.cnn_initialization(pred, [old_img_h, old_img_w])

            vector_drone = []

            # Checks if enough features are even detected to create a correspondence, the plus 1 is for the humvee
            if len(drone.p_x) > neighbor:

                drone.k_d_tree_test(neighbor)
                # if drone.humvee_detected:  # If humvee is detected move on, otherwise go back
                drone.pos_in_sat = np.copy(aflattenerthingy(drone.p_x, drone.p_y))

                for i in range(len(drone.feature_class)):
                    vector_drone.append(
                        binn.vector_def(i, drone.feature_class, drone.im_dist, drone.im_angle,
                                        drone.feature_closest)[0])

                threshold = 2
                # matched_idx = np.zeros(shape=(len(vector_drone), 3))  # stored indexes [[idx sat, idx drone],.....,]
                matched_idx = []

                for i in range(len(vector_drone)):
                    best_rating = threshold
                    matched_idx.append([])
                    j = 0
                    for name in satellite_obj:
                        # print(name)
                        k = 0
                        for vector in satellite_obj[name].feature_vectors:

                            # if vector_drone[i][0:12] == vector_sat_total[j][0:12]:
                            # compair drone to sat, keep only features of the same classes
                            rating = hamming2(vector_drone[i], vector)

                            if rating <= best_rating:
                                # print(rating,vector_drone[i],vector_sat_total[j])
                                best_rating = rating
                                matched_idx[i] = [i, name, k]
                            else:
                                pass
                            k += 1
                        j += 1

                # Show matched portions in original image
                # print(matched_idx)
                # satimgpath = './vermontjpg2.JPG'

                # satimgpath = './lebelingtime/9-1-2021_crop_testsite.png'
                satimgpath = './lebelingtime/9-1-2021_crop_residential.png'

                satimg = cv2.imread(satimgpath)
                # print(matched_idx)
                # print(path)

                for i in matched_idx:

                    if i:  # Check if list is empty IE no matches

                        pos_temp = aflattenerthingy(satellite_obj[i[1]].p_x, satellite_obj[i[1]].p_y)

                        satellite_obj[i[1]].pos_in_sat = np.copy(acrop_2_sat_pos(pos_temp, i[1], shift=50))
                        print(i[1])
                        print(int(drone.pos_in_sat[i[0], 1]), int(drone.pos_in_sat[i[0], 0]))
                        print(int(satellite_obj[i[1]].pos_in_sat[i[2], 1]), int(satellite_obj[i[1]].pos_in_sat[i[2], 0]))

                        cv2.circle(im0s, (int(drone.pos_in_sat[i[0], 1]), int(drone.pos_in_sat[i[0], 0])),
                                   radius=10, color=(i[0] * 255 / len(matched_idx), 0, 255 -(i[0] * 255 / len(matched_idx))), thickness=4)
                        cv2.circle(satimg,
                                   (int(satellite_obj[i[1]].pos_in_sat[i[2], 1]),
                                    int(satellite_obj[i[1]].pos_in_sat[i[2], 0])),
                                   radius=30, color=(i[0] * 255 / len(matched_idx), 0, 255 -(i[0] * 255 / len(matched_idx))), thickness=-1)
    
                # cv2.imwrite('./results/drone{}.png'.format(drone_file_idx), im0s)
                cv2.imwrite('./results/sat{}.png'.format(drone_file_idx), satimg)
                drone_file_idx +=1
                cv2.imshow('69', im0s)
                cv2.waitKey(1)  # 1 millisecond
                # cv2.imshow('69', satimg)
                # cv2.waitKey(1)  # 1 millisecond

            # im0s = cv2.resize(im0s, (satimg.shape[1], satimg.shape[0]), interpolation=cv2.INTER_AREA)

            # # Verti = np.concatenate((im0s, satimg), axis=0)
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(im0s)
            # plt.subplot(122)
            # plt.imshow(satimg)
            # plt.show()



if __name__ == "__main__":
    with torch.no_grad():
        main()
