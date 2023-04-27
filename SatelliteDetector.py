import argparse
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

# Converts from yolov7 to yolov4, probably wasts time but we will see -IP
def pred_2_list(prediction):
    myorder = [5, 0, 1, 2, 3]
    pred = prediction.tolist()
    idx = 0
    for vector in pred:

        pred[idx] = [vector[i] for i in myorder]
        idx +=1 
    return pred

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
def detect(weights,source,view_img=False,save_img=False,nosave = False,device = '0'):

    # source, weights, view_img, = opt.source, opt.weights, opt.view_img
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories

    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = 640
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # if trace:
    #     model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

  
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=True)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, .5, .45, agnostic='store_true')
        # print(pred.size)
        pred = pred[0]
        
        pred=pred.detach().cpu().numpy()
        pred = pred_2_list(pred)
        print(pred)
   

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()

    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # opt = parser.parse_args()

    weights = './satwts/yolov712/weights/best.pt'
    source = './satellite/cropped_300_sbcm/'
    with torch.no_grad():
     
        detect(weights=weights, source = source,view_img=True,device='cpu')

