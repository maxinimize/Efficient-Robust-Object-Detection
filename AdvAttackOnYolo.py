# %% [markdown]
# https://github.com/miladlink/TinyYoloV2
# 
# https://github.com/eriklindernoren/PyTorch-YOLOv3
# 

# %% [markdown]
# # Libraries

# %%
# import os
import time
from PIL import Image
import numpy as np
import json
import cv2
from tqdm import tqdm
# import skimage.io as io
# import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
import torchvision.transforms as transforms
from torchvision.datasets.coco import CocoDetection
from torch.utils.data import DataLoader

from utils.YOLOv2 import *
from models.YOLOv3 import load_model
from attacks.FGSM import FGSM
from attacks.PGD import PGD
from attacks.CW import CW
from attacks.noise import Noise
from detect import detect_image
from utils.loss import compute_loss
from utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from utils.augmentations import TRANSFORM_TRAIN, TRANSFORM_VAL
from utils.transforms import DEFAULT_TRANSFORMS, Resize, ResizeEval

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %% [markdown]
# # Model import

# %%
modelv = 3
img_size=416

if modelv == 2:
    model = load_model_v2(weights = './weights/yolov2-tiny-voc.weights').to(device)
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'TVmonitor'] 
    root_train = "./data/VOC2007/JPEGImages"
    annFile_train = "./data/VOC2007/annotations/train.json"
    root_val = "./data/VOC2007/JPEGImages"
    annFile_val = "./data/VOC2007/annotations/val.json"
    
elif modelv == 3:
    model = load_model("./config/yolov3.cfg", "./weights/yolov3.weights")
    class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    id_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90])
    root_train = "./data/COCO2017/train2017"
    annFile_train = "./data/COCO2017/annotations/instances_train2017_modified.json"
    root_val = "./data/COCO2017/val2017"
    annFile_val = "./data/COCO2017/annotations/instances_val2017_modified.json"
    
else:
    print("invalid model number!")

# %% [markdown]
# # Helper functions
# 

# %%
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def yolo2json(boxes, img_copy, image_id):
    # * put into coco format of x_min,y_min, width, height, bbox_conf, cls
    # yolo format is x_center, y_center, w, h, bbox_conf, cls_conf, cls
    predictions = []
    for box in boxes:
        x_center, y_center, w, h, conf, cls = box
        x_min = max(0, (x_center - w / 2) * img_copy.shape[3])
        y_min = max(0, (y_center - h / 2) * img_copy.shape[2])
        width = min(img_copy.shape[3], w * img_copy.shape[3])
        height = min(img_copy.shape[2], h * img_copy.shape[2])
        # print(x_min,y_min, width, height, bbox_conf, cls)
        predictions.append({
            'image_id': image_id,
            'category_id': int(id_list[int(cls)]) if modelv == 3 else int(cls),
            'bbox': [int(x_min), int(y_min), int(width), int(height)],
            'score': round(float(conf),2)
        })
    return predictions

def nms2yolo(boxes, img_copy):
    boxes = xyxy2xywh(boxes) # convert from coco to yolo: nms returns nx6 (x1, y1, x2, y2, conf, cls), change to center coordinates [x_center, y_center, width, height]
    boxes[:,0] = boxes[:,0]/img_copy.shape[3]
    boxes[:,1] = boxes[:,1]/img_copy.shape[2]
    boxes[:,2] = boxes[:,2]/img_copy.shape[3]
    boxes[:,3] = boxes[:,3]/img_copy.shape[2]
    return boxes

def saveImageWithBoxes(images, boxes, class_names, fileName):  
    to_pil = transforms.ToPILImage() 
    pil_image = to_pil(images.squeeze())
    pred_img = plot_boxes(pil_image, boxes, None, class_names)
    pred_img.save(fileName)
    
def saveImage(img):
    # * just for sanity check, output image. put the dim 3 at the back
    imageN = img.clone().detach()
    imageN = imageN.cpu().squeeze().permute(1, 2, 0).numpy() 
    imageN = cv2.cvtColor(imageN, cv2.COLOR_RGB2BGR)
    # print(imageN.shape)
    cv2.imwrite("data/results/mygraph.jpg", imageN*255) 
    
def getOneIter(dataloader):
    images, annotations = next(iter(dataloader))
    np.set_printoptions(linewidth=500)
    np.set_printoptions(suppress=True)
    print("dataloader out")
    print(annotations[0].numpy())

# %% [markdown]
# # COCO loader

# %% [markdown]
# create dataloader (make different train and val later)

# %%
# coco_dataset_train = CocoDetection(root=root_train, annFile=annFile_train, transform=TRANSFORM_TRAIN_IMG, target_transform=TRANSFORM_TRAIN_TARGET)
coco_dataset_val = CocoDetection(root=root_val, annFile=annFile_val, transforms=TRANSFORM_VAL)
coco_dataset_eval = CocoDetection(root=root_val, annFile=annFile_val, transform=transforms.Compose([transforms.ToTensor(),]))

def collate_fn(batch):
    return tuple(zip(*batch))

# Create a DataLoader for your COCO dataset
train_loader = DataLoader(coco_dataset_val, batch_size=4, shuffle=True, collate_fn=collate_fn) # multiple images per batch
val_loader = DataLoader(coco_dataset_val, batch_size=1, shuffle=True, collate_fn=collate_fn) # one per batch
cocoeval_loader = DataLoader(coco_dataset_eval, batch_size=1, shuffle=True, collate_fn=collate_fn) # original images without transformatios


# %%
getOneIter(val_loader) # print targets

# %% [markdown]
# # Attack Evaluation

# %%
attackImage = 0 # variable for saving attack image, run this first, change pruing ratio (attack), don't run this and only run below cells

# %%
# attacker = FGSM(model=model, epsilon=0.05)
# attacker = PGD(model=model, epsilon=0.05, epoch=5, lr=0.02)
attacker = CW(model=model, epsilon=0.05, lr=0.02, epoch=5, target=52) # 52 is banana
# attacker = Noise(model=model, epsilon=0.1)


# %%
predictionsBefore = []
predictionsAfter = []
lossesBefore = []
lossesAfter = []
mode = "image" # need different modes if i want to save image or output prediction json
# mode = "json"
# image_ids= [71711,19221,22192] # output images that i want, 19221 is broccoli, 22191 is dog, 71711 is plane
image_ids= [19221]

for i, (images, targets) in enumerate(tqdm(val_loader)):
    if targets[0].numel() != 0:
        with torch.no_grad():
            #* modify inputs to be in proper shape
            images = torch.stack(images) # images.shape is [n, 3, 416, 416] (even if n=1)
            images = images.to(device)
            image_id = int(targets[0][0,0].cpu().numpy()) # assume 1 image
            if image_id not in image_ids: continue # for when we want outputs of specific images
            for i, boxes in enumerate(targets): # targets is nx6, (image,class,x,y,w,h)
                if boxes.ndim == 2: boxes[:, 0] = i # change out image_id to id in batch to conform to compute_loss. this is normally done in ListDataset -> collate_fn. the id now starts at 0 for each image
            targets = torch.cat(targets, 0).to(device) # from tuples to one tensor
            originalImageSize = targets[0, 6:].cpu().numpy() # original image shape, assume one image per batch
            targets = targets[:, :6]
            
            #* loss
            model.train()
            # start = time.time()
            outputsBefore = model(images)
            # end = time.time()
            # print(end - start)
            lossBefore, loss_components = compute_loss(outputsBefore, targets, model)
            lossesBefore.append(lossBefore.cpu().numpy())
            
            images_adv = attacker.forward(images, targets) # get adversarial image
            
            outputsAfter = model(images_adv)
            lossAfter, loss_components = compute_loss(outputsAfter, targets, model)
            lossesAfter.append(lossAfter.cpu().numpy())
            
            #* plot
            model.eval()
            
            # ground truth
            print(targets) #(image,class,x,y,w,h), the class id starts from 1
            # nms is (x1, y1, x2, y2, conf, cls), the class id starts from 0
            # yolo is (x_center, y_center, width, height, conf. cls)
            
            # before attack
            outputsBefore = model(images[0].unsqueeze(0))
            boxesBefore = non_max_suppression(outputsBefore, conf_thres=0.3, iou_thres=0.5)[0].numpy()
            if mode == "json":
                boxesBefore = rescale_boxes(boxesBefore, img_size, originalImageSize)
            boxesBefore = nms2yolo(boxesBefore, images)
            if mode == "image":
                saveImageWithBoxes(images[0], boxesBefore, class_names, f"./data/results/images/attack_before_{image_id}.jpg")
            if mode == "json":
                predictionsBefore += yolo2json(boxesBefore, images[0].unsqueeze(0), image_id)
                
            # after attack
            outputsAfter = model(images_adv[0].unsqueeze(0))
            boxesAfter = non_max_suppression(outputsAfter, conf_thres=0.3, iou_thres=0.5)[0].numpy()
            
            
            if mode == "json":
                boxesAfter = rescale_boxes(boxesAfter, img_size, originalImageSize)
            print(boxesAfter)
            boxesAfter = nms2yolo(boxesAfter, images_adv)
            print(boxesAfter)
            if mode == "image":
                # attackImage = images_adv[0] # for saving the same attack image for different pruning ratios
                saveImageWithBoxes(images_adv[0], boxesAfter, class_names, f"./data/results/images/attack_after_{image_id}.jpg")
                # saveImageWithBoxes(attackImage, attackPredictions, class_names, f"./data/results/images/pruning/attack_after_{image_id}_90new.jpg") # plot different pruning ratios with same attack image
            if mode == "json":
                predictionsAfter += yolo2json(boxesAfter, images_adv[0].unsqueeze(0), image_id)
            attackPredictions = boxesAfter
            # time.sleep(0.1) # for using noise attack
            
    else: continue # pics without targets
    # break


with open(f'./data/results/predictionsBefore.json', 'w') as f:
    json.dump(predictionsBefore, f)
with open(f'./data/results/predictionsAfter.json', 'w') as f:
    json.dump(predictionsAfter, f)
np.savetxt("./data/results/lossesBefore.csv", lossesBefore, delimiter=",")
np.savetxt("./data/results/lossesAfter.csv", lossesAfter, delimiter=",")

# %%
data = np.loadtxt('./data/results/lossesBefore.csv', delimiter=',')
average = np.mean(data)
print("Avg loss before attack:", average)
data = np.loadtxt('./data/results/lossesAfter.csv', delimiter=',')
average = np.mean(data)
print("Avg loss after attack:", average)

# %% [markdown]
# # Adversarial training

# %%
losses = []
epochs = 10
checkpoint_interval = 1
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )

for epoch in range(1, epochs+1):
    for i, (images, targets) in enumerate(tqdm(train_loader)):
        model.train()
        lossesEpoch = []
        if targets[0].numel() != 0:
            #* modify inputs to be in proper shape
            images = torch.stack(images) # images.shape is [n, 3, 416, 416] (even if n=1)
            images = images.to(device)
            for i, boxes in enumerate(targets): # targets is nx6, (image,class,x,y,w,h)
                if boxes.ndim == 2: boxes[:, 0] = i # change out image_id to id in batch to conform to compute_loss. this is normally done in ListDataset -> collate_fn
            targets = torch.cat(targets, 0).to(device) # from tuples to one tensor
            targets = targets[:, :6]
            # if image_id not in image_ids: continue # for when we want outputs of specific images
            
            images_adv = attacker.forward(images, targets) # get adversarial image
            outputsBefore = model(images)
            lossBefore, loss_components = compute_loss(outputsBefore, targets, model)
            outputsAfter = model(images_adv)
            lossAfter, loss_components = compute_loss(outputsAfter, targets, model)
            loss = lossBefore + lossAfter
            lossesEpoch.append(loss.cpu().numpy())
            loss.backward()
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()
            
            time.sleep(0.1) # for using noise attack
        else: continue # pics without targets
    losses = np.average(lossesEpoch)
            
    if epoch % checkpoint_interval == 0:
        checkpoint_path = f"./data/results/checkpoints/yolov3_ckpt_{epoch}.pth"
        print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
        torch.save(model.state_dict(), checkpoint_path)
            
    # break

# %% [markdown]
# # Get mAP

# %%
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
        
coco_gld = COCO(annFile_val) # coco
if modelv == 2:
    coco_rst = coco_gld.loadRes('./data/results/v2predictions.json')
elif modelv == 3:
    coco_rst = coco_gld.loadRes('./data/results/v3predictions.json')
    
coco_rst = coco_gld.loadRes('./data/results/predictionsAfter.json')
cocoEval = COCOeval(coco_gld, coco_rst, iouType='bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


