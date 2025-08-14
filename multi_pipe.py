#!/usr/bin/env python
# coding: utf-8

# https://github.com/miladlink/TinyYoloV2
# 
# https://github.com/eriklindernoren/PyTorch-YOLOv3
# 

# # Setup (Only use the first time or if the dataset was changed)

# In[2]:


# Check if running in Jupyter
# import subprocess
# import sys

# if 'ipykernel' in sys.modules:
#     # In Jupyter, use IPython's system command
#     get_ipython().system('python filter_sample_json.py data/COCO2017/images/valid_sample data/COCO2017/annotations/instances_val2017_modified_sample.json data/COCO2017/images/50000 data/COCO2017/annotations/instances_train2017_modified_sample.json')
# else:
#     # In a standard Python script, use subprocess to run the command
#     subprocess.run(['python', 'filter_sample_json.py', 'data/COCO2017/images/valid_sample',
#                     'data/COCO2017/annotations/instances_val2017_modified_sample.json',
#                     'data/COCO2017/images/50000', 'data/COCO2017/annotations/instances_train2017_modified_sample.json'])


# # Libraries

# In[ ]:


import os
import time
# from PIL import Image
import numpy as np
import json
import cv2
from tqdm import tqdm
# import skimage.io as io
# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
import torch
import torch.optim as optim
# import torchvision
# from torchvision import transforms
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

import torch.distributed as dist
import torch.utils.data.distributed

import datetime
# # Helper functions + vars
# 

# In[3]:


def xyxy2xywh(x):
    """Convert [x1, y1, x2, y2] boxes to [x_center, y_center, w, h] where xy1=top left and xy2 = bottom right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def xywh2xyxy(x):
    """Convert [x_center, y_center, w, h] boxes to [x1, y1, x2, y2]."""
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
        predictions.append({
            'image_id': image_id,
            'category_id': int(id_list[int(cls)]) if modelv == 3 else int(cls),
            'bbox': [int(x_min), int(y_min), int(width), int(height)],
            'score': round(float(conf),2)
        })
    return predictions

def nms2yolo(boxes, img_copy):
    """Convert NMS output to normalized YOLO format."""
    boxes = xyxy2xywh(boxes)
    boxes[:,0] = boxes[:,0]/img_copy.shape[3]
    boxes[:,1] = boxes[:,1]/img_copy.shape[2]
    boxes[:,2] = boxes[:,2]/img_copy.shape[3]
    boxes[:,3] = boxes[:,3]/img_copy.shape[2]
    return boxes

def saveImageWithBoxes(images, boxes, class_names, fileName):
    """Save image with predicted boxes drawn on it."""
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(images.squeeze())
    pred_img = plot_boxes(pil_image, boxes, None, class_names)
    pred_img.save(fileName)

def saveImage(img):
    """Save a tensor image to disk for sanity check."""
    imageN = img.clone().detach()
    imageN = imageN.cpu().squeeze().permute(1, 2, 0).numpy()
    imageN = cv2.cvtColor(imageN, cv2.COLOR_RGB2BGR)
    cv2.imwrite("data/results/mygraph.jpg", imageN*255)

def getOneIter(dataloader):
    """Print one batch from dataloader for inspection."""
    images, annotations = next(iter(dataloader))
    np.set_printoptions(linewidth=500)
    np.set_printoptions(suppress=True)
    print("dataloader out")
    print(annotations[0].numpy())


def imgToGreyscale(img):
    """Convert RGB image tensor to greyscale (3 channels)."""
    if img.shape[0] != 3:
        raise ValueError("Input tensor must have shape [3, H, W].")
    grayscale = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    grayscale_tensor = grayscale.unsqueeze(0).repeat(3, 1, 1)
    return grayscale_tensor


# In[ ]:


# current_device = torch.current_device('cuda' if torch.cuda.is_available() else 'cpu')
# current_device

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # reset CUDA debugging environment variable
# os.environ['TORCH_USE_CUDA_DSA'] = '1' # enable CUDA DSA for debugging

# Define evaluation function for model performance assessment
def evaluate_model(model, dataloader, attacker=None, device=None):
    """
    Evaluate model performance with and without adversarial attacks
    
    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        attacker: Optional attacker to generate adversarial examples
        device: Device to run evaluation on
    
    Returns:
        clean_loss: Average loss on clean samples
        adv_loss: Average loss on adversarial samples (if attacker provided)
        metrics: Dictionary with additional evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set model to evaluation mode
    model.eval()
    
    clean_losses = []
    adv_losses = []
    metrics = {
        'clean_detections': 0,
        'adv_detections': 0,
        'total_samples': 0
    }
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if targets is None or len(targets) == 0:
                continue
                
            # Process images and targets
            try:
                images = torch.stack(images).to(device)
                
                for i, boxes in enumerate(targets):
                    if boxes is not None and boxes.ndim == 2:
                        boxes[:, 0] = i  # Update image ids
                
                if len(targets) > 0:
                    targets = torch.cat(targets, 0).to(device)
                    targets = targets[:, :6]
                else:
                    continue  # Skip if no targets
                
                # Clean evaluation
                with torch.no_grad():
                    outputs_clean = model(images)
                    loss_clean, _ = compute_loss(outputs_clean, targets, model)
                    clean_losses.append(loss_clean.item())
                
                # Get clean detections
                clean_dets = [len(non_max_suppression(outputs_clean[i:i+1], conf_thres=0.3, iou_thres=0.5)[0])
                             for i in range(len(images))]
                metrics['clean_detections'] += sum(clean_dets)
                
                # Adversarial evaluation if attacker provided
                if attacker is not None:
                    # Create adversarial examples - use requires_grad here
                    with torch.enable_grad():
                        model.train()  # Temporarily set to train for attack generation
                        images_adv = attacker(images, targets)
                        model.eval()  # Set back to eval for evaluation
                    
                    # Evaluate on adversarial examples
                    with torch.no_grad():
                        outputs_adv = model(images_adv)
                        loss_adv, _ = compute_loss(outputs_adv, targets, model)
                        adv_losses.append(loss_adv.item())
                    
                    # Get adversarial detections
                    adv_dets = [len(non_max_suppression(outputs_adv[i:i+1], conf_thres=0.3, iou_thres=0.5)[0])
                               for i in range(len(images))]
                    metrics['adv_detections'] += sum(adv_dets)
                
                metrics['total_samples'] += len(images)
                
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    
    # Calculate average losses
    clean_loss = sum(clean_losses) / len(clean_losses) if clean_losses else float('inf')
    adv_loss = sum(adv_losses) / len(adv_losses) if adv_losses else float('inf')
    
    # Calculate detection rates
    if metrics['total_samples'] > 0:
        metrics['clean_detection_rate'] = metrics['clean_detections'] / metrics['total_samples']
        if attacker is not None:
            metrics['adv_detection_rate'] = metrics['adv_detections'] / metrics['total_samples']
            metrics['detection_drop'] = metrics['clean_detection_rate'] - metrics['adv_detection_rate']
    
    return clean_loss, adv_loss, metrics


# In[ ]:


parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')


# In[ ]:


import argparse


parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')
parser.add_argument("--lr",           type=float, default=0.1)  # Used: not directly, but can be passed to optimizer
parser.add_argument("--batch_size",   type=int,   default=64)   # Used: DataLoader batch size
parser.add_argument("--max_epochs",   type=int,   default=4)    # Not used: hardcoded 'epochs' variable is used instead
parser.add_argument("--num_workers",  type=int,   default=4)    # Used: DataLoader num_workers
parser.add_argument("--load",         action="store_true")     # Used: triggers model loading from checkpoint
# DDP / distributed setup
parser.add_argument("--distributed",  action="store_true", help="launch in distributed data parallel mode")  # Used: enables DDP
parser.add_argument("--dist_backend", type=str, default="nccl")  # Used: DDP backend
parser.add_argument("--init_method",  type=str, default="tcp://127.0.0.1:29500", help="url used to set up distributed training")  # Not used: overridden by 'env://'
parser.add_argument("--world_size",   type=int, default=None, help="total number of processes (will fallback to SLURM_NTASKS)")  # Used: DDP world size
parser.add_argument("--rank",         type=int, default=None, help="global rank of this process (will fallback to SLURM_PROCID)")  # Used: DDP rank
parser.add_argument("--local_rank",   type=int, default=None, help="local gpu index (will fallback to SLURM_LOCALID)")  # Used: DDP local rank

args = parser.parse_args()


# In[ ]:

import socket

# Resolve local_rank and rank from args or environment (torchrun sets LOCAL_RANK)
_local_rank = args.local_rank
if _local_rank is None:
    _local_rank = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID"))
_local_rank = int(_local_rank) if _local_rank is not None else 0

_rank = args.rank
if _rank is None:
    _rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))

_world_size = args.world_size
if _world_size is None:
    _world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

rank = _rank
local_rank = _local_rank
world_size = _world_size
# Now determine device safely
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device_id = _local_rank % torch.cuda.device_count()
    # set the visible device for this process
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
else:
    device_id = None
    device = torch.device("cpu")

# Expose back to the rest of your script
local_rank = _local_rank
rank = _rank
current_device = device  # use this when calling model.to(...) or as map_location
# local_rank = args.local_rank
# rank = args.rank
# current_device = local_rank
# torch.cuda.set_device(local_rank)


# In[ ]:


""" this block initializes a process group and initiate communications
    between all processes running on all nodes """

print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
#init the process group
dist.init_process_group(
    backend=args.dist_backend,    # 'nccl' for GPUs
    init_method="env://",         # let torchrun/SLURM env drive rendezvous
    world_size=world_size,
    rank=rank
)

print("process group ready!")

print('From Rank: {}, ==> Making model..'.format(rank))


# In[5]:

epochs = 100 # currently, 100 seems like it works very well
checkpoint_interval = 10 
# if time is limited then make this smaller, do note that checkpoints are around 270MB per.
validation_interval = 20  # Set validation interval here
modelv = 3
img_size=416

root_train = "./data/COCO2017/images/50000"
annFile_train = "./data/COCO2017/annotations/instances_train2017_modified_sample.json"
root_val = "./data/COCO2017/images/valid_sample"
annFile_val = "./data/COCO2017/annotations/instances_val2017_modified_sample.json"

mode = "image" # need different modes if i want to save image or output prediction json
# mode = "json"
image_ids= [139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
            1268, 1296, 1353,1425, 1490, 1503, 1532, 1584, 1675, 1761]


# In[6]:


class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
id_list = np.array(range(0,80))


# # Model import

# In[ ]:


# if modelv == 2:
#     model = load_model_v2(weights = './weights/yolov2-tiny-voc.weights').to(current_device)
#     class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'TVmonitor']
#     root_train = "./data/VOC2007/JPEGImages"
#     annFile_train = "./data/VOC2007/annotations/train.json"
#     root_val = "./data/VOC2007/JPEGImages"
#     annFile_val = "./data/VOC2007/annotations/val.json"

if args.load:
    # Determine which checkpoint to load - best or latest
    checkpoint_path = "./data/results/checkpoints/yolov3_ckpt_best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found. Looking for latest checkpoint...")
        checkpoint_dir = "./data/results/checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("yolov3_ckpt_") and f.endswith(".pth")]
            if checkpoint_files:
                # Sort by epoch number or modification time
                checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                print(f"Found latest checkpoint: {checkpoint_path}")
            else:
                print("No checkpoints found. Starting from pretrained weights.")
                args.load = False
        else:
            print("Checkpoint directory doesn't exist. Starting from pretrained weights.")
            args.load = False
    
    if args.load:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=current_device)
        model = load_model("./config/yolov3.cfg", ckpt['model_state_dict'])
        model.to(device)
        from torch.nn import SyncBatchNorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
        if hasattr(model, "_set_static_graph"):
            model._set_static_graph()
        base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        
        # Load optimizer state and tracking variables
        min_loss = ckpt['loss']
        start_epoch = ckpt['epoch'] + 1  # Start from next epoch
        print(f"Resuming training from epoch {start_epoch}, best loss: {min_loss:.5f}")
    print('From Rank: {}, ==> Preparing data..'.format(rank))
elif modelv == 3:
    
    min_loss = 100
    model = load_model("./config/yolov3.cfg", "./weights/yolov3.weights")
    #model.cuda()
    model.to(device)
    from torch.nn import SyncBatchNorm
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
    model._set_static_graph()
    print('From Rank: {}, ==> Preparing data..'.format(rank))

else:
    print("invalid model number!")


# # COCO loader

# create dataloader (make different train and val later)

# In[ ]:


# coco_dataset_train = CocoDetection(root=root_train, annFile=annFile_train, transform=TRANSFORM_TRAIN_IMG, target_transform=TRANSFORM_TRAIN_TARGET)
# coco_dataset_train = CocoDetection(root=root_train, annFile=annFile_train, transforms=TRANSFORM_TRAIN)
coco_dataset_train = CocoDetection(root=root_train, annFile=annFile_train, transforms=TRANSFORM_TRAIN)
coco_dataset_val = CocoDetection(root=root_val, annFile=annFile_val, transforms=TRANSFORM_VAL)
# coco_dataset_eval = CocoDetection(root=root_val, annFile=annFile_val, transform=transforms.Compose([transforms.ToTensor(),]))

def collate_fn(batch):
    """Custom collate function for COCO detection: filters empty targets, adds batch index."""
    try:
        # Filter out empty targets and create batches correctly
        # Use len() > 0 check instead of direct boolean evaluation
        batch = [(imgs, targets) for imgs, targets in batch if targets is not None and len(targets) > 0]
        if not batch:
            return None, None
            
        imgs, targets = list(zip(*batch))
        
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            if boxes is not None and hasattr(boxes, 'ndim') and boxes.ndim == 2:
                boxes[:, 0] = i  # image id becomes batch index
        
        return imgs, targets
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # Return a safe default that won't crash the training loop
        return None, None

train_sampler = torch.utils.data.distributed.DistributedSampler(coco_dataset_train)
valid_sampler = torch.utils.data.distributed.DistributedSampler(coco_dataset_val)

# Create a DataLoader for your COCO dataset
train_loader = DataLoader(coco_dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, pin_memory=True,collate_fn=collate_fn) # multiple images per batch
val_loader = DataLoader(coco_dataset_val, batch_size=1, shuffle=(valid_sampler is None), num_workers=args.num_workers, sampler=valid_sampler, pin_memory=True, collate_fn=collate_fn)
# one per batch
# cocoeval_loader = DataLoader(coco_dataset_eval, batch_size=1, shuffle=True, collate_fn=collate_fn) # original images without transformatios


# In[9]:


if rank == 0:
    getOneIter(train_loader)
    getOneIter(val_loader)


# # Adversarial training

# In[10]:


eps = 0.05
# For adversarial training, PGD is often more effective than CW
# Using untargeted attack (no target specified) is better for general robustness
attacker = PGD(model=model, epsilon=eps, epoch=5, lr=0.01)
# Other options:
# attacker = FGSM(model=model, epsilon=0.05)  # Faster but less effective
# attacker = CW(model=model, epsilon=eps, lr=eps/3, epoch=5)  # No target for untargeted attack
# attacker = Noise(model=model, epsilon=0.1)  # Simple noise baseline


# In[ ]:


losses = []
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(
            params,
            lr=args.lr,  # Use command line argument instead of hardcoded value
            momentum=0.9,
            weight_decay=1e-4,
        )

# Load optimizer state if resuming from checkpoint
if args.load and 'ckpt' in locals() and 'optimizer_state_dict' in ckpt:
    try:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print("Successfully loaded optimizer state from checkpoint")
    except Exception as e:
        print(f"Warning: Failed to load optimizer state: {e}")
        print("Starting with fresh optimizer state")

min_loss = min_loss if 'min_loss' in locals() else float('inf')  # Track best loss for model saving
start_epoch = start_epoch if 'start_epoch' in locals() else 1  # Use loaded epoch or start from 1
torch.autograd.set_detect_anomaly(True)

# Use start_epoch variable for resuming training
for epoch in range(start_epoch, epochs+1):
    print(f"Starting epoch {epoch}/{epochs}")
    train_sampler.set_epoch(epoch)
    lossesEpoch = []
    torch.cuda.empty_cache()
    
    epoch_start = time.time()
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
        
        model.train()
        start = time.time() # just to check time taken

        if targets is None or targets[0] is None or targets[0].numel() == 0:
            continue
            
        try:
            #* modify inputs to be in proper shape
            images = torch.stack(images) # images.shape is [n, 3, 416, 416] (even if n=1)
            images = images.to(current_device)

            # modify targets to be in proper shape
            for i, boxes in enumerate(targets): # targets is nx6, (image,class,x,y,w,h)
                if boxes.ndim == 2:
                    boxes[:, 0] = i # change out image_id to id in batch to conform to compute_loss

            targets = torch.cat(targets, 0).to(current_device) # from tuples to one tensor
            targets = targets[:, :6]
            
            # Verify and fix class indices range
            class_indices = targets[:, 1].long()
            valid_classes = (class_indices >= 0) & (class_indices < 80)
            
            if not valid_classes.all():
                print(f"Warning: Invalid class indices found: {class_indices[~valid_classes]}")
                # Filter out invalid classes
                targets = targets[valid_classes]
                if targets.shape[0] == 0:
                    print("No valid targets after filtering, skipping batch")
                    continue
            
            # ensure all class indices are properly converted to long
            targets[:, 1] = targets[:, 1].long()

            print(f"Batch {batch_idx}: targets shape: {targets.shape}, class range: {targets[:, 1].min()}-{targets[:, 1].max()}")

            images_adv = attacker.forward(images, targets) # get adversarial image
            # Process full batch instead of just first image
            outputsBefore = model(images)
            lossBefore, loss_components_before = compute_loss(outputsBefore, targets, model)
            
            outputsAfter = model(images_adv)
            lossAfter, loss_components_after = compute_loss(outputsAfter, targets, model)
            
            # Balanced loss with more weight on clean examples for stability
            # 0.4 on clean, 0.6 on adversarial - adjust as needed
            loss = 0.4 * lossBefore + 0.6 * lossAfter

            # Implement gradient accumulation - scale loss by gradient_accumulation_steps
            gradient_accumulation_steps = 2  # Adjust based on your GPU memory
            loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
            
            # lossesEpoch.append(loss.detach().cpu().numpy())
            loss.backward()
            loss_val = loss.detach().clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                loss_val = loss_val / dist.get_world_size()
            lossesEpoch.append(loss_val.cpu().numpy())  # append the average loss across all GPUs
            
            # Apply gradient accumulation (update every gradient_accumulation_steps batches)
            # We've already scaled the loss above, now just need to decide when to optimize
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            # Otherwise gradients are automatically accumulated

            # Remove sleep that was slowing down training
            # time.sleep(0.1) # for using noise attack

            batch_time = time.time() - start

            elapse_time = time.time() - epoch_start
            elapse_time = datetime.timedelta(seconds=elapse_time)
            print("From Rank: {}, Training time {}".format(rank, elapse_time))
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            print(f"Targets shape: {targets.shape if 'targets' in locals() else 'undefined'}")
            if 'targets' in locals():
                print(f"Class indices: {targets[:, 1].unique()}")
            # clear gradients and continue to next batch
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue

    if lossesEpoch:
        losses_avg = np.average(lossesEpoch)
        print(f"Epoch {epoch} average loss: {losses_avg}")
        losses.append(losses_avg)
        
        # Run validation if on rank 0
        if rank == 0 and epoch % validation_interval == 0:  # Validate every validation_interval epochs
            print(f"Running validation after epoch {epoch}...")
            clean_loss, adv_loss, metrics = evaluate_model(
                model=model, 
                dataloader=val_loader, 
                attacker=attacker, 
                device=current_device
            )
            print(f"Validation results - Clean loss: {clean_loss:.4f}, Adv loss: {adv_loss:.4f}")
            print(f"Clean detection rate: {metrics.get('clean_detection_rate', 0):.4f}")
            print(f"Adversarial detection rate: {metrics.get('adv_detection_rate', 0):.4f}")
            print(f"Detection drop: {metrics.get('detection_drop', 0):.4f}")
        
        # Sync min_loss across all processes
        if dist.is_available() and dist.is_initialized():
            global_min_loss = torch.tensor([min_loss], device=current_device)
            dist.all_reduce(global_min_loss, op=dist.ReduceOp.MIN)
            min_loss = global_min_loss.item()
            
        if losses_avg < min_loss and rank == 0:  # only save the best model on rank 0
            checkpoint_path = f"./data/results/checkpoints/yolov3_ckpt_best.pth"
            print(f"---- Saving new best checkpoint to: '{checkpoint_path}' ----")
            print(f"---- Previous best: {min_loss:.5f}, New best: {losses_avg:.5f} ----")
            os.makedirs("./data/results/checkpoints", exist_ok=True)
            
            # Save checkpoint with comprehensive metadata
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses_avg,
                'epoch': epoch,
                'date_saved': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'validation_metrics': metrics if 'metrics' in locals() else {},
                'hyperparams': {
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'gradient_accumulation_steps': gradient_accumulation_steps,
                    'eps': eps  # adversarial training epsilon
                }
            }
            torch.save(checkpoint, checkpoint_path)
            min_loss = losses_avg

    if epoch % checkpoint_interval == 0 and rank == 0:  # save checkpoint every checkpoint_interval epochs on rank 0
        checkpoint_path = f"./data/results/checkpoints/yolov3_ckpt_{epoch}.pth"
        print(f"---- Saving periodic checkpoint to: '{checkpoint_path}' ----")
        os.makedirs("./data/results/checkpoints", exist_ok=True)
        
        # Run validation before saving if not already done
        if epoch % validation_interval != 0:  # Only if not already done above
            clean_loss, adv_loss, metrics = evaluate_model(
                model=model, 
                dataloader=val_loader, 
                attacker=attacker, 
                device=current_device
            )
        
        # Save checkpoint with comprehensive metadata
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses_avg,
            'epoch': epoch,
            'date_saved': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'validation_metrics': metrics if 'metrics' in locals() else {},
            'hyperparams': {
                'lr': args.lr,
                'batch_size': args.batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'eps': eps  # adversarial training epsilon
            }
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Create a symbolic link to the latest checkpoint for easy loading
        latest_link = "./data/results/checkpoints/yolov3_ckpt_latest.pth"
        if os.path.exists(latest_link):
            try:
                os.remove(latest_link)
            except:
                pass  # Ignore if can't remove
        try:
            # Create relative symlink for portability
            os.symlink(os.path.basename(checkpoint_path), latest_link)
        except:
            # If symlink fails, just copy the file
            import shutil
            shutil.copy2(checkpoint_path, latest_link)
        print(f"The best loss has been {min_loss}")


# ## Load adversarial trained model

# In[18]:


if modelv == 3:
    if modelv == 3:
        ckpt = torch.load("./data/results/checkpoints/yolov3_ckpt_best.pth", map_location=current_device)
        model = load_model("./config/yolov3.cfg", ckpt['model_state_dict'])
        model.to(current_device)

else:
    print("invalid model number!")


# # Attack Evaluation

# In[13]:


# attackImage = 0 # variable for saving attack image, run this first, change pruning ratio (attack), 
# #don't run this and only run below cells


# ### NOTE: Attacker was defined here.

# In[21]:


predictionsBefore = []
predictionsAfter = []
lossesBefore = []
lossesAfter = []

os.makedirs("./data/results/images", exist_ok=True)

for i, (images, targets) in enumerate(tqdm(val_loader)):
    if targets[0].numel() != 0:
        with torch.no_grad():
            #* modify inputs to be in proper shape
            images = torch.stack(images) # images.shape is [n, 3, 416, 416] (even if n=1)
            images = images.to(current_device)
            image_id = int(targets[0][0,0].cpu().numpy()) # assume 1 image
            if image_id not in image_ids: continue # for when we want outputs of specific images
            for i, boxes in enumerate(targets): # targets is nx6, (image,class,x,y,w,h)
                if boxes.ndim == 2: boxes[:, 0] = i # change out image_id to id in batch to conform to compute_loss. this is normally done in ListDataset -> collate_fn. the id now starts at 0 for each image
            targets = torch.cat(targets, 0).to(current_device) # from tuples to one tensor
            # originalImageSize = targets[0, 6:].cpu().numpy() # original image shape, assume one image per batch - NOT available in json format
            img_info = coco_dataset_val.coco.imgs[image_id]
            originalImageSize = (img_info['height'], img_info['width'])
            targets = targets[:, :6]

            # print debugging information
            print(f"Image ID: {image_id}")
            print(f"Original targets shape: {targets.shape}")
            print(f"Targets data:")
            print(targets)
            print(f"Class indices: {targets[:, 1]}")
            print(f"Class range: {targets[:, 1].min()} - {targets[:, 1].max()}")

            # check mapping of class indices
            original_classes = targets[:, 1].clone()
            print(f"Original class IDs: {original_classes}")

            # mapping class indices ([1, 80] to [0, 79] range
            targets[:, 1] = targets[:, 1] - 1

            # verify mapped class indices
            mapped_classes = targets[:, 1]
            print(f"Mapped class IDs: {mapped_classes}")
            print(f"Mapped class range: {mapped_classes.min()} - {mapped_classes.max()}")

            # # ensure all classes are in range [0, 79]
            # valid_mask = (mapped_classes >= 0) & (mapped_classes < 80)
            # if not valid_mask.all():
            #     print(f"Invalid class indices found: {mapped_classes[~valid_mask]}")
            #     targets = targets[valid_mask]
            #     if targets.shape[0] == 0:
            #         print("No valid targets after filtering, skipping image")
            #         continue
            #     print(f"Filtered targets shape: {targets.shape}")

            # ensure all class indices are long
            targets[:, 1] = targets[:, 1].long()

            # final validation
            final_classes = targets[:, 1]
            print(f"Final class indices: {final_classes}")
            print(f"Final class range: {final_classes.min()} - {final_classes.max()}")
            print(f"All classes in range [0, 79]: {((final_classes >= 0) & (final_classes < 80)).all()}")

            #* loss
            model.train()
            try:
                outputsBefore = model(images)
                print(f"Model output shapes: {[out.shape for out in outputsBefore]}")

                lossBefore, loss_components = compute_loss(outputsBefore, targets, model)
                lossesBefore.append(lossBefore.cpu().numpy())

                images_adv = attacker.forward(images, targets) # get adversarial image

                outputsAfter = model(images_adv)
                lossAfter, loss_components = compute_loss(outputsAfter, targets, model)
                lossesAfter.append(lossAfter.cpu().numpy())

            except RuntimeError as e:
                print(f"CUDA error occurred: {e}")
                print(f"Error details:")
                print(f"  Targets shape: {targets.shape}")
                print(f"  Class indices: {targets[:, 1]}")
                # print(f"  Class unique values: {targets[:, 1].unique()}")
                print(f"  Class data type: {targets[:, 1].dtype}")

                # clean CUDA cache and skip this iteration
                torch.cuda.empty_cache()
                continue

            #* plot
            model.eval()

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
            boxesAfter = nms2yolo(boxesAfter, images_adv)
            print(boxesAfter)
            if mode == "image":
                saveImageWithBoxes(images_adv[0], boxesAfter, class_names, f"./data/results/images/attack_after_{image_id}.jpg")
            if mode == "json":
                predictionsAfter += yolo2json(boxesAfter, images_adv[0].unsqueeze(0), image_id)

    else: continue # pics without targets

if rank == 0:
    with open(f'./data/results/predictionsBefore.json', 'w') as f:
        json.dump(predictionsBefore, f)
    with open(f'./data/results/predictionsAfter.json', 'w') as f:
        json.dump(predictionsAfter, f)
    np.savetxt("./data/results/lossesBefore.csv", lossesBefore, delimiter=",")
    np.savetxt("./data/results/lossesAfter.csv", lossesAfter, delimiter=",")


# In[15]:


# predictionsBefore = []
# predictionsAfter = []
# lossesBefore = []
# lossesAfter = []
# # mode = "image" # need different modes if i want to save image or output prediction json
# mode = "json"
# # image_ids= [71711,19221,22192] # output images that i want, 19221 is broccoli, 22192 is dog, 71711 is plane
# # image_ids= [139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
# #             1268, 1296, 1353,1425, 1490, 1503, 1532, 1584, 1675, 1761] # sample image id
# image_ids = [139]

# os.makedirs("./data/results/images", exist_ok=True)

# for i, (images, targets) in enumerate(tqdm(val_loader)):
#     if targets[0].numel() != 0:
#         with torch.no_grad():
#             #* modify inputs to be in proper shape
#             images = torch.stack(images) # images.shape is [n, 3, 416, 416] (even if n=1)
#             images = images.to(current_device)
#             image_id = int(targets[0][0,0].cpu().numpy()) # assume 1 image
#             if image_id not in image_ids: continue # for when we want outputs of specific images
#             for i, boxes in enumerate(targets): # targets is nx6, (image,class,x,y,w,h)
#                 if boxes.ndim == 2: boxes[:, 0] = i # change out image_id to id in batch to conform to compute_loss. this is normally done in ListDataset -> collate_fn. the id now starts at 0 for each image
#             targets = torch.cat(targets, 0).to(current_device) # from tuples to one tensor
#             # originalImageSize = targets[0, 6:].cpu().numpy() # original image shape, assume one image per batch - NOT available in json format
#             img_info = coco_dataset_val.coco.imgs[image_id]
#             originalImageSize = (img_info['height'], img_info['width'])
#             targets = targets[:, :6]

#             #* loss
#             model.train()
#             # start = time.time()
#             outputsBefore = model(images)
#             # end = time.time()
#             # print(end - start)
#             lossBefore, loss_components = compute_loss(outputsBefore, targets, model)
#             lossesBefore.append(lossBefore.cpu().numpy())

#             images_adv = attacker.forward(images, targets) # get adversarial image

#             outputsAfter = model(images_adv)
#             lossAfter, loss_components = compute_loss(outputsAfter, targets, model)
#             lossesAfter.append(lossAfter.cpu().numpy())

#             #* plot
#             model.eval()

#             # ground truth
#             # print(targets) #(ima ge,class,x,y,w,h), the class id starts from 1
#             # nms is (x1, y1, x2, y2, conf, cls), the class id starts from 0
#             # yolo is (x_center, y_center, width, height, conf. cls)

#             # before attack
#             outputsBefore = model(images[0].unsqueeze(0))
#             boxesBefore = non_max_suppression(outputsBefore, conf_thres=0.3, iou_thres=0.5)[0].numpy()
#             if mode == "json":
#                 boxesBefore = rescale_boxes(boxesBefore, img_size, originalImageSize)
#             boxesBefore = nms2yolo(boxesBefore, images)
#             if mode == "image":
#                 saveImageWithBoxes(images[0], boxesBefore, class_names, f"./data/results/images/attack_before_{image_id}.jpg")
#             if mode == "json":
#                 predictionsBefore += yolo2json(boxesBefore, images[0].unsqueeze(0), image_id)

#             # after attack
#             outputsAfter = model(images_adv[0].unsqueeze(0))
#             boxesAfter = non_max_suppression(outputsAfter, conf_thres=0.3, iou_thres=0.5)[0].numpy()


#             if mode == "json":
#                 boxesAfter = rescale_boxes(boxesAfter, img_size, originalImageSize)
#             # print(boxesAfter)
#             boxesAfter = nms2yolo(boxesAfter, images_adv)
#             print(boxesAfter)
#             if mode == "image":
#                 saveImageWithBoxes(images_adv[0], boxesAfter, class_names, f"./data/results/images/attack_after_{image_id}.jpg")

#                 # attackImage = images_adv[0] # for saving the same attack image for different pruning ratios, comment out after save
#                 # saveImageWithBoxes(attackImage, boxesAfter, class_names, f"./data/results/images/pruning/{image_id}/attack_after_99_x.jpg") # plot different pruning ratios with same attack image

#                 # greyscaleAttackImage = imgToGreyscale(attackImage)
#                 # saveImageWithBoxes(greyscaleAttackImage, boxesAfter, class_names, f"./data/results/images/pruning/{image_id}/attack_after_x_grey.jpg") # plot different pruning ratios with same attack image
#             if mode == "json":
#                 predictionsAfter += yolo2json(boxesAfter, images_adv[0].unsqueeze(0), image_id)
#             # time.sleep(0.1) # for using noise attack

#     else: continue # pics without targets
#     # break


# with open(f'./data/results/predictionsBefore.json', 'w') as f:
#     json.dump(predictionsBefore, f)
# with open(f'./data/results/predictionsAfter.json', 'w') as f:
#     json.dump(predictionsAfter, f)
# np.savetxt("./data/results/lossesBefore.csv", lossesBefore, delimiter=",")
# np.savetxt("./data/results/lossesAfter.csv", lossesAfter, delimiter=",")


# In[16]:


# data = np.loadtxt('./data/results/lossesBefore.csv', delimiter=',')
# average = np.mean(data)
# print("Avg loss before attack:", average)
# data = np.loadtxt('./data/results/lossesAfter.csv', delimiter=',')
# average = np.mean(data)
# print("Avg loss after attack:", average)


# # Get mAP

# In[17]:


# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

# coco_gld = COCO(annFile_val) # coco
# # if modelv == 2:
# #     coco_rst = coco_gld.loadRes('./data/results/v2predictions.json')
# # elif modelv == 3:
# #     coco_rst = coco_gld.loadRes('./data/results/v3predictions.json')

# coco_rst = coco_gld.loadRes('./data/results/predictionsAfter.json')
# cocoEval = COCOeval(coco_gld, coco_rst, iouType='bbox')
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()

