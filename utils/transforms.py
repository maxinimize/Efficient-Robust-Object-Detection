import torch
import torch.nn.functional as F
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from .utils import xywh2xyxy_np
import torchvision.transforms as transforms

#* modified transforms
class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, img, boxes):
        
        # np.set_printoptions(linewidth=500)
        # np.set_printoptions(suppress=True)
        # print("before padsquare", boxes.shape)
        # for box in boxes: print(box)
        
        
        if boxes.size != 0:
            if boxes.ndim > 2: 
                boxes = boxes.squeeze()
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)
                
            
            # Convert xywh to xyxy
            boxes = np.array(boxes)
            # image_ids = boxes[:, 0]
            boxes[:, 2:] = xywh2xyxy_np(boxes[:, 2:])
            # print("1", boxes.shape)
        
        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[2:], label=box[0:1]) for box in boxes],
            shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()
        

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 6))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0:1] = box.label
            boxes[box_idx, 2] = ((x1 + x2) / 2)
            boxes[box_idx, 3] = ((y1 + y2) / 2)
            boxes[box_idx, 4] = (x2 - x1)
            boxes[box_idx, 5] = (y2 - y1)
            
        # print(image_ids.shape, boxes.shape)    
        # boxes = np.hstack((image_ids[:, np.newaxis], boxes))
        if boxes.shape[0] == 0: boxes = boxes[:, 0]
        return img, boxes


class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, img, boxes):
        if boxes.shape[0] == 0: return img, boxes
        h, w, _ = img.shape
        if boxes.ndim > 2: 
            boxes = boxes.squeeze()
        if boxes.ndim < 2:
            boxes = boxes.unsqueeze(0)
        boxes[:, [2, 4]] /= w
        boxes[:, [3, 5]] /= h
        return img, boxes


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes


class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, img, boxes):
        img = transforms.ToTensor()(img)
        boxes = torch.tensor(boxes)
        return img, boxes


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, boxes):
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes


#* original transforms
class ImgAugEval(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = ((x1 + x2) / 2)
            boxes[box_idx, 2] = ((y1 + y2) / 2)
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return img, boxes
    
class ResizeEval(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes   
        
class RelativeLabelsEval(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes
    
class AbsoluteLabelsEval(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes
    
class PadSquareEval(ImgAugEval):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])
        
class ToTensorEval(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets
    
class ConvertToArrays():
    def __init__(self, ):
        pass
    
    def __call__(self, img, boxes):
        transformed_samples = []
        for box in boxes:
            # Extract the required fields and create the array
            transformed_sample = np.array([
                box['image_id'],
                box['category_id'],
                *box['bbox']
            ])
            transformed_samples.append(transformed_sample)
        # np.set_printoptions(suppress=True)
        # print("original")
        # for box in transformed_samples: print(box)
        return np.array(img), np.array(transformed_samples)


DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabelsEval(),
    PadSquareEval(),
    RelativeLabelsEval(),
    ToTensorEval(),
])
