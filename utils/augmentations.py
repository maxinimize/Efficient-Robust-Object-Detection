import imgaug.augmenters as iaa
from torchvision import transforms
from utils.transforms import Resize, ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug, ConvertToArrays


class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ])


class StrongAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ])


AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    DefaultAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

class MyCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, tar):
        data = (img, tar)
        for t in self.transforms:
            data = t(data)
        img, tar = data
        return img, tar
    
TRANSFORM_TRAIN = MyCompose([
    DefaultAug(),
    PadSquare(),
    RelativeLabels(),
    ConvertToArrays(),
])

TRANSFORM_VAL = MyCompose([
    ConvertToArrays(),
    # DefaultAug(),
    PadSquare(),
    ToTensor(),
    RelativeLabels(),
    Resize(416),
])