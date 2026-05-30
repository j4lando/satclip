import torchvision.transforms as T
import torch
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform  
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transform(resize_crop_size = 256,
                  mean = [0.02758, 0.03163, 0.02678, 0.0642 ],
                  std = [0.01754, 0.01855, 0.01973, 0.04098]
                  ):

    augmentation = A.Compose(
        [
            A.RandomResizedCrop(size=(resize_crop_size, resize_crop_size)),
            A.RandomBrightnessContrast(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GaussianBlur(),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    def transform(sample):
        image = np.asarray(sample["image"]).transpose(1,2,0)
        point = sample["point"]

        image = augmentation(image=image)["image"]
        point = coordinate_jitter(point)

        return dict(image=image, point=point)

    return transform

def get_s2_train_transform(resize_crop_size = 256):
    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.GaussianBlur(3),
    ])

    def transform(sample):
        image = sample["image"]
        point = sample["point"]
        image = torch.tensor(image)
        image = augmentation(image)
        point = coordinate_jitter(point)
        return dict(image=image, point=point)

    return transform

def get_pretrained_s2_train_transform(resize_crop_size = 256):
    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.GaussianBlur(3),
    ])

    def transform(sample):
        image = sample["image"]
        point = sample["point"]

        B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
        image = np.concatenate([image[:10], B10, image[10:]], axis=0)
        image = torch.tensor(image)

        image = augmentation(image)

        point = coordinate_jitter(point)

        return dict(image=image, point=point)

    return transform

def coordinate_jitter(
        point,
        radius=0.001 # approximately 100 m
    ):
    return point + torch.rand(point.shape) * radius