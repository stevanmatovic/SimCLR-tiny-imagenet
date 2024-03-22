
import torch
import torchvision.transforms as transforms

# Define data augmentation
def get_color_distortion(s=0.5):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    get_color_distortion(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class PairTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)

