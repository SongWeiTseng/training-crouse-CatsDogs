import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


def pad2square(img):
    w, h = img.size
    if w == h:
        return img
    diff = np.abs(w-h)
    if w > h:
        canvas = Image.new(img.mode, (w, w))
        canvas.paste(img, (0, diff//2))
        return canvas
    canvas = Image.new(img.mode, (h, h))
    canvas.paste(img, (diff//2, 0))
    # canvas.show()
    return canvas


class data(Dataset):
    def __init__(self, path, ontrain):
        self.filename = path
        self.label = torch.FloatTensor([0 if 'cat' in filename else 1 for filename in path])
        self.image = []
        self.ontrain = ontrain

    def __getitem__(self, index):
        data_transforms = transforms.Compose([
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor(),
            transforms.Resize([64, 64])
        ])

        img = pad2square(Image.open('../train/{}'.format(self.filename[index])).convert('RGB'))
        img = data_transforms(img)
        if self.ontrain:
            transforms.RandomHorizontalFlip()(img)
        return img, self.label[index]

    def __len__(self):
        return len(self.filename)
