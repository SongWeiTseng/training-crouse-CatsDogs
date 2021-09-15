import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os


def get_mean_std():
    data_transforms = transforms.Compose([
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
        transforms.Resize([256, 256])
    ])
    means_r, means_g, means_b = [], [], []
    stds_r, stds_g, stds_b = [], [], []
    for name in tqdm(os.listdir('../train')):
        # img = pad2square(Image.open('../train/{}'.format(name)).convert('RGB'))
        img = Image.open('../train/{}'.format(name)).convert('RGB')
        img = data_transforms(img)
        means_r.append(torch.mean(img[0]))
        means_g.append(torch.mean(img[1]))
        means_b.append(torch.mean(img[2]))
        stds_r.append(torch.std(img[0]))
        stds_g.append(torch.std(img[1]))
        stds_b.append(torch.std(img[2]))

    print(torch.mean(torch.tensor(means_r)))
    print(torch.mean(torch.tensor(means_g)))
    print(torch.mean(torch.tensor(means_b)))

    print(torch.mean(torch.tensor(stds_r)))
    print(torch.mean(torch.tensor(stds_g)))
    print(torch.mean(torch.tensor(stds_b)))


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
    def __init__(self, filename, ontrain, path):
        self.filename = filename
        self.datapath = path
        # self.label = torch.FloatTensor([0 if 'cat' in name else 1 for name in filename])
        self.image = []
        self.ontrain = ontrain

    def __getitem__(self, index):
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.3814, 0.3553, 0.3254], std=[0.2827, 0.2704, 0.2604]),
            transforms.Normalize(mean=[0.4883, 0.4551, 0.4170], std=[0.2276, 0.2230, 0.2233]),
            transforms.Resize([64, 64])
        ])

        img = pad2square(Image.open('{}/{}'.format(self.datapath, self.filename[index])).convert('RGB'))
        img = data_transforms(img)
        if self.ontrain:
            transforms.RandomHorizontalFlip()(img)

        return img, self.filename[index][:-4]  # , self.label[index]

    def __len__(self):
        return len(self.filename)


if __name__ == '__main__':
    get_mean_std()
