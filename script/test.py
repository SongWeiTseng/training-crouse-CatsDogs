import torch
import torch.nn as nn
from model import RestNet
from dataset_test import data
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import numpy as np
import os


def test_model(model, batch_size, data_path):
    file_name = [name for name in os.listdir(data_path)]
    dataset = data(file_name, ontrain=False, path=data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # criterion = nn.BCELoss()

    acc = 0
    # total_loss = 0
    arr = np.zeros(len(dataset))
    for i, (imgs, name) in enumerate(tqdm(dataloader)):
        model.eval()

        imgs = imgs.to(device)
        # labels = labels.unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(imgs)
            output = output.numpy()

            for j, d in enumerate(name):
                arr[int(d) - 1] = output[j]

    with open('submission_adam.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        row = ['id', 'label']
        writer.writerow(row)
        for i, d in enumerate(arr):
            row = [i+1, d]
            writer.writerow(row)
            # with open('sample_submission.csv', 'a+') as f:
            #     reader = csv.reader(f, delimiter=',')
            #     for d in reader:
            #         print(d)
            # loss = criterion(output, labels)
            # total_loss += loss.item()
            # acc += torch.sum((output >= 0.5) == labels).item()
    # print('acc:{}'.format(str(acc/len(dataset))))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = RestNet(pretrain=False).to(device)
    resnet.load_state_dict(torch.load('../weights/epoch63_acc0.923.pth', map_location='cpu'))
    test_model(resnet, 16, '../test')
