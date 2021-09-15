from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import RestNet
from dataset import data
from tqdm import tqdm
import random
import copy
import os


def train_model(model, epochs, batch_size, data_path):
    total_data = [path for path in os.listdir(data_path)]
    random.shuffle(total_data)
    num_train = int(len(total_data)*0.8)
    train_data = total_data[:num_train]
    val_data = total_data[num_train:]
    train_set = data(train_data, True, data_path)
    val_set = data(val_data, False, data_path)

    dataloaders = {
        'train': DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True),
        'val': DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = filter(lambda p: p.requires_grad, model.parameters())
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(params)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        for phase in ['train', 'val']:
            flag = phase == 'train'
            if flag:
                model.train()
            else:
                model.eval()
            acc = 0
            total_loss = 0
            for i, (imgs, labels) in enumerate(tqdm(dataloaders[phase])):
                imgs = Variable(imgs.to(device))
                labels = Variable(labels.unsqueeze(1).to(device), requires_grad=False)

                optimizer.zero_grad()

                with torch.set_grad_enabled(flag):
                    output = model(imgs)
                    loss = criterion(output, labels)
                    total_loss += loss.item()
                    acc += torch.sum((output >= 0.5) == labels).item()
                    # print('epoch:{} loss:{} acc:{}'.format(epoch, Loss, acc))
                    if flag:
                        loss.backward()
                        optimizer.step()

            if flag:
                print('epoch:{} training_loss:{} training_acc:{}'.format(epoch + 1, str(total_loss / len(train_set)), str(acc / len(train_set))))
                with open('../log_b256/train_loss.txt', 'a+') as file:
                    file.write('epoch_' + str(epoch + 1) + ' ' + str(total_loss / len(train_set)) + '\n')
                with open('../log_b256/train_acc.txt', 'a+') as file:
                    file.write('epoch_' + str(epoch + 1) + ' ' + str(acc / len(train_set)) + '\n')
            else:
                epoch_val_acc = acc / len(val_set)
                if epoch_val_acc > best_acc:
                    best_acc = epoch_val_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, '../weights_256/epoch%d_acc%.3f.pth' % (epoch + 1, epoch_val_acc))
                print('epoch:{} val_loss:{} val_acc:{}'.format(epoch + 1, str(total_loss / len(val_set)), str(epoch_val_acc)))
                with open('../log_b256/val_loss.txt', 'a+') as file:
                    file.write('epoch_' + str(epoch + 1) + ' ' + str(total_loss / len(val_set)) + '\n')
                with open('../log_b256/val_acc.txt', 'a+') as file:
                    file.write('epoch_' + str(epoch + 1) + ' ' + str(epoch_val_acc) + '\n')


if __name__ == '__main__':
    # resnet = models.resnet152(pretrained=True)
    # resnet.fc = nn.Linear(2048, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = RestNet(pretrain=True).to(device)
    train_model(resnet, epochs=100, batch_size=256, data_path='../train')
