import torchvision.models as models
import torch.nn as nn


class RestNet(nn.Module):
    def __init__(self, pretrain):
        super(RestNet, self).__init__()
        resnet = models.resnet152(pretrained=pretrain)
        # for param in resnet.parameters():
        #     param.requires_grad = False
        # resnet.conv1 = nn.Conv2d(3, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # resnet.bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # resnet.layer1[0].conv1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # resnet.layer1[0].downsample[0] = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # resnet.layer1[0].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        resnet.fc = nn.Linear(2048, 1)
        self.resnet = resnet
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)
