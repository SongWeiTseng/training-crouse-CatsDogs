import torchvision.models as models
import torch.nn as nn


class RestNet(nn.Module):
    def __init__(self):
        super(RestNet, self).__init__()
        resNet = models.resnet152(pretrained=True)
        resNet.fc = nn.Linear(2048, 1)
        self.resNet = resNet
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resNet(x)
        return self.sigmoid(x)
