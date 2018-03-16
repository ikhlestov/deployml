import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


np.random.seed(42)
torch.manual_seed(42)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.conv1_1 = nn.Conv2d(3, 6, 3)
        self.bn_1_1 = nn.BatchNorm2d(6)
        self.conv1_2 = nn.Conv2d(6, 9, 3)
        self.bn_1_2 = nn.BatchNorm2d(9)
        self.conv2_1 = nn.Conv2d(9, 12, 3)
        self.bn_2_1 = nn.BatchNorm2d(12)
        self.conv2_2 = nn.Conv2d(12, 16, 3)
        self.bn_2_2 = nn.BatchNorm2d(16)
        self.conv3_1 = nn.Conv2d(16, 32, 3)
        self.bn_3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(32, 64, 3)
        self.bn_3_2 = nn.BatchNorm2d(64)
        self.conv4_1 = nn.Conv2d(64, 128, 3)
        self.bn_4_1 = nn.BatchNorm2d(128)
        self.conv4_2 = nn.Conv2d(128, 256, 3)
        self.bn_4_2 = nn.BatchNorm2d(256)
        self.last_avg_pool = nn.AvgPool2d(20)
        self.fc_1 = nn.Linear(256, 512)
        self.fc_bn = nn.BatchNorm2d(512)
        self.fc_2 = nn.Linear(512, 100)
        super().eval()

    def forward(self, x):
        # first block
        x = self.conv1_1(x)
        x = F.relu(self.bn_1_1(x))
        x = self.conv1_2(x)
        x = F.relu(self.bn_1_2(x))
        x = self.avg_pool(x)

        # second block
        x = self.conv2_1(x)
        x = F.relu(self.bn_2_1(x))
        x = self.conv2_2(x)
        x = F.relu(self.bn_2_2(x))
        x = self.avg_pool(x)

        # third block
        x = self.conv3_1(x)
        x = F.relu(self.bn_3_1(x))
        x = self.conv3_2(x)
        x = F.relu(self.bn_3_2(x))
        x = self.avg_pool(x)

        # forth block
        x = self.conv4_1(x)
        x = F.relu(self.bn_4_1(x))
        x = self.conv4_2(x)
        x = F.relu(self.bn_4_2(x))

        # transition to classes
        x = self.last_avg_pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc_1(x)
        x = self.fc_bn(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

    def predict(self, x):
        x = torch.from_numpy(x)
        x = x.float()
        x = Variable(x)
        x = self.forward(x)
        x = F.sigmoid(x)
        return x


if __name__ == '__main__':
    sample_image = np.random.random((1, 3, 224, 224))
    model = Model()
    preds = model.predict(sample_image)
    print(model)
