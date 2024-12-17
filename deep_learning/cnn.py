import torch.nn as nn
import torch.nn.functional as F

# cnn with pooling (comment out pooling lines to remove pooling (remember to recalculate dimensions))
class cnn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # 3x224x224 -> 1x6x251x251
        self.pool1 = nn.MaxPool2d(2,2) # 1x6x125x125
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # 1x16x121x121
        self.pool2 = nn.MaxPool2d(2,2) # 1x16x60x60
        self.fc1 = nn.Linear(16 * 53* 53, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        softmax = F.log_softmax(x, 1)
        return softmax