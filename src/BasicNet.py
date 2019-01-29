import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        # Basic two-layer network
        self.fc1 = nn.Linear(28 * 28, 10)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        # Get the batch size
        in_size = x.size(0)
        # Flatten data, -1 is inferred from the other dimensions
        x = x.view(in_size, -1)

        # Forward rule
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc1(x)

        # Softmax on predictions
        x = F.softmax(x, dim=1)

        return x
