import torch.nn as nn
import numpy as np

class LeNet(nn.Module):
    def __init__(self, filter_size, fc_input):
        super(LeNet, self).__init__()
        self.filter_size = filter_size
        self.fc_input = fc_input
        
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 6, self.filter_size)
        self.conv2 = nn.Conv2d(6, 16, self.filter_size)
        self.fc = nn.Linear(self.fc_input, 120)  
        self.fc1 = nn.Linear(120, 84) 
        self.fc2 = nn.Linear(84, 200)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        return np.prod(size)