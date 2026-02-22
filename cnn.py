import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackmaniaCNN(nn.Module):
    def __init__(self, input_channels=4, num_actions=4):
        super(TrackmaniaCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc_input_dim = 64 * 5 * 5
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):
        if x.max() > 1.0:
            x = x / 255.0
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        q_values = self.fc3(x)
        
        return q_values