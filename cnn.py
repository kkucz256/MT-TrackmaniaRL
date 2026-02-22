import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackmaniaCNN(nn.Module):
    def __init__(self, input_channels=4, num_actions=4):
        super(TrackmaniaCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc_input_dim = 64 * 5 * 5
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values