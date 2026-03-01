import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackmaniaNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(TrackmaniaNet, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_out_size = 64 * 12 * 12 
        
        self.tele_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        combined_size = cnn_out_size + 64
        
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 9)
        )

    def forward(self, vision, telemetry):
        x_vision = vision.permute(0, 3, 1, 2).float() / 255.0
        vision_features = self.cnn(x_vision)
        tele_features = self.tele_mlp(telemetry.float())
        combined = torch.cat((vision_features, tele_features), dim=1)
        return self.fc(combined)