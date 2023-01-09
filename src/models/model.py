from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        # Now with dropout
        x1 = self.dropout(F.relu(self.fc1(x)))
        x2 = self.dropout(F.relu(self.fc2(x1)))

        # save intermediate features after second layer
        self.features_fc2 = x2

        x3 = self.dropout(F.relu(self.fc3(x2)))
        
        # output so no dropout here
        x4 = F.log_softmax(self.fc4(x3), dim=1)
        
        return x4



