import torch.nn as nn

class MNET(nn.Module):
    def __init__(self):
        # conv1
	super(MNET, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 16, 5, padding=2)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(16, 16, 5, padding=2)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv1_3 = nn.Conv2d(16, 8, 5, padding=2)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv1_4 = nn.Conv2d(8, 8, 5, padding=2)
        self.relu1_4 = nn.ReLU(inplace=True)
        self.conv1_5 = nn.Conv2d(8, 8, 5, padding=2)
        self.relu1_5 = nn.ReLU(inplace=True)
        self.conv1_6 = nn.Conv2d(8, 8, 5, padding=2)
        self.relu1_6 = nn.ReLU(inplace=True)
        self.conv1_7 = nn.Conv2d(8, 8, 5, padding=2)
        self.relu1_7 = nn.ReLU(inplace=True)
        self.conv1_8 = nn.Conv2d(16, 1, 5, padding=2)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        #h = self.relu1_2(self.conv1_2(h))
        #h = self.relu1_3(self.conv1_3(h))
        #h = self.relu1_4(self.conv1_4(h))
        #h = self.relu1_5(self.conv1_5(h))
        #h = self.relu1_6(self.conv1_6(h))
        #h = self.relu1_7(self.conv1_7(h))
	h = self.conv1_8(h)
        return h

