#%pycat architectures.py

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)[source]


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ImgModDetector(nn.Module):
    def __init__(self):
        super(ImgModDetector, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 3, stride=1, kernel_size=1, padding=0))
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 10, stride=1, kernel_size=1, padding=0),
            nn.Sigmoid())
        self.fc = nn.Linear(10,3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        mx = nn.Sequential(nn.MaxPool2d((out.size()[-2],out.size()[-1])))
        out = mx(out).squeeze(-1).squeeze(-1)
        out = self.fc(out)
        return out
        
        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.mod = ImgModDetector()
        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 16, stride=1, kernel_size=3, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, stride=1, kernel_size=3, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, stride=1, kernel_size=3, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 1, stride=1, kernel_size=1, padding=0))

    def forward(self, x):
        img_tp = self.mod(x)
        #print 'tp', img_tp.size()
        img_tp = img_tp.unsqueeze(2).unsqueeze(2).expand(-1,-1,x.size()[-2],x.size()[-1])
        #print 'tp2', img_tp.size()
        #print 'x', x.size()
        conc = torch.cat((x,img_tp),1)
        #print 'conc', conc.size()
        out = self.layer1(conc)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
