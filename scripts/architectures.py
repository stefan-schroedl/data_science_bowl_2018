#!/usr/bin/env python

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import weight_norm

def init_weights(net, method='kaiming'):
    if method not in ['kaiming', 'xavier']:
        raise ValueError('no such init method: %s' % method)
    for m in net.modules():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if method == 'kaiming':
                m.weight.data = init.kaiming_normal(m.weight.data)
            else:
                init.xavier_uniform(m.weight)
            if m.bias is not None:
                init.constant(m.bias, 0)
            elif classname.find('BatchNorm') != -1 and m.weight and m.weight.data:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)
        else:
            if m != net:
                init_weights(m, method=method)


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
        
        # replicate to width x height
        out = out.unsqueeze(2).unsqueeze(2).expand(-1,-1,x.size()[-2],x.size()[-1])

        return out

class Coarse(nn.Module):
    def __init__(self):
        super(Coarse, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 3, stride=2, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=0))
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 3, stride=2, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=0))
        self.layer3 = nn.Sequential(
            nn.Conv2d(3, 3, stride=2, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=0))

    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)

        up = nn.Upsample((x.shape[-2],x.shape[-1]), mode='bilinear')
        l1out = up(l1)
        l2out = up(l2) 
        l3out = up(l3)
        return l1out, l2out, l3out

        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.mod = ImgModDetector()
        self.coarse = Coarse()

        # make the bn layer usable identically for train and test!
        affine = True
        mom = 0.0

        self.color_adjust = nn.Sequential(
            nn.Conv2d(6, 3, stride=1, kernel_size=1, padding=0),
            nn.Sigmoid(),
            nn.Conv2d(3, 3, stride=1, kernel_size=1, padding=0))

        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(15, affine=affine, momentum=mom),
            nn.Conv2d(15, 16, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1))
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(16, affine=affine, momentum=mom),
            nn.Conv2d(16, 16, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1))
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(16, affine=affine, momentum=mom),
            nn.Conv2d(16, 16, stride=1, kernel_size=3, padding=1),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.BatchNorm2d(16, affine=affine, momentum=mom),
            nn.Conv2d(16, 16, stride=1, kernel_size=3, padding=1),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.BatchNorm2d(16, affine=affine, momentum=mom),
            nn.Conv2d(16, 16, stride=1, kernel_size=3, padding=1),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.BatchNorm2d(16, affine=affine, momentum=mom),
            nn.Conv2d(16, 16, stride=1, kernel_size=3, padding=1),
            nn.ReLU())

        self.layer7 = nn.Sequential(
            nn.Conv2d(16, 1, stride=1, kernel_size=1, padding=0))


    def forward(self, x):
        img_tp = self.mod(x)
        img_and_type = torch.cat((x,img_tp),1)
        norm_img = self.color_adjust(img_and_type)
        c1, c2, c3 = self.coarse(norm_img)
        norm_img_and_type_and_coarse = torch.cat((norm_img,img_tp,c1,c2,c3),1)

        out = self.layer1(norm_img_and_type_and_coarse)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
       
        return out
    
    def get_color_adjust(self, x):
        img_tp = self.mod(x)
        img_and_type = torch.cat((x,img_tp),1)
        norm_img = self.color_adjust(img_and_type)
        return norm_img
    
    def get_coarse(self, x):
        img_tp = self.mod(x)
        img_and_type = torch.cat((x,img_tp),1)
        norm_img = self.color_adjust(img_and_type)
        c1, c2, c3 = self.coarse(norm_img)
        return c1, c2, c3
        
        
