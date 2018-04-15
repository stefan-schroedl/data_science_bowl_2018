#!/usr/bin/env python

import math
import logging
from collections import OrderedDict

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import weight_norm

from groupnorm import GroupNorm

INPLACE = True

USE_GROUPNORM = True

def norm_layer(num_filters, dummy=None):
    if USE_GROUPNORM:
        groups = num_filters
        return GroupNorm(num_filters, groups)
    else:
        affine = True
        mom = 0.0
        return nn.BatchNorm2d(num_filters)


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

def fwd_hook(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    print 'Inside ' + self.__class__.__name__ + ' forward'
    print ''
    print 'children:'
    for k,v in self.named_modules():
        print k, v.__class__.__name__
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())


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
            nn.ReLU(inplace=INPLACE),
            nn.MaxPool2d(3, stride=2, padding=0))
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 3, stride=2, kernel_size=2, padding=0),
            nn.ReLU(inplace=INPLACE),
            nn.MaxPool2d(3, stride=2, padding=0))
        self.layer3 = nn.Sequential(
            nn.Conv2d(3, 3, stride=2, kernel_size=2, padding=0),
            nn.ReLU(inplace=INPLACE),
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
    def __init__(self, num_filters=16):
        super(CNN, self).__init__()
        self.mod = ImgModDetector()
        self.coarse = Coarse()

        # make the bn layer usable identically for train and test!
        affine = True
        mom = 0.0
        groups = num_filters

        self.color_adjust = nn.Sequential(
            nn.Conv2d(6, 3, stride=1, kernel_size=1, padding=0),
            nn.Sigmoid(),
            nn.Conv2d(3, 3, stride=1, kernel_size=1, padding=0))

        self.layer1 = nn.Sequential(
            #nn.BatchNorm2d(15, affine=affine, momentum=mom),
            norm_layer(15,15),
            nn.Conv2d(15, num_filters, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=INPLACE),
            nn.MaxPool2d(3, stride=1, padding=1))
        self.layer2 = nn.Sequential(
            #nn.BatchNorm2d(num_filters, affine=affine, momentum=mom),
            norm_layer(num_filters, groups),
            nn.Conv2d(num_filters, num_filters, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=INPLACE),
            nn.MaxPool2d(3, stride=1, padding=1))
        self.layer3 = nn.Sequential(
            #nn.BatchNorm2d(num_filters, affine=affine, momentum=mom),
            norm_layer(num_filters, groups),
            nn.Conv2d(num_filters, num_filters, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=INPLACE))
        self.layer4 = nn.Sequential(
            #nn.BatchNorm2d(num_filters, affine=affine, momentum=mom),
            norm_layer(num_filters, groups),
            nn.Conv2d(num_filters, num_filters, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=INPLACE))
        self.layer5 = nn.Sequential(
            #nn.BatchNorm2d(num_filters, affine=affine, momentum=mom),
            norm_layer(num_filters, groups),
            nn.Conv2d(num_filters, num_filters, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=INPLACE))
        self.layer6 = nn.Sequential(
            #nn.BatchNorm2d(num_filters, affine=affine, momentum=mom),
            norm_layer(num_filters, groups),
            nn.Conv2d(num_filters, num_filters, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=INPLACE))
        self.layer7 = nn.Sequential(
            nn.Conv2d(num_filters, 1, stride=1, kernel_size=1, padding=0))

        # self.register_forward_hook(fwd_hook)

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

## UNET
## https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c
## (adapted from python 3, and gray scale images)

DROPOUT = 0.5
k=3
class UNetBlock(nn.Module):
    def __init__(self, filters_in, filters_out):
        super(UNetBlock, self).__init__()
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.conv1 = nn.Conv2d(filters_in, filters_out, (k, k), padding=k/2)
        #self.norm1 = nn.BatchNorm2d(filters_out)
        self.norm1 = GroupNorm(filters_out, filters_out)
        self.conv2 = nn.Conv2d(filters_out, filters_out, (k, k), padding=k/2)
        #self.norm2 = nn.BatchNorm2d(filters_out)
        self.norm2 = norm_layer(filters_out, filters_out)

        self.activation = nn.ReLU(inplace=INPLACE)

    def forward(self, x):
        conved1 = self.conv1(x)
        conved1 = self.activation(conved1)
        conved1 = self.norm1(conved1)
        conved2 = self.conv2(conved1)
        conved2 = self.activation(conved2)
        conved2 = self.norm2(conved2)
        return conved2

# note: python pickle doesn't like lambdas!
def noop(x):
    return x

class UNetDownBlock(UNetBlock):
    def __init__(self, filters_in, filters_out, pool=True):
        super(UNetDownBlock, self).__init__(filters_in, filters_out)
        if pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = noop

    def forward(self, x):
        return self.pool(super(UNetDownBlock, self).forward(x))

class UNetUpBlock(UNetBlock):
    def __init__(self, filters_in, filters_out):
        super(UNetUpBlock, self).__init__(filters_in, filters_out)
        self.upconv = nn.Conv2d(filters_in, filters_in // 2, (k, k), padding=k/2)
        #self.upnorm = nn.BatchNorm2d(filters_in // 2)
        self.upnorm = norm_layer(filters_in // 2, filters_in // 2)

    def forward(self, x, cross_x):
        x = F.upsample(x, size=cross_x.size()[-2:], mode='bilinear')
        x = self.upnorm(self.activation(self.upconv(x)))
        x = torch.cat((x, cross_x), 1)
        return super(UNetUpBlock, self).forward(x)

class UNet(nn.Module):
    def __init__(self, layers, init_filters):
        super(UNet, self).__init__()
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.init_filters = init_filters

        filter_size = init_filters
        for _ in range(layers - 1):
            self.down_layers.append(
                UNetDownBlock(filter_size, filter_size*2)
            )
            filter_size *= 2
        self.down_layers.append(UNetDownBlock(filter_size, filter_size * 2, pool=False))
        for i in range(layers):
            self.up_layers.append(
                UNetUpBlock(filter_size * 2, filter_size)
            )
            filter_size //= 2

        # new
        #self.squash_layer = nn.Conv2d(3, 1, stride=1, kernel_size=1, padding=0)
        #self.data_norm = nn.BatchNorm2d(1)
        #self.data_norm = GroupNorm(3,3)
        #self.init_layer = nn.Conv2d(3, init_filters, (7, 7), padding=3)
        self.init_layer = nn.Conv2d(1, init_filters, (7, 7), padding=3)
        self.activation = nn.ReLU(inplace=INPLACE)
        #self.init_norm = nn.BatchNorm2d(init_filters)
        self.init_norm = norm_layer(init_filters, init_filters)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        #x = self.squash_layer(x)
        #x = self.data_norm(x)
        x = self.init_norm(self.activation(self.init_layer(x)))

        saved_x = [x]
        for layer in self.down_layers:
            saved_x.append(x)
            x = layer(x) #self.dropout(layer(x))
        is_first = True
        for layer, saved_x in zip(self.up_layers, reversed(saved_x)):
            if not is_first:
                is_first = False
                #x = self.dropout(x)
            x = layer(x, saved_x)
        return x

class UNetClassify(UNet):
    def __init__(self, *args, **kwargs):
        init_val = kwargs.pop('init_val', 0.5)
        super(UNetClassify, self).__init__(*args, **kwargs)
        self.output_layer = nn.Conv2d(self.init_filters, 1, (3, 3), padding=1)

        for name, param in self.named_parameters():
            typ = name.split('.')[-1]
            if typ == 'bias':
                if 'output_layer' in name:
                    # Init so that the average will end up being init_val
                    param.data.fill_(-math.log((1-init_val)/init_val))
                else:
                    param.data.zero_()

    def forward(self, x):
        x = super(UNetClassify, self).forward(x)
        # Note that we don't perform the sigmoid here.
        return self.output_layer(x)


class UNetClassifyMulti(UNet):
    def __init__(self, targets, *args, **kwargs):
        init_val = kwargs.pop('init_val', 0.5)
        super(UNetClassifyMulti, self).__init__(*args, **kwargs)
        self.output_layers = {}
        self.target_names = []
        for target in targets:
            conv = nn.Conv2d(self.init_filters, 1, (3, 3), padding=1)
            name = target['name']
            self.target_names.append(name)
            self.add_module(name, conv) # needed such that cuda() etc finds submodules!

        for name, param in self.named_parameters():
            typ = name.split('.')[-1]
            if typ == 'bias':
                if 'output_layer' in name:
                    # Init so that the average will end up being init_val
                    param.data.fill_(-math.log((1-init_val)/init_val))
                else:
                    param.data.zero_()

    def forward(self, x):
        x = super(UNetClassifyMulti, self).forward(x)
        # Note that we don't perform the sigmoid here.
        pred = OrderedDict()
        for n in self.target_names:
            pred[n] = self.__getattr__(n)(x)
        return pred
