#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Net2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.conv_loc = nn.Sequential(
            nn.Conv2d(10, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 6, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0.1)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        sz= xs.size()
        #xs = xs.view(-1, sz[1]*sz[2]*sz[3])
        theta = self.conv_loc(xs)
        theta = theta.mean(3)
        theta = theta.mean(2)
        theta /= 10
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 6, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(6, 8, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveMaxPool2d((10,10))
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(100*8, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        #self.fc_loc[2].weight.data.fill_(0.1)
        #self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        print "XXX", xs.size()
        xs = xs.view(-1,8*100)
        print "XS",xs.size()
        xs = self.fc_loc(xs)
        print "XS2",xs.size()
        return xs

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class STN():
    def __init__(self):
        self.nuclei=[]
        self.model=Net()
        
    def torch_from_numpy(self,img):
        return torch.from_numpy(img.transpose(2,0,1)).float()/255

    def numpy_from_torch(self,img):
        return (img.data.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)

    def torch_random_morph(self,x):
        theta = torch.FloatTensor([-1, 0, 0,0,-1, 0])
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x,theta

    def torch_rotate(self,x,angle):
        angle_in_rads = ((angle%360)/360.0)*2*np.pi
        theta = torch.FloatTensor([np.cos(angle_in_rads), -np.sin(angle_in_rads), 0,np.sin(angle_in_rads),np.cos(angle_in_rads), 0])
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x,theta

    def torch_batch_rotate(self,angles):
        angles_in_rads = ((angles%360)/360.0)*2*np.pi
        ms = np.zeros((angles.shape[0],2,3))
        ms[:,0,0]=np.cos(angles_in_rads)
        ms[:,0,1]=-np.sin(angles_in_rads)
        ms[:,1,0]=np.sin(angles_in_rads)
        ms[:,1,1]=np.cos(angles_in_rads)
        ms = torch.from_numpy(ms).float()
        return ms

    def fit(self):
        nuclei,nuclei_mask,nuclei_bound=self.nuclei[0]
        nuclei_bound_torch = self.torch_from_numpy(nuclei_bound).expand(1,-1,-1,-1)
        mb=32
        loss_fn = torch.nn.MSELoss(size_average=True)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for i in xrange(10000):
            angles = np.random.uniform(0,360,mb)
            rotation_ms = self.torch_batch_rotate(angles)
            targets = Variable(self.torch_batch_rotate(-angles))
            nuclei_bound_torch=nuclei_bound_torch.expand(mb,-1,-1,-1)
            grid = F.affine_grid(rotation_ms, nuclei_bound_torch.size())
            x = F.grid_sample(nuclei_bound_torch, grid)

            #xzero the grad
            optimizer.zero_grad()

            #fwd
            out=self.model.forward(x)
            out=out.view(-1,2,3)
            print out[0],targets[0]

            loss = loss_fn(out, targets)
            loss.backward()
            optimizer.step()
            print  loss.data[0]

            print out.size()
        #ok now we have batch of input data x, and targets 
        

        sys.exit(1)
        cv2.imshow('xx',nuclei_bound)
        out = self.model(nuclei_bound_torch)
        img = (out.data.numpy()[0].transpose(1,2,0)).astype(np.uint8)
        print img.mean()
        cv2.imshow('img',img)
        cv2.imshow('nc',nuclei_bound)
        cv2.waitKey(50000)
        pass

    def get_stacked(self,img,mask,mask_seg):
        kernel = np.ones((3,3), np.uint8)
        mask_eroded=cv2.erode(mask, kernel, iterations=1)[:,:,None].astype(np.uint8)
        
        super_boundary = mask.copy()[:,:,0]*0
        super_boundary_2 = mask.copy()[:,:,0]*0
        max_components=mask.max()
        kernel = np.ones((5,5), np.uint8)
        for x in xrange(max_components):
            this_one = ((mask==(x+1))*255).astype(np.uint8)[:,:,0]
            boundary = cv2.Laplacian(this_one,cv2.CV_8U,ksize=3)
            super_boundary = np.maximum(super_boundary,boundary)

            #super boundary 2 by dilation of border
            #boundary = cv2.dilate(boundary, kernel, iterations=3)
            #_,boundary_thresh = cv2.threshold(boundary,100,255,cv2.THRESH_BINARY)
            #super_boundary_2 += boundary_thresh/255

            #super boundary 2 by dilation of mask
            boundary = cv2.dilate(this_one, kernel, iterations=1)
            _,boundary_thresh = cv2.threshold(boundary,100,255,cv2.THRESH_BINARY)
            super_boundary_2 += boundary_thresh/255
        #print "X",super_boundary_2.max()
        super_boundary_2 = ((super_boundary_2>1)*255).astype(np.uint8)
        #cv2.imshow('sup 2',np.concatenate((super_boundary_2,mask_seg[:,:,0])))
        #cv2.waitKey(3000)
        boundary = cv2.Laplacian(mask_seg,cv2.CV_8U,ksize=3)
        boundary = boundary.reshape(boundary.shape[0],boundary.shape[1],1)
        assert(boundary.max()<=255)
        #0-2 RGB
        #3 SEG
        #4 BOUNDARY
        #5 BLEND
        #6 SUPER BOUNDARY
        #7 SUPER BOUNDARY 2
        stacked_img = np.concatenate((img,mask_seg,boundary,np.maximum(mask_seg/2,boundary),super_boundary[:,:,None],super_boundary_2[:,:,None],mask_eroded),axis=2)
        return stacked_img.astype(np.uint8)

    def crop_minAreaRect(self,img, rect):
        # rotate img
        angle = rect[2]
        rows,cols = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        img_rot = cv2.warpAffine(img,M,(cols,rows))

        # rotate bounding box
        rect0 = (rect[0], rect[1], 0.0)
        box = cv2.boxPoints(rect)
        pts = np.int0(cv2.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0

        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1],
                           pts[1][0]:pts[2][0]]

        return img_crop

    def prepare_fit(self,img,mask,mask_seg):
        img = (img.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        mask = (mask.numpy()[0].transpose(1,2,0)).astype(np.uint8)
        mask_seg = (mask_seg.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        stacked=self.get_stacked(img,mask,mask_seg)
        #blurred boundary generation
        boundary=stacked[:,:,4]
        k=9
        blurred_boundary=np.maximum(boundary,cv2.GaussianBlur(boundary,(k,k),0))
        for x in xrange(10):
            blurred_boundary=np.maximum(boundary,cv2.GaussianBlur(blurred_boundary,(k,k),0))
        blurred_boundary=blurred_boundary.astype(np.float)
        blurred_boundary/=blurred_boundary.max()
        blurred_boundary*=255
        blurred_boundary=blurred_boundary.astype(np.uint8)
        for x in xrange(mask.max()):
            idx=x+1
            cur_mask = np.zeros_like(mask).astype(np.uint8)
            cur_mask[mask==idx]=1
            sub_img=np.multiply(cur_mask,img).astype(np.uint8)
            sub_bound=np.multiply(cur_mask,blurred_boundary[:,:,None]).astype(np.uint8)
            self.nuclei.append((sub_img,cur_mask*255,sub_bound))
            #_, contours, hierarchy = cv2.findContours(cur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #for cnt in contours:
            #    rect = cv2.minAreaRect(cnt)
            #    ximg=self.crop_minAreaRect(sub_img,rect)
            #    ximg_mask=self.crop_minAreaRect(cur_mask,rect)
            #    ximg_bound=self.crop_minAreaRect(sub_bound,rect)
            #    #cv2.imshow("BOUND",ximg_bound)
            #    #cv2.waitKey(5000)
            #    if ximg.shape[0]<5 or ximg.shape[1]<5:
            #        continue
            #    self.nuclei.append((ximg,ximg_mask,ximg_bound))

    def predict(self,img):
        img = (img.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
         

