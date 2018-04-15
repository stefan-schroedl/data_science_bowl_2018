import cv2
import sys
import shutil
from nuc_trans  import random_rotate90_transform2
from architectures import UNetClassify
import numpy as np
import torch
from torch.autograd import Variable
import cv2
import random
import os
import sys
import torch.optim as optim
from mnet import MNET
import torch.nn.functional as F
import torch.nn as nn

def n2t(im):
	return torch.from_numpy(np.transpose(im, (2, 0, 1)).astype(np.float)/255).float()

def t2n(im):
	n=im.cpu().data.numpy()
	n-=n.min()
	#n[n<0]=0
	div=255.0/max(1.0,n.max())
	return (np.transpose(n,(1,2,0))*div).astype(np.uint8)

if len(sys.argv)!=4:
    print sys.argv[0] + "model_fn in_fn im_fn_out"
    sys.exit(1)

model_fn=sys.argv[1]
fn=sys.argv[2]
fn_out=sys.argv[3]
checkpoint = torch.load(model_fn, map_location='cpu') # always load to cpu first!


model = UNetClassify(layers=4, init_filters=32)
model.load_state_dict(checkpoint['state_dict'])
im_in=cv2.imread(fn)[:,:,:1] # get the gray scale
im_in[im_in>0]=255

print im_in.shape

in_im_t=Variable(n2t(im_in).unsqueeze(0))
output=model(in_im_t)
im_pred_np=t2n(output[0])
cv2.imwrite(fn_out,im_pred_np)

