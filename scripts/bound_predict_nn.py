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

if len(sys.argv)!=3:
    print "%s data_dir superbounds/bounds/center" % sys.argv[0]
    sys.exit(1)

data_dir=sys.argv[1]
pred_ty=sys.argv[2]
if pred_ty not in ['superbounds','bounds','center']:
    print "ONLY TYPES ARE",'superbounds','bounds','center'
    sys.exit(1)

test_d={}
train_d={}

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



def load_dataset(dir):
    for root, dirs, files in os.walk(dir):
	path = root.split(os.sep)
	for file in files:
		fn=os.path.basename(file.replace('.png',''))
		partition=fn.split('_')[0]
		id=fn.split('_')[1]
		ty=fn.split('_')[2]
		p=train_d
		if partition=="test":
			p=test_d
		if id not in p:
			p[id]={}
		p[id][ty]=file

def n2t(im):
	return torch.from_numpy(np.transpose(im, (2, 0, 1)).astype(np.float)/255).float()

def t2n(im):
	n=im.cpu().data.numpy()
	n-=n.min()
	#n[n<0]=0
	div=255.0/max(1.0,n.max())
	return (np.transpose(n,(1,2,0))*div).astype(np.uint8)

def load_image(d,fn,c=-1):
	im=cv2.imread(d+'/'+fn) #[:,:,0] # all are BW
	if c>=0:
		im=im[:,:,c]
		im=im[:,:,None]
	return im

def blur(im):
	k=5
	blurred=im.copy()
	for x in xrange(3):
		new=cv2.GaussianBlur(blurred,(k,k),0)
		blurred=np.maximum(im,cv2.GaussianBlur(blurred,(k,k),0)[:,:,None])
	return blurred

def dilate(im):
	kernel=(3,3)
	dilated=cv2.dilate(im,kernel,iterations = 1)[:,:,None]
	return dilated


d=load_dataset(data_dir)

model = UNetClassify(layers=4, init_filters=32)
#model = MNET()

cuda=False
if cuda:
    model=model.cuda()

#im_torch=Variable(n2t(im_in).unsqueeze(0))
#print im_torch.size()
#out=model.forward(im_torch)
#print out.size()
#criterion=nn.BCELoss()
#criterion=nn.MSELoss()
criterion=nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
batch_size=8# 32
model.train()
it=0
bit=0
for batch_id in xrange(1000000):
    optimizer.zero_grad()
    mini_loss=0
    for img_id in xrange(batch_size):
        im_id=random.choice(train_d.keys())
        while pred_ty not in train_d[im_id]:# or load_image(data_dir,train_d[im_id]['seg']).shape[0]>350:
            im_id=random.choice(train_d.keys())
	it+=1
	#print load_image(data_dir,train_d[im_id]['seg']).shape

	im_in=load_image(data_dir,train_d[im_id]['seg'],c=0)
        im_out=None
        if pred_ty in ['center']:
	    im_out=load_image(data_dir,train_d[im_id][pred_ty],c=0)
        else:
	    im_out=dilate(load_image(data_dir,train_d[im_id][pred_ty],c=0))
	im_in, im_out = random_rotate90_transform2(im_in, im_out)
        im_in=Variable(n2t(im_in).unsqueeze(0))
        im_out=Variable(n2t(im_out).unsqueeze(0))
        if cuda:
            im_in=im_in.cuda()
            im_out=im_out.cuda()
        output = model(im_in)
        loss = criterion(output, im_out)
        mini_loss+=loss.data[0]/batch_size
	#visualize
	im_in_np=t2n(im_in[0])
	im_out_np=t2n(im_out[0])
	im_pred_np=t2n(output[0])
        if it%100==0:
		cv2.imwrite('out%d.png' % (np.random.randint(10)),np.concatenate((im_pred_np,im_out_np),axis=1))
	#cv2.waitKey(20)
    bit+=1
    if bit%100==1:
        save_checkpoint({
            'epoch': bit + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, True)
    loss.backward()
    print mini_loss
    optimizer.step()
