import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
import sys

gauss_blur=True
normalize=False
similarity=False

if len(sys.argv)!=2:
    print "%s d"
    sys.exit(1)

data_dir=sys.argv[1]

def load_dataset(dn):
    d={}
    for root, dirs, files in os.walk(dn):
        for name in files:
            if name[-4:]=='.png':
                i=root.split('/')[-2]
		if i not in d:
		    d[i]={'img':"",'masks':[]}
		t=root.split('/')[-1]
		if t in ['images']:
		    #its img
		    d[i]['img']=root+'/'+name
		elif t in ['masks']:
		    #its mask
		    d[i]['masks'].append(root+'/'+name)     
    return d

import random

nuclei=[]

def load_mask(d):
    for im_fn in d['masks']:
        im_cur = cv2.imread(im_fn)[:,:,0].copy()
        _,contours, _ = cv2.findContours(im_cur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            continue
        b_area=cv2.contourArea(contours[0])
        b_idx=0
        for cnt in xrange(1,len(contours)):
            if cv2.contourArea(contours[cnt])>b_area:
                b_idx=cnt
        x,y,w,h = cv2.boundingRect(contours[b_idx])
        if y<5 or y+h+5>im_cur.shape[0] or x<5 or x+w+5>im_cur.shape[1]:
            continue
        cropped=im_cur[y-2:y+h+2,x-2:x+w+2]
        #cv2.imshow('crop',cropped)
        #cv2.waitKey(5)
        outline=cv2.Laplacian(cropped,cv2.CV_8U,ksize=3)
        #outline=cv2.dilate(outline, (5,5), iterations=3)
        nuclei.append((cropped,outline))

ds=load_dataset(data_dir)

for k in ds:
    d=ds[k]
    load_mask(d)

#https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def make_data():
    out=None
    out_bounds=None
    while True:
	n1,n1b=random.choice(nuclei)
	n2,n2b=random.choice(nuclei)
	if n1.sum()>4*n2.sum() or n2.sum()>4*n1.sum():
	    continue
	n1r=random.randint(1,360)   
	n2r=random.randint(1,360)   
	n1=rotate_bound(n1,n1r)
	n1b=rotate_bound(n1b,n1r)
	n2=rotate_bound(n2,n2r)
	n2b=rotate_bound(n2b,n2r)
	h=4*max(n1.shape[0],n2.shape[0])
	w=4*max(n1.shape[1],n2.shape[1])
	out=np.zeros((h,w))
	out_bounds=np.zeros((h,w))
	print 'n1',n1.shape
	print 'n2',n2.shape
	print out_bounds,h,w

	ch=h/2
	cw=w/2
	n1h,n1w=n1.shape
	n2h,n2w=n2.shape
	print "C",ch,cw
	print n1.shape
	out[(ch-n1h/2):(ch-n1h/2+n1h),(cw-n1w/2):(cw-n1w/2+n1w)]=n1
	out_bounds[ch-n1h/2:ch-n1h/2+n1h,cw-n1w/2:cw-n1w/2+n1w]=n1b
	ch+=n1.shape[0]/3
	cw+=n1.shape[1]/3
	out[ch-n2h/2:ch-n2h/2+n2h,cw-n2w/2:cw-n2w/2+n2w]=np.maximum(out[ch-n2h/2:ch-n2h/2+n2h,cw-n2w/2:cw-n2w/2+n2w],n2)
	out_bounds[ch-n2h/2:ch-n2h/2+n2h,cw-n2w/2:cw-n2w/2+n2w]=np.maximum(out_bounds[ch-n2h/2:ch-n2h/2+n2h,cw-n2w/2:cw-n2w/2+n2w],n2b)
       
	#check that neither fully contain each other
	if n1.sum()+n2.sum()/2>out.sum() or n2.sum()+n1.sum()/2>out.sum():
	    continue
    	return out,out_bounds

out_dir="./gen_data"
i=0
while i<100000:
	out,out_bounds=make_data()
        print i
	cv2.imwrite(out_dir+'/'+str(i)+'_seg.png',out)
	cv2.imwrite(out_dir+'/'+str(i)+'_bounds.png',out_bounds)
        i+=1