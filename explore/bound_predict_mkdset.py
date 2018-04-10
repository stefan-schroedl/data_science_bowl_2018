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

if len(sys.argv)!=3:
    print "%s train_dir test_dir"
    sys.exit(1)

train_dir=sys.argv[1]
test_dir=sys.argv[2]

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

train_d=load_dataset(train_dir)
test_d=load_dataset(train_dir)

print "TRY LEARNING CENTERS"

def load_mask(d):
    im=np.zeros((1,1))
    bounds=np.zeros((1,1))
    super_bounds=np.zeros((1,1))
    for im_fn in d['masks']:
        im_cur = cv2.imread(im_fn)[:,:,0]
        if im.sum()==0:
            im=np.zeros(im_cur.shape[:2])
            bounds=np.zeros(im_cur.shape[:2])
            super_bounds=np.zeros(im_cur.shape[:2])
        im=np.maximum(im,im_cur[:,:])
        outline=cv2.Laplacian(im_cur,cv2.CV_8U,ksize=3)
        bounds=np.maximum(bounds,outline)
        outline=cv2.dilate(outline, (5,5), iterations=3)
        super_bounds=np.minimum(2,super_bounds+outline/255).astype(np.uint8)
    im=im.reshape(im.shape[0],im.shape[1],1)
    bounds=bounds.reshape(im.shape[0],im.shape[1],1)
    super_bounds[super_bounds<2]=0
    super_bounds[super_bounds>1]=255
    super_bounds=cv2.dilate(super_bounds, (3,3), iterations=1)
    super_bounds=super_bounds.reshape(im.shape[0],im.shape[1],1)
    return im,bounds,super_bounds

outd='mask_bounds_data'
for k in train_d:
    d=train_d[k]
    mask,bounds,super_bounds=load_mask(d)
    cv2.imwrite(outd+'/train_'+k+'_seg.png',mask)
    cv2.imwrite(outd+'/train_'+k+'_bounds.png',bounds)
    if super_bounds.sum()>0:
        cv2.imwrite(outd+'/train_'+k+'_superbounds.png',super_bounds)
for k in test_d:
    d=test_d[k]
    mask,bounds,super_bounds=load_mask(d)
    cv2.imwrite(outd+'/test_'+k+'_seg.png',mask)
    cv2.imwrite(outd+'/test_'+k+'_bounds.png',bounds)
    if super_bounds.sum()>0:
        cv2.imwrite(outd+'/test_'+k+'_superbounds.png',super_bounds)

