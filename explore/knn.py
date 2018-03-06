from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
import sys
from random import shuffle

normalize=False
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

def read_list(fn):
    f=open(fn,'r')
    fns=[ line.strip() for line in  f]
    f.close()
    shuffle(fns)
    return fns

def load_mask(d):
    im=np.zeros((1,1))
    for im_fn in d['masks']:
        im_cur = cv2.imread(im_fn)
        if im.sum()==0:
            im=np.zeros(im_cur.shape[:2])
        im=np.maximum(im,im_cur[:,:,0])
    im=im.reshape(im.shape[0],im.shape[1],1)
    #cv2.imshow('m',im)
    #cv2.imshow('s',cv2.Sobel(im,cv2.CV_8U,1,0,ksize=5))
    #cv2.imshow('l',cv2.Laplacian(im,cv2.CV_64F))
    l=cv2.Laplacian(im,cv2.CV_64F)
    l=l.reshape(l.shape[0],l.shape[1],1)
    return im,l

read_imgs=[]
test_img=None
all_patches=[]
all_patches3=[]
pz=9
patch_size=(pz,pz)
for k in train_d:
    d=train_d[k]
    _,mask=load_mask(d)
    im = cv2.imread(d['img'])
    #cv2.imshow('im',im)
    #cv2.imshow('mask',mask)
    #cv2.waitKey(10000)
    if len(all_patches)>100:
    	test_img=d
        break
    im = np.concatenate((im,mask),axis=2)
    read_imgs.append(d['img'])
    data_patches = extract_patches_2d(im, patch_size,random_state=1000,max_patches=1000).astype(np.float64)
    data = data_patches.copy().reshape(data_patches.shape[0], -1)
    #data -= np.mean(data, axis=0)
    all_patches.append(data)
    data3 = data_patches[:,:,:,:3].copy().reshape(data_patches.shape[0], -1)
    if normalize:
	data3 -= np.mean(data3, axis=0)
	data3 /= np.std(data3, axis=0)
    all_patches3.append(data3)

print "Fitting dictionary"
data=np.concatenate(all_patches,0)
data3=np.concatenate(all_patches3,0)


#dictionary
#dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
#V = dico.fit(data).components_

#pca
#dict_PCA = PCA(n_components=99)
#dict_PCA.fit(data)
#
#knn
neigh = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
neigh.fit(data3, np.zeros((len(data))))
print "D3",data3.shape

im=cv2.imread(test_img['img'])
_,im_mask=load_mask(test_img)
cv2.imshow('im1',im.astype(np.uint8))
cv2.waitKey(10)
height,width,channels = im.shape

print "Extract test patches"
im_patches = extract_patches_2d(im, patch_size).astype(np.float64)
im_patches = im_patches.reshape(im_patches.shape[0], -1)
if normalize:
    im_patches -= np.mean(im_patches, axis=0)
    im_patches /= np.std(im_patches, axis=0)
#intercept = np.mean(im_patches, axis=0)
#im_patches -= intercept
#print intercept

print "Running dico transform"
#code = dico.transform(im_data)
#patches = np.dot(code, V)
#PCAd=dict_PCA.transform(im_data)
#patches=dict_PCA.inverse_transform(PCAd)

im_patches=im_patches.reshape(im_patches.shape[0],-1)
nearest_wds=neigh.kneighbors(im_patches, return_distance=True)

knn_patches=np.array([])
for x in xrange(im_patches.shape[0]):
    #find the closest patches and average them
    #nearest=neigh.kneighbors(im_patches[x].reshape(1,-1), return_distance=False)
    idxs=nearest_wds[1][x]

    #use averaging
    new_patch=data[idxs].mean(axis=0)
    
    #use similarity
    #distances=nearest_wds[0][x]
    #similarity=1.0/distances
    #similarity/=similarity.sum()
    #new_patch=(data[idxs]*similarity[:,np.newaxis]).sum(axis=0)

    if knn_patches.ndim==1:
        knn_patches=np.zeros((im_patches.shape[0],new_patch.shape[0]))
    knn_patches[x]=new_patch

print "Reconstructing"
#patches += intercept
#knn_patches += intercept

#use knn

knn_patches = knn_patches.reshape(knn_patches.shape[0], *(pz,pz,4)).astype(np.uint8)
reconstructed = reconstruct_from_patches_2d( knn_patches, (height, width,4)).astype(np.uint8)
reconstructed_img = reconstructed[:,:,:3]
reconstructed_mask = reconstructed[:,:,3]
#cv2.imshow('mask1',thresh1)
cv2.imshow('im vs reconstructed',np.concatenate((im,reconstructed_img),axis=1))
cv2.imshow('mask vs predicted',np.concatenate((im_mask[:,:,0],reconstructed_mask.astype(np.uint8)),axis=1))
#cv2.imshow('mask3',thresh3)
cv2.waitKey(10000)



#print('done in %.2fs.' % (time() - t0))
#
## #############################################################################
## Learn the dictionary from reference patches
#
#print('Learning the dictionary...')
#t0 = time()
#dt = time() - t0
#print('done in %.2fs.' % dt)
#
