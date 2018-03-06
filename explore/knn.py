import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
import sys

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

def load_mask(d):
    im=np.zeros((1,1))
    for im_fn in d['masks']:
        im_cur = cv2.imread(im_fn)
        if im.sum()==0:
            im=np.zeros(im_cur.shape[:2])
        im=np.maximum(im,im_cur[:,:,0])
    im=im.reshape(im.shape[0],im.shape[1],1)
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
    if len(all_patches)>100: # stop reading after the 100th image, and test on the 101st image
    	test_img=d
        break
    im = np.concatenate((im,mask),axis=2)
    read_imgs.append(d['img'])
    data_patches = extract_patches_2d(im, patch_size,random_state=1000,max_patches=1000).astype(np.float64) # only take 1000 patches from each image
    #lets store the 4D data , RGB+Mask and also the RGB seperately , RGB is used for KNN lookup
    data = data_patches.copy().reshape(data_patches.shape[0], -1)
    all_patches.append(data)
    data3 = data_patches[:,:,:,:3].copy().reshape(data_patches.shape[0], -1)
    if normalize:
	data3 -= np.mean(data3, axis=0)
	data3 /= np.std(data3, axis=0)
    all_patches3.append(data3)

data=np.concatenate(all_patches,0)
data3=np.concatenate(all_patches3,0)

#knn
print "Fitting KNN"
neigh = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
neigh.fit(data3, np.zeros((len(data))))

print "Loading test image"
im=cv2.imread(test_img['img'])
_,im_mask=load_mask(test_img)
cv2.imshow('im1',im.astype(np.uint8))
cv2.waitKey(10)
height,width,channels = im.shape

print "Extracting test patches"
im_patches = extract_patches_2d(im, patch_size).astype(np.float64)
im_patches = im_patches.reshape(im_patches.shape[0], -1)
if normalize:
    im_patches -= np.mean(im_patches, axis=0)
    im_patches /= np.std(im_patches, axis=0)
#intercept = np.mean(im_patches, axis=0)
#im_patches -= intercept
#print intercept

print "Running KNN lookup"
#code = dico.transform(im_data)
#patches = np.dot(code, V)
#PCAd=dict_PCA.transform(im_data)
#patches=dict_PCA.inverse_transform(PCAd)

im_patches=im_patches.reshape(im_patches.shape[0],-1)
nearest_wds=neigh.kneighbors(im_patches, return_distance=True)

knn_patches=np.array([])
for x in xrange(im_patches.shape[0]):
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
knn_patches = knn_patches.reshape(knn_patches.shape[0], *(pz,pz,4)).astype(np.uint8)
reconstructed = reconstruct_from_patches_2d( knn_patches, (height, width,4)).astype(np.uint8)
reconstructed_img = reconstructed[:,:,:3]
reconstructed_mask = reconstructed[:,:,3]
cv2.imshow('im vs reconstructed',np.concatenate((im,reconstructed_img),axis=1))
cv2.imshow('mask vs predicted',np.concatenate((im_mask[:,:,0],reconstructed_mask.astype(np.uint8)),axis=1))
cv2.waitKey(10000)
