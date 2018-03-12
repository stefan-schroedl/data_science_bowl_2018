from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import cv2
import faiss   

class KNN():
    def __init__(self,n=5,patch_size=13,sample=100,gauss_blur=False,similarity=False,normalize=True):
        self.n=5
        self.patch_size=patch_size
        self.model =  KNeighborsClassifier(n_neighbors=n,n_jobs=-1,algorithm='kd_tree') 
        self.sample = sample
        self.patches = np.array([]) 
        self.patches_3d = np.array([])
	self.normalize = normalize
	gkernel=cv2.getGaussianKernel(ksize=patch_size,sigma=1)
	gkernel=gkernel*gkernel.T
	gkernel=gkernel.reshape(patch_size,patch_size,1)
	gkernel=np.concatenate((gkernel,gkernel,gkernel,gkernel,gkernel,gkernel),axis=2)
	gkernel=gkernel.reshape(-1)*self.patch_size*self.patch_size
        self.gkernel=gkernel
        self.similarity=similarity
        self.gauss_blur=gauss_blur
        self.faiss=True

    def prepare_fit(self,img,mask,mask_seg):
	img = (img.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        mask = (mask.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        mask_seg = (mask_seg.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        #cv2.imshow('img',img)
        #cv2.imshow('mask',mask)
        #cv2.imshow('seg',mask_seg)
        #cv2.waitKey(5000)
        #img=img.numpy()[0]
        #mask_seg=(mask_seg.numpy()[0].copy()*255).astype(np.uint8)
        #mask=mask.numpy()[0]
        #mask=np.minimum(mask,1)
        #mask*=255
        #boundary = cv2.Laplacian(mask_seg,cv2.CV_64F,ksize=3)
        boundary = cv2.Laplacian(mask_seg,cv2.CV_8U,ksize=3)
        boundary = boundary.reshape(boundary.shape[0],boundary.shape[1],1)
        assert(boundary.max()<=255)
        stacked_img = np.concatenate((img,mask_seg,boundary,np.maximum(mask_seg/2,boundary)),axis=2)
        data_patches = extract_patches_2d(stacked_img, (self.patch_size,self.patch_size) ,random_state=1000,max_patches=self.sample).astype(np.float64)
        if self.patches.ndim==1:
            self.patches = data_patches.copy().reshape(data_patches.shape[0], -1)
        else:
            self.patches = np.concatenate( (self.patches, data_patches.copy().reshape(data_patches.shape[0], -1)),axis=0 )
	data_patches_3d = data_patches[:,:,:,:3].copy().reshape(data_patches.shape[0], -1)
	if self.normalize:
	    data_patches_3d -= np.mean(data_patches_3d, axis=0)
	    data_patches_3d /= np.std(data_patches_3d, axis=0)
        if self.patches_3d.ndim==1:
            self.patches_3d=data_patches_3d
        else:
            self.patches_3d=np.concatenate( (self.patches_3d, data_patches_3d) , axis=0)

    def fit(self):
        if self.faiss:
            self.faiss_model = faiss.IndexFlatL2(self.patches_3d.shape[1])
            self.faiss_model.add(self.patches_3d.astype(np.float32))
        else:
            self.model.fit(self.patches_3d, np.zeros((len(self.patches_3d))))

    def predict(self,img):
	img = (img.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        cv2.imshow('predict img in',img)
        cv2.waitKey(1)
        #img=img.numpy()[0]
        height,width,channels = img.shape
        img_patches = extract_patches_2d(img, (self.patch_size,self.patch_size)).astype(np.float64)
        img_patches = img_patches.reshape(img_patches.shape[0], -1)
        mean = np.mean(img_patches, axis=0)
        std = np.std(img_patches, axis=0)
	if self.normalize:
	    img_patches -= mean
	    img_patches /= std
        nearest_wds=None
        if self.faiss:
            nearest_wds=self.faiss_model.search(img_patches.astype(np.float32), self.n)
        else:
            nearest_wds=self.model.kneighbors(img_patches, return_distance=True)
	knn_patches=np.array([])
	for x in xrange(img_patches.shape[0]):
	    idxs=nearest_wds[1][x]

	    #use averaging
	    new_patch=self.patches[idxs].mean(axis=0)
	    #use similarity
	    if self.similarity:
		distances=nearest_wds[0][x]
		similarity=1.0/distances
		similarity/=similarity.sum()
		new_patch=(self.patches[idxs]*similarity[:,np.newaxis]).sum(axis=0)
	    #gaussian spread
	    if self.gauss_blur:
		new_patch=np.multiply(new_patch,self.gkernel)

	    if knn_patches.ndim==1:
		knn_patches=np.zeros((img_patches.shape[0],new_patch.shape[0]))
	    knn_patches[x]=new_patch
	knn_patches = knn_patches.reshape(knn_patches.shape[0], *(self.patch_size,self.patch_size,6))
	reconstructed = reconstruct_from_patches_2d( knn_patches, (height, width,6))
	reconstructed_img = reconstructed[:,:,:3].astype(np.uint8)
        #cv2.imshow('recon',reconstructed_img)
        #cv2.waitKey(10000)
	reconstructed_mask = reconstructed[:,:,3].astype(np.uint8)
	reconstructed_boundary = reconstructed[:,:,4].astype(np.uint8)
	reconstructed_blend = reconstructed[:,:,5].astype(np.uint8)
        return reconstructed_img,reconstructed_mask,reconstructed_boundary,reconstructed_blend
        
        
