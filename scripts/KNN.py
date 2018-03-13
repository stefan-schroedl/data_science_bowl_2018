from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import cv2
import faiss   
from transform import random_rotate90_transform1
class KNN():
    def __init__(self,n=5,patch_size=13,sample=200,gauss_blur=False,similarity=False,normalize=True):
        self.n=5 # nearest patches to average
        self.nn=40 # nearest images to use as training
        self.cutoff=1e5
        self.channels=8
        self.patch_size=patch_size
        self.model =  KNeighborsClassifier(n_neighbors=n,n_jobs=-1,algorithm='kd_tree') 
        self.sample = sample
        self.patches = [] #np.array([]) 
        self.patches_3d = [] #np.array([])
        self.histograms = []
        self.images = []
	self.normalize = normalize
	gkernel=cv2.getGaussianKernel(ksize=patch_size,sigma=1)
	gkernel=gkernel*gkernel.T
	gkernel=gkernel.reshape(patch_size,patch_size,1)
	gkernel=np.concatenate((gkernel,gkernel,gkernel,gkernel,gkernel,gkernel),axis=2)
	gkernel=gkernel.reshape(-1)*self.patch_size*self.patch_size
        self.gkernel=gkernel
        self.similarity=similarity
        self.gauss_blur=gauss_blur
        self.hist_match=True
        self.img_sig=set([])

    def color_check(self,img):
        if img.std(axis=2).mean()<0.01:
            #this is black and white?
            return False
        return True

    def get_hist(self,img):
        color = self.color_check(img)
        h=[[],[],[]]
        for x in range(3):
            h[x]=cv2.calcHist([img],[x],None,[256],[0,256])
            h[x]/=h[x].sum()
            h[x]*=255
        if not color:
            h[1]*=0+255
            h[2]*=0
        return np.concatenate(h)[:,0]

    def get_stacked(self,img,mask,mask_seg):
        super_boundary = mask.copy()[:,:,0]*0
        super_boundary_2 = mask.copy()[:,:,0]*0
        max_components=mask.max()
        kernel = np.ones((5,5), np.uint8)
        for x in xrange(max_components):
            this_one = ((mask==(x+1))*255).astype(np.uint8)[:,:,0]
            boundary = cv2.Laplacian(this_one,cv2.CV_8U,ksize=3)
            super_boundary = np.maximum(super_boundary,boundary)
            boundary = cv2.dilate(boundary, kernel, iterations=1)
            _,boundary_thresh = cv2.threshold(boundary,100,255,cv2.THRESH_BINARY)
            super_boundary_2 += boundary_thresh/255
        #print "X",super_boundary_2.max()
        super_boundary_2 = (super_boundary_2>1)*255
        boundary = cv2.Laplacian(mask_seg,cv2.CV_8U,ksize=3)
        boundary = boundary.reshape(boundary.shape[0],boundary.shape[1],1)
        assert(boundary.max()<=255)
        #0-2 RGB
        #3 SEG
        #4 BOUNDARY
        #5 BLEND
        #6 SUPER BOUNDARY
        #7 SUPER BOUNDARY 2
        stacked_img = np.concatenate((img,mask_seg,boundary,np.maximum(mask_seg/2,boundary),super_boundary[:,:,None],super_boundary_2[:,:,None]),axis=2)
        return stacked_img.astype(np.uint8)

    def prepare_fit(self,img,mask,mask_seg):
        if img.sum() in self.img_sig:
            return 
        self.img_sig.add(img.sum())
	img = (img.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        mask = (mask.numpy()[0].transpose(1,2,0)).astype(np.uint8)
        mask_seg = (mask_seg.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)

        self.images.append((img,mask,mask_seg))
        self.histograms.append(self.get_hist(img))
        self.faiss_model=None

    def fit(self):
        #generate the histogram index
        self.histograms_numpy = np.reshape(self.histograms, newshape=(len(self.histograms), 256*3))
        self.histogram_model = faiss.IndexFlatL2(256*3)
        self.histogram_model.add(self.histograms_numpy.astype(np.float32))


    def resize(self,img,f):
        return cv2.resize(img.astype(np.uint8), (0,0), fx=f, fy=f) 

    def make_index(self,image_idxs,use_all=False):
        print "MAKE INDEX WITH VARIOUS SCALES?? CAN SUPER SCALE? multiply image by 2x, KNN fills in details?"
        self.patches_3d = []
        self.patches = []
        self.patches_super_boundary = []
        #generate the patches
        total_patches=0
        total_patches_super=0
        for idx in image_idxs:
            img,mask,mask_seg = self.images[idx]
            stacked_img_orig=self.get_stacked(img,mask,mask_seg)
            h,w=stacked_img_orig.shape[:2]
            for y in [1,2]:
                stacked_img_scaled=self.resize(stacked_img_orig,y) 
                for x in xrange(4):
                    stacked_img=random_rotate90_transform1(stacked_img_scaled,x)
                    #data_patches = extract_patches_2d(stacked_img, (self.patch_size,self.patch_size) ,random_state=1000,max_patches=self.sample).astype(np.float64)
                    data_patches=None
                    if use_all:
                        data_patches = extract_patches_2d(stacked_img, (self.patch_size,self.patch_size) ,random_state=1000).astype(np.float64)
                    else:
                        data_patches = extract_patches_2d(stacked_img, (self.patch_size,self.patch_size) ,random_state=1000,max_patches=self.sample).astype(np.float64)
                    total_patches+=data_patches.shape[0]
                    self.patches.append(data_patches.reshape(-1))
                    data_patches_3d = data_patches[:,:,:,:3].copy().reshape(data_patches.shape[0], -1).astype(np.float32)
                    if self.normalize:
                        data_patches_3d -= np.mean(data_patches_3d, axis=0)
                        data_patches_3d /= np.std(data_patches_3d, axis=0)
                    self.patches_3d.append(data_patches_3d.reshape(-1))
            
                    if y==1:
                        super_boundary = stacked_img[:,:,6].copy()
                        kernel = np.ones((5,5), np.uint8)
                        #super_boundary = cv2.dilate(super_boundary, kernel, iterations=2)
                        data_patches_super_boundary = extract_patches_2d(super_boundary, (self.patch_size,self.patch_size) ,random_state=1000,max_patches=self.sample).astype(np.float64)
                        #data_patches_super_boundary = extract_patches_2d(super_boundary, (self.patch_size,self.patch_size) ,random_state=1000).astype(np.float64)
                        data_patches_super_boundary = data_patches_super_boundary.reshape(data_patches_super_boundary.shape[0], -1)
                        total_patches_super+=data_patches_super_boundary.shape[0]
                        self.patches_super_boundary.append(data_patches_super_boundary)
    
        #generate the patch index
        c=self.patch_size*self.patch_size*self.channels
        self.patches_numpy = np.reshape(self.patches, newshape=(total_patches, c))

        c=self.patch_size*self.patch_size*3
        self.patches_3d_numpy = np.reshape(self.patches_3d, newshape=(total_patches, c))

        model = faiss.IndexFlatL2(self.patches_3d_numpy.shape[1])
        model.add(self.patches_3d_numpy.astype(np.float32))

        c=self.patch_size*self.patch_size*1
        self.patches_super_boundary_numpy = np.reshape(self.patches_super_boundary, newshape=(total_patches_super, c))

        super_boundary_model = faiss.IndexFlatL2(self.patches_super_boundary_numpy.shape[1])
        super_boundary_model.add(self.patches_super_boundary_numpy.astype(np.float32))

        return model,super_boundary_model


    def enhance(self,super_boundary,super_boundary_model):
        super_boundary=((super_boundary.astype(np.float32)/super_boundary.max())*255).astype(np.uint8)
        #_,super_boundary = cv2.threshold(super_boundary,50,255,cv2.THRESH_BINARY)
        reconstructed=super_boundary.copy()
        height,width = super_boundary.shape
	gkernel=cv2.getGaussianKernel(ksize=self.patch_size,sigma=1)
	gkernel=(gkernel*gkernel.T).reshape(-1)
        for xx in range(4):
            super_boundary_patches = extract_patches_2d(reconstructed, (self.patch_size,self.patch_size)).astype(np.float64)
            super_boundary_patches = super_boundary_patches.reshape(super_boundary_patches.shape[0], -1)

            nearest_wds=super_boundary_model.search(super_boundary_patches.astype(np.float32), self.n)
            knn_patches=np.array([])
            for x in xrange(super_boundary_patches.shape[0]):
                idxs=nearest_wds[1][x]

                #use averaging
                new_patch=self.patches_super_boundary_numpy[idxs].mean(axis=0)
		#new_patch=np.multiply(new_patch,gkernel)
                if knn_patches.ndim==1:
                    knn_patches=np.zeros((super_boundary_patches.shape[0],new_patch.shape[0]))
                knn_patches[x]=new_patch
            knn_patches = knn_patches.reshape(knn_patches.shape[0], *(self.patch_size,self.patch_size))
            reconstructed = reconstruct_from_patches_2d( knn_patches, (height, width)).astype(np.uint8)
        #print reconstructed.shape,"WTF"
        cv2.imshow('w',super_boundary[:,:,None])
        cv2.imshow('r',np.concatenate((reconstructed[:,:,None],super_boundary[:,:,None]),axis=0).astype(np.uint8))
        cv2.waitKey(20000)



    def label(self,img): 
        img[0,:]=255
        img[-1,:]=255
        img[:,0]=255
        img[:,-1]=255
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        l = img[:,:]*0
        stack=[0]
        tree={}
        roots=set([])
        children={}
        for x in xrange(len(contours)):
            nxt,prv,child,parent = hierarchy[0][x]
            if x not in children:
                children[x]=set([])
            if parent>=0:
                if parent not in children:
                    children[parent]=set([])
                children[parent].add(x)
            else:
                roots.add(x)
        idx=1 
        while len(stack)>0:
            cur=stack[-1]
            if cur not in tree:
                tree[cur]=set([])
            a=float(cv2.contourArea(contours[cur]))/(img.shape[0]*img.shape[1])

            nxt,prv,child,parent = hierarchy[0][cur]
            if child==-1 and a<0.3: #True: # and len(children[cur])<=2:
                cv2.fillPoly(l, [contours[cur]], int(idx))
                idx+=1
            if parent>=0:
                if parent not in tree:
                    print "WTF TREE"
                    sys.exit(1)
                tree[parent].add(cur)

            if child>=0 and child not in tree:
                stack.append(child)
                continue

            stack.pop()

            if nxt>=0 and nxt not in tree:
                stack.append(nxt)
                continue
        cv2.imshow('l',np.concatenate((l*255,img,l,cv2.drawContours(img*0, contours, -1, 255, 3)),axis=1))
        cv2.waitKey(3)
        return l



    def predict(self,img):
	img = (img.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)

        patch_model = None
        super_boundary_model = None

        if self.hist_match:
            hist = self.get_hist(img)
            nearest_wds=self.histogram_model.search(hist[None,:].astype(np.float32), self.nn)
            images_to_use=[]
            for x in xrange(len(nearest_wds[1][0])):
	        idx=nearest_wds[1][0][x]
                if idx==-1:
                    continue
                dist=nearest_wds[0][0][x]
                if dist<self.cutoff:
                    images_to_use.append(idx)
            if len(images_to_use)>0:
                patch_model,super_boundary_model = self.make_index(images_to_use) #,use_all=True)
            else:
                patch_model,super_boundary_model = self.make_index(nearest_wds[1][0])

        if patch_model==None:
            if self.faiss_model==None:
                self.faiss_model = self.make_index(xrange(len(self.images)))
            patch_model,super_boundary_model=self.faiss_model

        #cv2.imshow('predict img in',img)
        #cv2.waitKey(1)
        #img=img.numpy()[0]
        height,width,channels = img.shape
        img_patches = extract_patches_2d(img, (self.patch_size,self.patch_size)).astype(np.float64)
        img_patches = img_patches.reshape(img_patches.shape[0], -1).astype(np.float32)
	if self.normalize:
	    img_patches -= np.mean(img_patches, axis=0)
	    img_patches /= np.std(img_patches, axis=0)
        nearest_wds=patch_model.search(img_patches.astype(np.float32), self.n)
	knn_patches=np.array([])
	for x in xrange(img_patches.shape[0]):
	    idxs=nearest_wds[1][x]

	    #use averaging
	    new_patch=self.patches_numpy[idxs].mean(axis=0)
	    #use similarity
	    if self.similarity:
		distances=nearest_wds[0][x]
		similarity=1.0/distances
		similarity/=similarity.sum()
		new_patch=(self.patches_numpy[idxs]*similarity[:,np.newaxis]).sum(axis=0)
	    #gaussian spread
	    if self.gauss_blur:
		new_patch=np.multiply(new_patch,self.gkernel)

	    if knn_patches.ndim==1:
		knn_patches=np.zeros((img_patches.shape[0],new_patch.shape[0]))
	    knn_patches[x]=new_patch
	knn_patches = knn_patches.reshape(knn_patches.shape[0], *(self.patch_size,self.patch_size,self.channels))
	reconstructed = reconstruct_from_patches_2d( knn_patches, (height, width,self.channels))
	reconstructed_img = reconstructed[:,:,:3].astype(np.uint8)
        #cv2.imshow('recon',reconstructed_img)
        #cv2.waitKey(10000)
	reconstructed_mask = reconstructed[:,:,3].astype(np.uint8)
	reconstructed_boundary = reconstructed[:,:,4].astype(np.uint8)
	reconstructed_blend = reconstructed[:,:,5].astype(np.uint8)
	reconstructed_super_boundary = reconstructed[:,:,6].astype(np.uint8)
	reconstructed_super_boundary_2 = reconstructed[:,:,7].astype(np.uint8)

        _,reconstructed_super_boundary_thresh = cv2.threshold(reconstructed_super_boundary,20,255,cv2.THRESH_BINARY)
        l=self.label(reconstructed_super_boundary_thresh).astype(np.int32)
        l2=cv2.watershed(img,l.copy())
        l2[l2==-1]=0
        l2=l2.astype(np.uint8)
        cv2.imshow('watershed',l2)
        cv2.waitKey(3)
        #self.enhance(reconstructed_super_boundary,super_boundary_model)
        return reconstructed_img,reconstructed_mask,reconstructed_boundary,reconstructed_blend,reconstructed_super_boundary,reconstructed_super_boundary_2,l,l2
        
