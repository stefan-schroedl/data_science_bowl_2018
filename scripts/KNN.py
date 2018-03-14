from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import cv2
import faiss   
from transform import random_rotate90_transform1
class KNN():
    def __init__(self,n=5,hist_n=50,patch_size=13,sample=400,gauss_blur=False,similarity=False,normalize=True,super_boundary_threshold=20):
        self.n=n # nearest patches to average
        self.nn=hist_n # nearest images to use as training
        self.super_boundary_threshold=super_boundary_threshold
        self.cutoff=1e5
        self.boundary_cutoff=1 #50
        self.channels=8
        self.boundary_blur=9 #9
        self.patch_size=patch_size
        self.boundary_patch_size=13
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
            boundary = cv2.dilate(boundary, kernel, iterations=2)
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
        self.patches_super_boundary_2 = []
        #generate the patches
        total_patches=0
        for idx in image_idxs:
            if idx<0:
                continue
            img,mask,mask_seg = self.images[idx]
            stacked_img_orig=self.get_stacked(img,mask,mask_seg)
            h,w=stacked_img_orig.shape[:2]
            for y in [0.5,1,2]:
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
            
                    if True or y==1:
                        super_boundary = stacked_img[:,:,6].copy()
                        #kernel = np.ones((5,5), np.uint8)
                        #super_boundary = cv2.dilate(super_boundary, kernel, iterations=2)
                        #print data_patches_super_boundary.shape
                        #sys.exit(1)
                        #data_patches_super_boundary = extract_patches_2d(super_boundary, (self.patch_size,self.patch_size) ,random_state=1000).astype(np.float64)

                        #add regular patches
                        blurred_super_boundary_2=cv2.GaussianBlur(super_boundary,(3,3),0)
                        data_patches_super_boundary_2 = extract_patches_2d(blurred_super_boundary_2, (self.boundary_patch_size,self.boundary_patch_size) ,random_state=1000,max_patches=self.sample).astype(np.float64)
                        data_patches_super_boundary_2 = data_patches_super_boundary_2.reshape(data_patches_super_boundary_2.shape[0], -1)
                        #there are patches that have a max of 0 or 1 , which results just in a solid white or black patch?
                        data_patches_super_boundary_2 = data_patches_super_boundary_2[data_patches_super_boundary_2.max(axis=1)>self.boundary_cutoff]
                        data_patches_super_boundary_2 = (data_patches_super_boundary_2 / data_patches_super_boundary_2.max(axis=1)[:,None])*255
                        self.patches_super_boundary_2.append(data_patches_super_boundary_2)

                        #add the blurred patches
                        blurred_super_boundary=cv2.GaussianBlur(super_boundary,(self.boundary_blur,self.boundary_blur),0,0.1)
                        data_patches_super_boundary = extract_patches_2d(blurred_super_boundary, (self.boundary_patch_size,self.boundary_patch_size) ,random_state=1000,max_patches=self.sample).astype(np.float64)
                        data_patches_super_boundary = data_patches_super_boundary.reshape(data_patches_super_boundary.shape[0], -1)
                        #there are patches that have a max of 0 or 1 , which results just in a solid white or black patch?
                        data_patches_super_boundary = data_patches_super_boundary[data_patches_super_boundary.max(axis=1)>self.boundary_cutoff]
                        data_patches_super_boundary = (data_patches_super_boundary / data_patches_super_boundary.max(axis=1)[:,None])*255
                        self.patches_super_boundary.append(data_patches_super_boundary)
    
        #generate the patch index
        c=self.patch_size*self.patch_size*self.channels
        self.patches_numpy = np.reshape(self.patches, newshape=(total_patches, c))

        c=self.patch_size*self.patch_size*3
        self.patches_3d_numpy = np.reshape(self.patches_3d, newshape=(total_patches, c))

        model = faiss.IndexFlatL2(self.patches_3d_numpy.shape[1])
        model.add(self.patches_3d_numpy.astype(np.float32))

        c=self.boundary_patch_size*self.boundary_patch_size*1
        self.patches_super_boundary_numpy = np.concatenate(self.patches_super_boundary,axis=0)
        super_boundary_model = faiss.IndexFlatL2(self.patches_super_boundary_numpy.shape[1])
        super_boundary_model.add(self.patches_super_boundary_numpy.astype(np.float32))

        c=self.boundary_patch_size*self.boundary_patch_size*1
        self.patches_super_boundary_2_numpy = np.concatenate(self.patches_super_boundary_2,axis=0)
        super_boundary_2_model = faiss.IndexFlatL2(self.patches_super_boundary_2_numpy.shape[1])
        super_boundary_2_model.add(self.patches_super_boundary_2_numpy.astype(np.float32))

        return model,super_boundary_model,super_boundary_2_model


    def reconstruct(self,img,lib_patches,patch_size,model):
            height,width = img.shape
            patches = extract_patches_2d(img, (patch_size,patch_size)).astype(np.float64)
            patches = patches.reshape(patches.shape[0], -1)

            nearest_wds=model.search(patches.astype(np.float32), self.n)
            knn_patches=np.array([])
            for x in xrange(patches.shape[0]):
                idxs=nearest_wds[1][x]
                #idxs=idxs[idxs>=0]
                assert(idxs.min()>=0)
                #use averaging
                new_patch=lib_patches[idxs].mean(axis=0)
		#new_patch=np.multiply(new_patch,gkernel)
                if knn_patches.ndim==1:
                    knn_patches=np.zeros((patches.shape[0],new_patch.shape[0]))
                knn_patches[x]=new_patch
            knn_patches = knn_patches.reshape(knn_patches.shape[0], *(self.boundary_patch_size,self.boundary_patch_size))
            #add in some original :) 
            r= reconstruct_from_patches_2d( knn_patches, (height, width)).astype(np.uint8)
            return r

    def enhance(self,super_boundary_orig,super_boundary_2,super_boundary_model,super_boundary_2_model):
        super_boundary=np.maximum(super_boundary_orig,super_boundary_2)
        #super_boundary=((super_boundary.astype(np.float32)/super_boundary.max())*255).astype(np.uint8)
        #_,super_boundary = cv2.threshold(super_boundary,50,255,cv2.THRESH_BINARY)
        reconstructed=super_boundary.copy()
	gkernel=cv2.getGaussianKernel(ksize=self.boundary_patch_size,sigma=1)
	gkernel=(gkernel*gkernel.T).reshape(-1)
        for xx in range(5):
            r=self.reconstruct(reconstructed,self.patches_super_boundary_numpy,self.boundary_patch_size,super_boundary_model)
            #take out border artifacts
            r[:2,:]=0
            r[-2:,:]=0
            r[:,:2]=0
            r[:,-2:]=0
            reconstructed = np.maximum(r , super_boundary_orig)
        for xx in range(0):
            r=self.reconstruct(reconstructed,self.patches_super_boundary_2_numpy,self.boundary_patch_size,super_boundary_2_model)
            #take out border artifacts
            r[:2,:]=0
            r[-2:,:]=0
            r[:,:2]=0
            r[:,-2:]=0
            reconstructed = r#np.maximum(r , super_boundary_orig)
        #print reconstructed.shape,"WTF"
        #cv2.imshow('reconstructed / orig + 2 / orig',np.concatenate((reconstructed[:,:,None],super_boundary[:,:,None],super_boundary_orig[:,:,None]),axis=1).astype(np.uint8))
        #cv2.waitKey(10000)
        return reconstructed



    def label(self,img,seg):
        img[0,:]=255
        img[-1,:]=255
        img[:,0]=255
        img[:,-1]=255
        img3=np.zeros((img.shape[0],img.shape[1],3))
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        l = img[:,:]*0
        roots=set([])
        tree={}
        for x in xrange(len(contours)):
            nxt,prv,child,parent = hierarchy[0][x]
            if parent>=0:
                if parent not in tree:
                    tree[parent]=set([])
                tree[parent].add(x)
            else:
                roots.add(x)
        idx=2 
        stack=list(roots)
        while len(stack)>0:
            cur=stack[-1]
            stack.pop()
            if cur not in tree:
                tree[cur]=set([])
            a=float(cv2.contourArea(contours[cur]))/(img.shape[0]*img.shape[1])
            color = np.random.randint(0,255,(3)).tolist() 
            cv2.drawContours(img3,[contours[cur]],0,color,2)
            nxt,prv,child,parent = hierarchy[0][cur]
            if True:# child==-1 and a<0.3: #True: # and len(children[cur])<=2:
                cv2.fillPoly(l, [contours[cur]], int(idx))
                idx+=1
            #add all the kids to the stack
            if cur in tree:
                stack+=tree[cur]

        #now lets take out the parts that are mostly white
        #need to watch out for missed nuclei that are totally in the white
        #maybe use the masks as a sanity check?
        #or just start with the mask and then break it out ... that would actually work better?
        #print "HEUY HEFJLDKSJFL"
        #ssadfjkalsdfja
        for x in xrange(idx):
            if img[l==x].mean()>253:
                l[l==x]=0
            elif seg[l==x].mean()<50:
                l[l==x]=1
        cv2.imshow('img/l/l*255/contours',np.concatenate((img,l,l*255,cv2.drawContours(img*0, contours, -1, 255, 3)),axis=1))
        cv2.imshow('img3',img3.astype(np.uint8))
        cv2.waitKey(3)
        return l


    def color_label(self,labels,colors=None):
        out=np.zeros((labels.shape[0],labels.shape[1],3)).astype(np.uint8)
        for x in xrange(labels.max()):
            color = np.random.randint(0,255,(3)).tolist() 
            out[labels==(x+1)]=color
        return out

    def img_normalize(self,img):
        img_norm=((img.astype(np.float32)/img.max())*255).astype(np.uint8)
        return img_norm

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
                patch_model,super_boundary_model, super_boundary_2_model = self.make_index(images_to_use) #,use_all=True)
            else:
                patch_model,super_boundary_model, super_boundary_2_model = self.make_index(nearest_wds[1][0])

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
            #idxs=idxs[idxs>=0]
            assert(idxs.min()>=0)
	    #use averaging
	    new_patch=self.patches_numpy[idxs].mean(axis=0)
	    #new_patch=np.median(self.patches_numpy[idxs],axis=0)
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

        _,reconstructed_super_boundary_thresh = cv2.threshold(reconstructed_super_boundary,self.super_boundary_threshold,255,cv2.THRESH_BINARY)
        l=self.label(reconstructed_super_boundary_thresh,reconstructed_mask).astype(np.int32)-1 #background from 1 -> 0
        l2=cv2.watershed(img,l+1)-1
        l2[l2<0]=0
        l2=l2.astype(np.uint8)
        #cv2.imshow('watershed',l2)
        #cv2.waitKey(3)
        print "TODO ADD THIS ON A PER TILE BASIS? NORMALIZE THERE?"
        enhanced=self.enhance(self.img_normalize(reconstructed_super_boundary),self.img_normalize(reconstructed_super_boundary_2),super_boundary_model,super_boundary_2_model)
        enhance_boundary = cv2.Laplacian(enhanced,cv2.CV_8U,ksize=13)
        cv2.imshow('lap',enhance_boundary)
        _,enhanced_thresh = cv2.threshold(enhanced,self.super_boundary_threshold*4,255,cv2.THRESH_BINARY)
        #print enhanced.shape,enhanced_thresh.shape
        enhanced_label = self.label(enhanced_thresh,reconstructed_mask).astype(np.int32)
        enhanced_label = cv2.watershed(img,enhanced_label)-1
        enhanced_label[enhanced_label<0]=0
        enhanced_label=enhanced_label.astype(np.uint8)
        cv2.imshow('en',enhanced_thresh)
        cv2.imshow('color lab l',np.concatenate((self.color_label(l,reconstructed_mask),self.color_label(l2,reconstructed_mask),self.color_label(enhanced_label,reconstructed_mask),img),axis=1))
        cv2.waitKey(20000)
        return reconstructed_img,reconstructed_mask,reconstructed_boundary,reconstructed_blend,reconstructed_super_boundary,reconstructed_super_boundary_2,l,l2,enhanced_label
        
