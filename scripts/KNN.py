from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import imutils
import cv2
import faiss   
from transform import random_rotate90_transform1
class KNN():
    def __init__(self,n=5,hist_n=50,patch_size=13,sample=400,gauss_blur=False,similarity=False,normalize=True,super_boundary_threshold=20,erode=False,match_method='hist'):
        print "SAMPLE IS ",sample, "SHOULD BE ~400?"
        print "TRY TO START WITH SUPER BOUNDARY INSTEAD OF JUST BOUNDARY>????"
        print "http://answers.opencv.org/question/60974/matching-shapes-with-hausdorff-and-shape-context-distance/"
        print "LOG SIMILAR IMAGES!!!"
        print "TRY TO RECONSTRUCT RECURSIVELY?"
        #sys.exit(1)
        self.n=n # nearest patches to average
        self.nn=hist_n # nearest images to use as training
        self.super_boundary_threshold=super_boundary_threshold
        self.cutoff=1e5
        self.boundary_cutoff=1 #50
        self.channels=9
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
        self.match_method=match_method
        self.img_sig=set([])
        self.mask_contours=[]
        self.erode_training_masks=erode
        self.unique_patches=[]
        self.unique_patches_idxs=[]
        self.top_patches=15
        self.similar_patches_cutoff=1

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

    def get_stacked(self,img,mask,mask_seg,mask_eroded):
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

    def prepare_fit(self,img,mask,mask_seg):
        if img.sum() in self.img_sig:
            return 
        self.img_sig.add(img.sum())
	img = (img.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        mask = (mask.numpy()[0].transpose(1,2,0)).astype(np.uint8)
        kernel = np.ones((3,3), np.uint8)
        mask_eroded=cv2.erode(mask, kernel, iterations=1)[:,:,None].astype(np.uint8)

        mask_seg = (mask_seg.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)

        self.images.append((img,mask,mask_seg,mask_eroded))
        #find and store the mask contours
        for x in xrange(mask.max()):
            idx=x+1
            cur_mask = np.zeros_like(mask).astype(np.uint8)
            cur_mask[mask==idx]=255
            _, contours, hierarchy = cv2.findContours(cur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #assert(len(contours)==1) # hmmm
            if len(contours)==0:
                print "OH NO..."
            if False:
                img=img.copy()
                for x in xrange(len(contours)):
                    color = np.random.randint(0,255,(3)).tolist() 
                    cv2.drawContours(img,[contours[x]],0,color,2)
                cv2.imshow('curmask',img)
                cv2.waitKey(5000)
            self.mask_contours.append(contours)

        self.histograms.append(self.get_hist(img))
        self.faiss_model=None

        
        data_patches=self.get_top_patches(img,how_many=-1)
        #add the labels
        self.unique_patches_idxs.append([len(self.unique_patches_idxs)]*data_patches.shape[0])
        #add the patches
        self.unique_patches.append(data_patches)

    def get_top_patches(self,img,how_many=None):
        if how_many==None:
            how_many=self.sample
        #get patches for similarity compare
        data_patches=None
        if how_many>0:
            data_patches = extract_patches_2d(img, (self.patch_size,self.patch_size) ,random_state=1000, max_patches=how_many).astype(np.float64)
        else:
            data_patches = extract_patches_2d(img, (self.patch_size,self.patch_size) ,random_state=1000).astype(np.float64)
        data_patches = data_patches.reshape(data_patches.shape[0],-1)
        stds=data_patches.std(axis=1).argsort()[-self.top_patches:]
        data_patches = data_patches.take(stds,axis=0)
        return data_patches/255

    def similar_by_patches(self,img):
        top_patches = self.get_top_patches(img)
        similar_images_idxs=[]
        print "SEARCH NEARTEST PATCHES"
        hits={}
        hits_full={}
        nearest_wds=self.unique_model.search(top_patches.astype(np.float32), (self.n*max(self.top_patches,4))/4)
	for x in xrange(top_patches.shape[0]):
	    idxs=nearest_wds[1][x]
	    dists=nearest_wds[0][x]
            print dists.min(),dists.mean(),dists.max()
            #idxs=idxs[idxs>=0]
            for y in xrange(len(idxs)):
                idx=idxs[y]
                if idx<0:
                    continue
                dist=dists[y]
                idx=self.unique_patches_idxs_numpy[idx]
                if dist<self.similar_patches_cutoff:
                    if idx not in hits:
                        hits[idx]=0
                    hits[idx]+=1
                if idx not in hits_full:
                    hits_full[idx]=0
                hits_full[idx]+=1

        if len(hits)==0:
            hits=hits_full # back off 'gracefully'
        keys=[]
        for key in hits:
            if len(keys)>=self.nn:
                break
            keys.append((hits[key],key))
        keys.sort(reverse=True)

        similar_img_idxs=[ k[1] for k in keys ]
        return similar_img_idxs

        
    def fit(self):
        #generate the histogram index
        self.histograms_numpy = np.reshape(self.histograms, newshape=(len(self.histograms), 256*3))
        self.histogram_model = faiss.IndexFlatL2(256*3)
        self.histogram_model.add(self.histograms_numpy.astype(np.float32))
        #make unique model 
        self.unique_patches_idxs_numpy = np.concatenate(self.unique_patches_idxs,axis=0)
        self.unique_model = faiss.IndexFlatL2(self.patch_size*self.patch_size*3)
        self.unique_patches_numpy = np.concatenate(self.unique_patches,axis=0) # np.reshape(self.unique_patches, newshape=(len(self.unique_patches_idxs),self.patch_size*self.patch_size*3))
        self.unique_model.add(self.unique_patches_numpy.astype(np.float32))

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
            img,mask,mask_seg,mask_eroded = self.images[idx]
            stacked_img_orig=self.get_stacked(img,mask,mask_seg,mask_eroded)
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
                        boundary = stacked_img[:,:,4].copy()
                        #kernel = np.ones((5,5), np.uint8)
                        #super_boundary = cv2.dilate(super_boundary, kernel, iterations=2)
                        #print data_patches_super_boundary.shape
                        #sys.exit(1)
                        #data_patches_super_boundary = extract_patches_2d(super_boundary, (self.patch_size,self.patch_size) ,random_state=1000).astype(np.float64)

                        #add regular patches
                        blurred_super_boundary_2=cv2.GaussianBlur(super_boundary,(3,3),0)
                        data_patches_super_boundary_2 = extract_patches_2d(blurred_super_boundary_2, (self.boundary_patch_size,self.boundary_patch_size) ,random_state=1000,max_patches=self.sample).astype(np.float64)
                        data_patches_super_boundary_2 = data_patches_super_boundary_2.reshape(data_patches_super_boundary_2.shape[0], -1)
                        ##there are patches that have a max of 0 or 1 , which results just in a solid white or black patch?
                        data_patches_super_boundary_2 = data_patches_super_boundary_2[data_patches_super_boundary_2.max(axis=1)>self.boundary_cutoff]
                        data_patches_super_boundary_2 = (data_patches_super_boundary_2 / data_patches_super_boundary_2.max(axis=1)[:,None])*255
                        self.patches_super_boundary_2.append(data_patches_super_boundary_2)

                        #add the blurred patches
                        #imgs_to_use=[super_boundary,boundary,np.maximum(super_boundary,boundary)]]
                        imgs_to_use=[super_boundary,boundary,np.maximum(super_boundary,boundary)]
                        for img_to_use in imgs_to_use:
                            #add regular patches
                            img_to_use_blurred=cv2.GaussianBlur(img_to_use,(self.boundary_blur,self.boundary_blur),0)
                            mask_patches = extract_patches_2d(img_to_use_blurred, (self.boundary_patch_size,self.boundary_patch_size) ,random_state=1000,max_patches=self.sample).astype(np.float64)
                            mask_patches = mask_patches.reshape(mask_patches.shape[0], -1)
                            #there are patches that have a max of 0 or 1 , which results just in a solid white or black patch?
                            mask_patches = mask_patches[mask_patches.max(axis=1)>self.boundary_cutoff]
                            mask_patches = (mask_patches / mask_patches.max(axis=1)[:,None])*255
                            self.patches_super_boundary.append(mask_patches)
    
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

    def enhance(self,super_boundary_orig,super_boundary_2):
        super_boundary=np.maximum(super_boundary_orig,super_boundary_2)
        #super_boundary=((super_boundary.astype(np.float32)/super_boundary.max())*255).astype(np.uint8)
        #_,super_boundary = cv2.threshold(super_boundary,50,255,cv2.THRESH_BINARY)
        reconstructed=super_boundary.copy()
	gkernel=cv2.getGaussianKernel(ksize=self.boundary_patch_size,sigma=1)
	gkernel=(gkernel*gkernel.T).reshape(-1)
        for xx in range(5):
            r=self.reconstruct(reconstructed,self.patches_super_boundary_numpy,self.boundary_patch_size,self.super_boundary_model)
            #take out border artifacts
            r[:2,:]=0
            r[-2:,:]=0
            r[:,:2]=0
            r[:,-2:]=0
            reconstructed = np.maximum(r , super_boundary_orig)
        for xx in range(0):
            r=self.reconstruct(reconstructed,self.patches_super_boundary_2_numpy,self.boundary_patch_size,self.super_boundary_2_model)
            #take out border artifacts
            r[:2,:]=0
            r[-2:,:]=0
            r[:,:2]=0
            r[:,-2:]=0
            reconstructed = r#np.maximum(r , super_boundary_orig)
        #print reconstructed.shape,"WTF"
        cv2.imshow('reconstructed / orig + 2 / orig',np.concatenate((reconstructed[:,:,None],super_boundary[:,:,None],super_boundary_orig[:,:,None]),axis=1).astype(np.uint8))
        cv2.waitKey(10000)
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
            if x<=1:
                continue
            if img[l==x].mean()>253:
                l[l==x]=0
            elif seg[l==x].mean()<50:
                l[l==x]=1
        #cv2.imshow('img/l/l*255/contours',np.concatenate((img,l,l*255,cv2.drawContours(img*0, contours, -1, 255, 3)),axis=1))
        #cv2.imshow('img3',img3.astype(np.uint8))
        #cv2.waitKey(3)
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

    def by_cluster(self,img,reconstructed,similar_contours):
        assert(reconstructed.shape[2]==self.channels)
        img=img.copy()
        height,width,_=img.shape
	reconstructed_img = reconstructed[:,:,:3].astype(np.uint8)
	reconstructed_mask = reconstructed[:,:,3].astype(np.uint8)
	reconstructed_boundary = reconstructed[:,:,4].astype(np.uint8)
	reconstructed_blend = reconstructed[:,:,5].astype(np.uint8)
	reconstructed_super_boundary = reconstructed[:,:,6].astype(np.uint8)
	reconstructed_super_boundary_2 = reconstructed[:,:,7].astype(np.uint8)

        cv2.imshow('mask',reconstructed_mask)
        _,reconstructed_mask_thresh = cv2.threshold(reconstructed_mask,50,255,cv2.THRESH_BINARY)

        _, contours, hierarchy = cv2.findContours(reconstructed_mask_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        marking_idx=2
        labels=np.ones_like(reconstructed_mask).astype(np.int32)
        enhanced=self.enhance(self.img_normalize(reconstructed_super_boundary),self.img_normalize(reconstructed_super_boundary_2))
        _,enhanced_thresh=cv2.threshold(enhanced,80,255,cv2.THRESH_BINARY)
        enhanced_thresh=-enhanced_thresh+255
        #enhanced_thresh=np.maximum(enhanced_thresh,reconstructed_mask_thresh) # make sure to not remove the original bounds?
        for x in xrange(len(contours)):
            color = np.random.randint(0,255,(3)).tolist() 
            cv2.drawContours(img,[contours[x]],0,color,2)
            ix,iy,iw,ih = cv2.boundingRect(contours[x])
            edge=False
            if ix<=2 or iy<=2 or (ix+iw+2)>=width or (iy+ih+2)>=height:
                edge=True
            min_dist=None
            min_idx=-1
            #for idx in xrange(len(similar_contours)):
            #    d=cv2.matchShapes(contours[x],similar_contours[idx],1,0.0)
            #    if min_dist==None:
            #        min_dist=d
            #        min_idx=idx
            #    elif min_dist>d:
            #        min_dist=d
            #        min_idx=idx
            if True: #not edge and min_dist>0.08:
                roi_mask = np.zeros_like(reconstructed_mask)[:,:,None]
                cv2.fillPoly(roi_mask, [contours[x]], 1)
                roi = np.multiply(reconstructed,roi_mask)
                roi_img = roi[:,:,:3].astype(np.uint8)
                roi_mask = roi[:,:,3].astype(np.uint8)
                roi_boundary = roi[:,:,4].astype(np.uint8)
                roi_blend = roi[:,:,5].astype(np.uint8)
                roi_super_boundary = roi[:,:,6].astype(np.uint8)
                roi_super_boundary_2 = roi[:,:,7].astype(np.uint8)
                roi_enhanced_thresh=np.multiply(roi_mask,enhanced_thresh)
                _, roi_contours, roi_hierarchy = cv2.findContours(roi_enhanced_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(roi_contours)>0:
                    cv2.fillPoly(labels, [contours[x]], 0)
                    for roi_x in xrange(len(roi_contours)):
                        color = np.random.randint(0,255,(3)).tolist() 
                        cv2.drawContours(roi_img,[roi_contours[roi_x]],0,color,2)
                        cv2.fillPoly(labels, [roi_contours[roi_x]], int(marking_idx))
                        marking_idx+=1
                else:
                    color = np.random.randint(0,255,(3)).tolist() 
                    cv2.drawContours(roi_img,[contours[x]],0,color,2)
                    cv2.fillPoly(labels, [contours[x]], int(marking_idx))
                    marking_idx+=1
                #cv2.imshow('area',roi_img)
                #cv2.imshow('area2',roi_enhanced_thresh)
                #cv2.waitKey(20000)
                #run local watershed?
            else:
                cv2.fillPoly(labels, [contours[x]], int(marking_idx))
                marking_idx+=1
        #water=cv2.watershed(img,labels)-1
        water=cv2.watershed(cv2.cvtColor(reconstructed_mask, cv2.COLOR_GRAY2BGR),labels)-1
        water[water<0]=0 # set bg and unknown to 0
        cv2.imshow('labeled', self.color_label(labels))
        cv2.imshow('water', self.color_label(water))
        cv2.imshow('img',img)
        cv2.waitKey(20)
        return water

        #get the clusters from the mask


    def similar_by_hist(self,img):
        hist = self.get_hist(img)
        similar_images_idxs=[]
        print "SEARCH NEARTEST HIST"
        nearest_wds=self.histogram_model.search(hist[None,:].astype(np.float32), self.nn)
        for x in xrange(len(nearest_wds[1][0])):
            idx=nearest_wds[1][0][x]
            if idx==-1:
                continue
            dist=nearest_wds[0][0][x]
            if dist<self.cutoff:
                similar_images_idxs.append(idx)
        if len(similar_images_idxs)==0:
            similar_images_idxs=nearest_wds[1][0]
        return similar_images_idxs



    def predict(self,img):
        print "CONVEX HULL CHECK"
        print "SHRINK ALL BOUNDARIES, ESPECIALLY THOSE WITH OTHER MASKS NEARBY! OVERLAP -> 0 PREDICTION"
        print "TODO: subtract super boundaries from regular"
        print "START PREDICT"
	img = (img.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)

        similar_contours = []

        similar_hist_images=[]
        similar_hist_images_idxs=self.similar_by_hist(img)
        for idx in similar_hist_images_idxs:
            img2,_,_,_ = self.images[idx]
            similar_hist_images.append(cv2.resize(img2, (256, 256)))
            #similar_hist_images.append(imutils.resize(img2, height=256,width=256))
            #cv2.imshow("y",img2)
            #cv2.waitKey(2000)

        similar_patch_images=[]
        similar_patch_images_idxs=self.similar_by_patches(img)
        for idx in similar_patch_images_idxs:
            img2,_,_,_ = self.images[idx]
            similar_patch_images.append(cv2.resize(img2, (256, 256)))
            #similar_patch_images.append(imutils.resize(img2, height=256).astype(np.uint8))
            #cv2.imshow("x",similar_patch_images[-1])
            #cv2.waitKey(2000)

        patch_model = None
        super_boundary_model = None
        if self.match_method=='hist':
            patch_model,self.super_boundary_model, self.super_boundary_2_model = self.make_index(similar_hist_images_idxs) #,use_all=True)
        elif self.match_method=='patch':
            patch_model,self.super_boundary_model, self.super_boundary_2_model = self.make_index(similar_patch_images_idxs) #,use_all=True)
        else:
            print "INVALID MODEL TYPE"

        if patch_model==None:
            if self.faiss_model==None:
                self.faiss_model = self.make_index(xrange(len(self.images)))
            patch_model,super_boundary_model=self.faiss_model
        print "START IMG PATCH GEN"

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
        print "START IMG PATCH GEN - STEP 2"
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
        print "START RECONSTRUCT"
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
	reconstructed_mask_eroded = reconstructed[:,:,7].astype(np.uint8)

        #cv2.imshow('mask vs eroded',np.concatenate((reconstructed_mask,reconstructed_mask_eroded),axis=1))
        #cv2.waitKey(10)

        clustered=self.by_cluster(img,reconstructed,similar_contours)

        _,reconstructed_super_boundary_thresh = cv2.threshold(reconstructed_super_boundary,self.super_boundary_threshold,255,cv2.THRESH_BINARY)
        labeled=self.label(reconstructed_super_boundary_thresh,reconstructed_mask).astype(np.int32)-1 #background from 1 -> 0
        labeled2=cv2.watershed(img,labeled+1)-1
        labeled2[labeled2<0]=0
        labeled2=labeled2.astype(np.uint8)
        #lets try to do something one cluster at a time?
        #cv2.imshow('watershed',labeled2)
        #cv2.waitKey(3)
        print "SHRINK TRAINING MASKS?????!?!??!?!?!?!?"
        print "TODO ADD THIS ON A PER TILE BASIS? NORMALIZE THERE?"
        enhanced=self.enhance(self.img_normalize(reconstructed_super_boundary),self.img_normalize(reconstructed_super_boundary_2))
        #enhance_boundary = cv2.Laplacian(enhanced,cv2.CV_8U,ksize=13)
        #cv2.imshow('lap',enhance_boundary)
        _,enhanced_thresh = cv2.threshold(enhanced,self.super_boundary_threshold*4,255,cv2.THRESH_BINARY)
        #print enhanced.shape,enhanced_thresh.shape
        enhanced_label = self.label(enhanced_thresh,reconstructed_mask).astype(np.int32)
        enhanced_label = cv2.watershed(img,enhanced_label)-1
        enhanced_label[enhanced_label<0]=0
        enhanced_label=enhanced_label.astype(np.uint8)
        #cv2.imshow('en',enhanced_thresh)
        cv2.imshow('color lab l',np.concatenate((self.color_label(labeled,reconstructed_mask),self.color_label(labeled2,reconstructed_mask),self.color_label(enhanced_label,reconstructed_mask),img),axis=1))
        cv2.waitKey(2)
        d={}
        d['img']=reconstructed_img
        d['seg']=reconstructed_mask
        d['boundary']=reconstructed_boundary
        d['blend']=reconstructed_blend
        d['super_boundary']=reconstructed_super_boundary
        d['super_boundary_2']=reconstructed_super_boundary_2
        d['labeled']=labeled
        d['labeled2']=labeled2
        d['enhanced_label']=enhanced_label
        d['clustered']=clustered
        d['similar_patch_images']=similar_patch_images
        d['similar_hist_images']=similar_hist_images
        return d 
    
    
