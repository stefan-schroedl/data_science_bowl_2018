from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import imutils
import cv2
import faiss   
from transform import random_rotate90_transform1
from GUESS import *
import cPickle as pickle

class KNN():
    def eval():
        return 
    def __init__(self,n=5,hist_n=10,patch_size=13,sample=400,gauss_blur=False,similarity=False,normalize=True,super_boundary_threshold=20,erode=False,match_method='hist'):
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
        self.channels=8
        self.boundary_blur=9 #9
        self.patch_size=patch_size
        self.boundary_patch_size=13
        #self.model =  KNeighborsClassifier(n_neighbors=n,n_jobs=-1,algorithm='kd_tree') 
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
        self.erode_training_masks=erode
        #self.unique_patches=[]
        #self.unique_patches_idxs=[]
        #self.top_patches=15
        self.similar_patches_cutoff=1

    def parameters(self):
        return []

    def load(fn):
        m=pickle.load( open(fn, "rb" ) )
        m.fit()
        return m

    def save(self,filename):
        k=KNN(self.n,self.nn,self.patch_size,self.sample,self.gauss_blur,self.similarity,self.normalize,self.super_boundary_threshold,self.erode_training_masks,self.match_method)
        k.histograms=self.histograms
        k.histogram_numpy=self.histograms_numpy
        #k.unique_patches=self.unique_patches
        #k.unique_patches_numpy=self.unique_patches_numpy
        output=open(filename, 'wb')
        pickle.dump(k, output, pickle.HIGHEST_PROTOCOL)

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

            #super boundary 2 by dilation of border
            #kernel = np.ones((5,5), np.uint8)
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
        stacked_img = np.concatenate((img,mask_seg,boundary,np.maximum(mask_seg/2,boundary),super_boundary[:,:,None],super_boundary_2[:,:,None]),axis=2)
        return stacked_img.astype(np.uint8)

    def prepare_fit(self,img,mask,mask_seg):
        s=img.sum()
        if s in self.img_sig:
            return 
        print "ADDING IMAGE TO DB!"
        self.img_sig.add(s)
	img = (img.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        mask = (mask.numpy()[0].transpose(1,2,0)).astype(np.uint8)
        #kernel = np.ones((3,3), np.uint8)
        #mask_eroded=cv2.erode(mask, kernel, iterations=1)[:,:,None].astype(np.uint8)

        mask_seg = (mask_seg.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)

        self.images.append((img,mask,mask_seg))
        self.histograms.append(self.get_hist(img))
        self.faiss_model=None

        
        #data_patches=self.get_top_patches(img,how_many=-1)
        #add the labels
        #self.unique_patches_idxs.append([len(self.unique_patches_idxs)]*data_patches.shape[0])
        #add the patches
        #self.unique_patches.append(data_patches)

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
            #print dists.min(),dists.mean(),dists.max()
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
        #self.unique_patches_idxs_numpy = np.concatenate(self.unique_patches_idxs,axis=0)
        #self.unique_model = faiss.IndexFlatL2(self.patch_size*self.patch_size*3)
        #self.unique_patches_numpy = np.concatenate(self.unique_patches,axis=0) # np.reshape(self.unique_patches, newshape=(len(self.unique_patches_idxs),self.patch_size*self.patch_size*3))
        #self.unique_model.add(self.unique_patches_numpy.astype(np.float32))

    def resize(self,img,f):
        return cv2.resize(img.astype(np.uint8), (0,0), fx=f, fy=f) 

    def make_index(self,image_idxs,use_all=False):
        print "MAKE INDEX WITH VARIOUS SCALES?? CAN SUPER SCALE? multiply image by 2x, KNN fills in details?"
        self.patches_3d = []
        self.patches = []
        self.patches_super_boundary = []
        #self.patches_super_boundary_2 = []
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
                        boundary = stacked_img[:,:,4].copy()
                        #kernel = np.ones((5,5), np.uint8)
                        #super_boundary = cv2.dilate(super_boundary, kernel, iterations=2)
                        #print data_patches_super_boundary.shape
                        #sys.exit(1)
                        #data_patches_super_boundary = extract_patches_2d(super_boundary, (self.patch_size,self.patch_size) ,random_state=1000).astype(np.float64)

                        #add regular patches
                        #blurred_super_boundary_2=cv2.GaussianBlur(super_boundary,(3,3),0)
                        #data_patches_super_boundary_2 = extract_patches_2d(blurred_super_boundary_2, (self.boundary_patch_size,self.boundary_patch_size) ,random_state=1000,max_patches=self.sample).astype(np.float64)
                        #data_patches_super_boundary_2 = data_patches_super_boundary_2.reshape(data_patches_super_boundary_2.shape[0], -1)
                        ##there are patches that have a max of 0 or 1 , which results just in a solid white or black patch?
                        #data_patches_super_boundary_2 = data_patches_super_boundary_2[data_patches_super_boundary_2.max(axis=1)>self.boundary_cutoff]
                        #data_patches_super_boundary_2 = (data_patches_super_boundary_2 / data_patches_super_boundary_2.max(axis=1)[:,None])*255
                        #self.patches_super_boundary_2.append(data_patches_super_boundary_2)

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

        #c=self.boundary_patch_size*self.boundary_patch_size*1
        #self.patches_super_boundary_2_numpy = np.concatenate(self.patches_super_boundary_2,axis=0)
        super_boundary_2_model = None #faiss.IndexFlatL2(self.patches_super_boundary_2_numpy.shape[1])
        #super_boundary_2_model.add(self.patches_super_boundary_2_numpy.astype(np.float32))

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
            print "ENHANCE",xx
            r=self.reconstruct(reconstructed,self.patches_super_boundary_numpy,self.boundary_patch_size,self.super_boundary_model)
            #take out border artifacts
            r[:2,:]=0
            r[-2:,:]=0
            r[:,:2]=0
            r[:,-2:]=0
            reconstructed = np.maximum(r , super_boundary_orig)
        #for xx in range(0):
        #    r=self.reconstruct(reconstructed,self.patches_super_boundary_2_numpy,self.boundary_patch_size,self.super_boundary_2_model)
        #    #take out border artifacts
        #    r[:2,:]=0
        #    r[-2:,:]=0
        #    r[:,:2]=0
        #    r[:,-2:]=0
        #    reconstructed = r#np.maximum(r , super_boundary_orig)
        #cv2.imshow('reconstructed / orig + 2 / orig',np.concatenate((reconstructed[:,:,None],super_boundary[:,:,None],super_boundary_orig[:,:,None]),axis=1).astype(np.uint8))
        #cv2.waitKey(4000)
        return reconstructed

    def label(self,img,seg):
        #img[0,:]=255
        #img[-1,:]=255
        #img[:,0]=255
        #img[:,-1]=255
        cv2.imshow("WTF WTF",img)
        cv2.waitKey(100)
        img[0,:]=seg[0,:]
        img[-1,:]=seg[-1,:]
        img[:,0]=seg[:,0]
        img[:,-1]=seg[:,-1]
        img3=np.zeros((img.shape[0],img.shape[1],3))
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        l = img[:,:]*0
        roots=set([])
        tree={}
        for x in xrange(len(contours)):
	    #rect = cv2.minAreaRect(contours[x])
            area = cv2.contourArea(contours[x])
            print "AREA",area
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
                #if img[l==idx].mean()>253:
                #    l[l==idx]=0
                #elif seg[l==x].mean()<100:
                #    l[l==idx]=1
                #else:
                if l[l==idx].sum()>0:
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
        #for x in xrange(idx):
        #    if x<=1:
        #        continue
        #cv2.imshow('img/l/l*255/contours',np.concatenate((img,l,l*255,cv2.drawContours(img*0, contours, -1, 255, 3)),axis=1))
        #cv2.imshow('img3',img3.astype(np.uint8))
        #cv2.waitKey(3)
        return l


    def color_label(self,labels,colors=None):
        out=np.zeros((labels.shape[0],labels.shape[1],3)).astype(np.uint8)
        labels=np.squeeze(labels)
        for x in xrange(labels.max()):
            color = np.random.randint(0,255,(3)).tolist() 
	    if labels[labels==(x+1)].sum()>0:
            	out[labels==(x+1)]=color
        return out

    def img_normalize(self,img):
        img_norm=((img.astype(np.float32)/img.max())*255).astype(np.uint8)
        return img_norm

    #https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python
    def crop_minAreaRect(self,img, rect):
        # rotate img
        angle = rect[2]
        rows,cols = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        img_rot = cv2.warpAffine(img,M,(cols,rows))

        # rotate bounding box
        rect0 = (rect[0], rect[1], 0.0)
        box = cv2.boxPoints(rect)
        pts = np.int0(cv2.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0

        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1],
                           pts[1][0]:pts[2][0]]

        return img_crop

    def crop(self,img,rect,bound=0):
        x,y,w,h = rect
        x_start=x-bound/2
        x_end=x+w+bound/2
        y_start=y-bound/2
        y_end=y+h+bound/2
        if x_start<0:
            x_start=0
        if y_start<0:
            y_start=0
        if x_end>img.shape[1]:
            x_end=img.shape[1]-1
        if y_end>img.shape[0]:
            y_end=img.shape[0]-1
        
        # crop
        #return img[y:y+h,x:x+w,:]
        return img[y_start:y_end,x_start:x_end,:]


    def blur_bounds(self,boundary):
        k=9
	boundary=np.squeeze(boundary).copy()
	print "NBOUND",boundary.shape
        blurred_boundary=np.maximum(boundary,cv2.GaussianBlur(boundary,(k,k),0))
        for x in xrange(10):
            blurred_boundary=np.maximum(boundary,cv2.GaussianBlur(blurred_boundary,(k,k),0))
        blurred_boundary=blurred_boundary.astype(np.float)
        blurred_boundary/=blurred_boundary.max()
        blurred_boundary*=255
        burred_boundary=blurred_boundary.astype(np.uint8)
	return burred_boundary

    def by_cluster(self,img,reconstructed):
        assert(reconstructed.shape[2]==self.channels)
        img=img.copy()
        height,width,_=img.shape
	reconstructed_img = reconstructed[:,:,:3].astype(np.uint8)
	reconstructed_mask = reconstructed[:,:,3].astype(np.uint8)
	reconstructed_super_boundary = reconstructed[:,:,6].astype(np.uint8)
	reconstructed_super_boundary_2 = reconstructed[:,:,7].astype(np.uint8)

        _,reconstructed_mask_thresh = cv2.threshold(reconstructed_mask,50,255,cv2.THRESH_BINARY)

        _, contours, hierarchy = cv2.findContours(reconstructed_mask_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        marking_idx=2
        labels=np.ones_like(reconstructed_mask).astype(np.int32)
        enhanced=self.enhance(self.img_normalize(reconstructed_super_boundary),self.img_normalize(reconstructed_super_boundary_2))
        _,enhanced_thresh=cv2.threshold(enhanced,80,255,cv2.THRESH_BINARY)
        enhanced_thresh=-enhanced_thresh+255
        for x in xrange(len(contours)):
            ix,iy,iw,ih = cv2.boundingRect(contours[x])
            min_dist=None
            min_idx=-1
            roi_mask = np.zeros_like(reconstructed_mask)[:,:,None]
            cv2.fillPoly(roi_mask, [contours[x]], 1)
            roi = np.multiply(reconstructed,roi_mask)
            roi_img = roi[:,:,:3].astype(np.uint8)
	    rect = cv2.minAreaRect(contours[x])
            roi_mask = roi[:,:,3].astype(np.uint8)
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
        water=cv2.watershed(cv2.cvtColor(reconstructed_mask, cv2.COLOR_GRAY2BGR),labels)-1
        water[water<0]=0 # set bg and unknown to 0
        return enhanced,water

    def guess_prepare(self,image_idxs):
        self.guess=GUESS()
        for idx in image_idxs:
            if idx<0:
                continue
            img,mask,mask_seg = self.images[idx]
            self.guess.prepare_fit(img,mask,mask_seg,torch=False)

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


    def shrink_nuclei(self,labels):
        labels_new=labels*0
        kernel = np.ones((3,3), np.uint8)
        offset=0
        for x in xrange(labels.max()):
            idx=x+1
            cur_mask = np.zeros_like(labels).astype(np.uint8)
            cur_mask[labels==idx]=255
            cur_mask = cv2.erode(cur_mask, kernel, iterations=1)
            if cur_mask.sum()<100:
                offset+=1
            else:
                labels_new[cur_mask>100]=idx-offset
        return labels_new

    def remove_overlap(self,labels):
        overlaps= np.zeros_like(labels).astype(np.uint8)
        kernel = np.ones((5,5), np.uint8)
        for x in xrange(labels.max()):
            idx=x+1
            cur_mask = np.zeros_like(labels).astype(np.uint8)
            cur_mask[labels==idx]=255
            cur_mask = cv2.dilate(cur_mask, kernel, iterations=1)
            cur_mask[cur_mask>100]=1
            overlaps+=cur_mask
        labels=labels.copy()
        labels[overlaps>1]=0
        return labels

    def remove_weird_labels(self,l,mask):
        l=l.copy()
        offset=0
        for x in xrange(l.max()):
            if x<=0:
                continue
            if mask[l==x].mean()<60:
                l[l==x]=0
                offset+=1
            else:
                l[l==x]=x-offset
        return l

    def predict(self,img,gt=None):
        #self.save('out_model.pkl')
	img = (img.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
	gt = (gt.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)

        similar_hist_images=[]
        similar_hist_images_idxs=self.similar_by_hist(img)
        for idx in similar_hist_images_idxs:
            if idx>=0:
                img2,_,_ = self.images[idx]
                similar_hist_images.append(cv2.resize(img2, (256, 256)))

        patch_model = None
        super_boundary_model = None
        if self.match_method=='hist':
            patch_model,self.super_boundary_model, self.super_boundary_2_model = self.make_index(similar_hist_images_idxs) #,use_all=True)
        else:
            print "INVALID MODEL TYPE"

        if patch_model==None:
            if self.faiss_model==None:
                self.faiss_model = self.make_index(xrange(len(self.images)))
            patch_model,super_boundary_model=self.faiss_model
        print "START IMG PATCH GEN"

        reconstructed=None
        for reconstruct_it in xrange(1):
            print "RECONSTRUCTING", reconstruct_it
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
            img=reconstructed_img.copy()
	reconstructed_mask = reconstructed[:,:,3].astype(np.uint8)
	reconstructed_boundary = reconstructed[:,:,4].astype(np.uint8)
	reconstructed_blend = reconstructed[:,:,5].astype(np.uint8)
	reconstructed_super_boundary = reconstructed[:,:,6].astype(np.uint8)
	reconstructed_super_boundary_2 = reconstructed[:,:,7].astype(np.uint8)
	#reconstructed_mask_eroded = reconstructed[:,:,7].astype(np.uint8)

        _,clustered=self.by_cluster(img,reconstructed)

        d={}
        d['img']=reconstructed_img
        d['seg']=reconstructed_mask
        d['boundary']=reconstructed_boundary
        d['blend']=reconstructed_blend
        d['super_boundary']=reconstructed_super_boundary
        d['super_boundary_2']=reconstructed_super_boundary_2
        d['similar_hist_images']=similar_hist_images
        d['clustered_r4']=self.shrink_nuclei(clustered)
	d['gt']=gt
	for x in ['clustered_r4']:
		d["color_"+x]=self.color_label(d[x])
        return d 
    
    
