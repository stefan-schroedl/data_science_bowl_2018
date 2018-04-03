import numpy as np
import cv2
import imutils
class GUESS():
    def __init__(self):
        pass



    def transform_into(self,img,sub_img,p,method='diff',show=False):
        if img.ndim==2:
            img=img[:,:,None]
        if img.shape[2]==1:
            img=np.concatenate((img,img,img),axis=2)
        if sub_img.ndim==2:
            sub_img=sub_img[:,:,None]
            sub_img=np.concatenate((sub_img,sub_img,sub_img),axis=2)
        y_shift,x_shift,rotation,x_scale,y_scale = p
	y_shift=int(y_shift)
	x_shift=int(x_shift)
	x_scale=max(abs(x_scale),0.2)
	y_scale=max(abs(y_scale),0.2)
        scaled = cv2.resize(sub_img, (0,0), fx=x_scale, fy=y_scale) 
        rotated = imutils.rotate_bound(scaled, rotation)
	sub_img = rotated
        height = img.shape[0]
        width = img.shape[1]
        sub_height = sub_img.shape[0]
        sub_width = sub_img.shape[1]
        roi_y_start=height/2-sub_height/2
        roi_x_start=width/2-sub_width/2

        #apply shift
        roi_y_start+=y_shift
        roi_x_start+=x_shift

        #if the images do not intersect at all... just return?
        if roi_y_start+sub_height<=0 or roi_y_start>=height:
            return -1,None
        if roi_x_start+sub_width<=0 or roi_x_start>=width:
            return -1,None

        sub_y_start=max(0,-roi_y_start) # start at the edge of sub_img, or if we are off above then start at the first pixel
        sub_y_end=min(sub_height-1,height-roi_y_start-1) # end at the bottom or if we are off below 
        sub_x_start=max(0,-roi_x_start) # start at the edge of sub_img, or if we are off above then start at the first pixel
        sub_x_end=min(sub_width-1,width-roi_x_start-1) # end at the bottom or if we are off below 

        roi_width=sub_x_end-sub_x_start+1
        roi_height=sub_y_end-sub_y_start+1
	
	roi_y_start=max(roi_y_start,0)
	roi_x_start=max(roi_x_start,0)

        img=img.copy().astype(np.float32)

        img_roi=img[roi_y_start:roi_y_start+roi_height,roi_x_start:roi_x_start+roi_width,:]
        sub_img_roi=sub_img[sub_y_start:sub_y_end+1,sub_x_start:sub_x_end+1,:]

	if method=='diff':
		img_roi-=sub_img_roi
	elif method=='paste':
		img_roi[:,:,:]=sub_img_roi
        else:
                print "unsupported method"

        img=img.astype(np.float)

        #original scoring
	#img[(img<0) * (img>-25)]=-25 # higher pentaly here #TODO FILL IN WITH BACKGROUND VARIANCE?
        #new scoring?
	#img[(img<0)]=-100 # higher pentaly here #TODO FILL IN WITH BACKGROUND VARIANCE?
	img=np.absolute(img)

	mse=np.multiply(img,img).sum()

	#lets get an image without the region of interest 

	if show:
		img2=img.copy()
		img2=(img2/img2.max())*255
		img2=img2.astype(np.uint8)
       		cv2.imshow('curmask',img2)
        	cv2.waitKey(100)
	return mse,img.astype(np.uint8)

        #sys.exit(1)
	
        
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

    #img is the img to search
    #sub img is the cropped needle
    #sub mask is the mask of the nuclei in the sub_img
    def search_fit(self,img,sub_img,sub_mask):
        # parameters , 
        # Y shift 
        # X shift
        # Rotation
        # X scale
        # Y scale
        shape = img.shape
        height = shape[0]
        width = shape[1]

        _, contours, hierarchy = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	base_mse=np.multiply(img,img).sum()
        keep=0.25
	for cnt in contours:
            rect = cv2.minAreaRect(cnt)
	    ximg=self.crop_minAreaRect(sub_img,rect)
	    ximg_mask=self.crop_minAreaRect(sub_mask,rect)
	    small_it=80
            for y in range(10):
		means = [0,0,0,1,1]
		vs = [ height*5, width*5 , 600, 2, 2 ]
		meta_param = {'u':np.array(means), 'o':np.array(vs)}
		best_p=None
		for y in xrange(150):
		    params = np.random.multivariate_normal(meta_param['u'], np.diag(meta_param['o']), small_it)
		    scores=[]
		    for x in xrange(params.shape[0]):
			mse,_=self.transform_into(img,ximg,params[x],show=False)
                        if mse>=0:
			    scores.append((mse-base_mse,x))
		    scores.sort(reverse=True)
		    tops=np.vstack([ params[i] for x,i in scores[int(len(scores)*(1-keep)):] ])
		    meta_param['u']=meta_param['u']*0.9+0.1*tops.mean(0)
		    meta_param['o']=meta_param['o']*0.9+0.1*tops.var(0)
		    #print meta_param['o']
		    #meta_param['o'][2]=max(meta_param['o'][2],100)
		    #box=cv2.boxPoints(rect)
		    #print sub_img[box]
		    print scores[0],meta_param['o']
		    best_p=params[scores[0][1]]
		    #_,i=self.transform_into(img,ximg,best_p,show=True)
                    #cv2.imshow('x',i)
                    #cv2.waitKey(1000)
		self.transform_into(img,ximg,best_p,show=True)
		cv2.waitKey(2000)
		_,img=self.transform_into(img,ximg_mask*0,best_p,show=True,method='paste')
		cv2.waitKey(100)

        sys.exit(1)

    def get_stacked(self,img,mask,mask_seg):
        kernel = np.ones((3,3), np.uint8)
        mask_eroded=cv2.erode(mask, kernel, iterations=1)[:,:,None].astype(np.uint8)
        
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
	img = (img.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        mask = (mask.numpy()[0].transpose(1,2,0)).astype(np.uint8)
        #kernel = np.ones((3,3), np.uint8)
        #mask_eroded=cv2.erode(mask, kernel, iterations=1)[:,:,None].astype(np.uint8)

        mask_seg = (mask_seg.numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        stacked=self.get_stacked(img,mask,mask_seg)

        boundary=stacked[:,:,4]
        k=35
        blurred_boundary=np.maximum(boundary,cv2.GaussianBlur(boundary,(k,k),0))
        for x in xrange(10):
            blurred_boundary=np.maximum(boundary,cv2.GaussianBlur(blurred_boundary,(k,k),0))
        blurred_boundary=blurred_boundary.astype(np.float)
        blurred_boundary/=blurred_boundary.max()
        blurred_boundary*=255
        blurred_boundary=blurred_boundary.astype(np.uint8)

    
        #self.images.append((img,mask,mask_seg,mask_eroded))
        #find and store the mask contours
        for x in xrange(mask.max()):
            idx=x+1
            cur_mask = np.zeros_like(mask).astype(np.uint8)
            cur_mask[mask==idx]=1
            _, contours, hierarchy = cv2.findContours(cur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #want to pass in hard contour as pin and blurred boundaries as haystack
            use_contours=False
            if use_contours:
                contour=img.copy()*0
                for y in xrange(len(contours)):
                    cv2.fillPoly(contour, [contours[y]], (255,255,255))
                    #cv2.drawContours(contour,[contours[y]],0,(255,255,255),2)
                #self.search_fit(blurred_boundary,contour.astype(np.uint8),cur_mask)
                self.search_fit(mask_seg,contour.astype(np.uint8),cur_mask)
            else:
                self.search_fit(img,np.multiply(cur_mask,img).astype(np.uint8),cur_mask)
            if len(contours)==0:
                print "OH NO..."
            if True:
                img=img.copy()
                for x in xrange(len(contours)):
                    color = np.random.randint(0,255,(3)).tolist() 
                    cv2.drawContours(img,[contours[x]],0,color,2)
            #self.mask_contours.append(contours)
        print "PREAPRE FIT",img.sum()

    def fit(self):
        pass

    def predict(self,img):
        pass