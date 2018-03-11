from skimage.morphology import reconstruction
from skimage import img_as_float, exposure
from skimage.util import invert
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, dilation, binary_dilation, binary_opening, opening, closing, white_tophat
from skimage.morphology import disk
from scipy import ndimage as ndi

# same as v1, but using skimage instead of cv2
def parametric_pipeline(img,
                invert_thresh_pd = .5,
                circle_size = 7,
                disk_size=10,
                min_distance=9,
                use_watershed=False
                ):
    
    circle_size = np.clip(int(circle_size), 1, 30)
    if use_watershed:
        disk_size = np.clip(int(disk_size), 0, 50)
        min_distance = np.clip(int(min_distance), 1, 50)

    # Invert the image in case the objects of interest are in the dark side
    
    thresh = threshold_otsu(img)
    img_th = img > thresh

    if len(np.where(img_th)[0]) > invert_thresh_pd * img.size:
        img=invert(img)

    # morphological opening (size tuned on training data)
    #circle7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(circle_size, circle_size))
    circle7=disk(circle_size / 2.0)
    img_open = opening(img, circle7)
    #img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, circle7)
    #return img_open

    thresh = threshold_otsu(img_open)
    img_th = (img_open > thresh).astype(int)
    
    # second morphological opening (on binary image this time)
    bin_open = binary_opening(img_th, circle7)
    if not use_watershed:
        return ndi.label(bin_open)[0]

    # WATERSHED
    selem=disk(disk_size)
    dil = binary_dilation(bin_open, selem)
    img_dist = ndi.distance_transform_edt(dil)
    local_maxi = peak_local_max(img_dist,
                            min_distance=min_distance,
                             indices=False,
                           exclude_border=False)
    markers = ndi.label(local_maxi)[0]
    cc = watershed(-img_dist, markers, mask=bin_open, compactness=0, watershed_line=True)

    return cc

def parametric_pipeline_v1(img,
                invert_thresh_pd = 10,
                circle_size = 7,
                disk_size=10,
                min_distance=9,
                use_watershed=False
                ):
    
    circle_size = np.clip(int(circle_size), 1, 30)
    if use_watershed:
        disk_size = np.clip(int(disk_size), 0, 50)
        min_distance = np.clip(int(min_distance), 1, 50)

    # Invert the image in case the objects of interest are in the dark side
        
    img_grey = img_as_ubyte(img)
    img_th = cv2.threshold(img_grey,0,255,cv2.THRESH_OTSU)[1]

    if(np.sum(img_th==255)>((invert_thresh_pd/10.0)*np.sum(img_th==0))):
        print 'invert'
        img=invert(img)
        
    # reconstruction with dilation
    # best h value = 0.87
    #img_float = img_as_float(img)
    #seed = img_float - h
    #dilated = reconstruction(seed, img_float, method='dilation')
    #hdome = img_float - dilated
    #print 'hdome', hdome.min(), hdome.max()
    #print 'image', img.min(), img.max()
    #print 'dilated', dilated.min(), dilated.max()
    #show_images([img,dilated,hdome])
    hdome = img
    
    # NOTE: the ali pipeline does the opening before flipping, makes a difference
    # at the edges!
    
    # morphological opening (size tuned on training data)
    circle7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(circle_size, circle_size))
    img_open = cv2.morphologyEx(hdome, cv2.MORPH_OPEN, circle7)
    
    # thresholding
    # isodata thresholding is comparable to otsu ...
    #Otsu thresholding
    
    img_grey = img_as_ubyte(img_open)
    #th = skimage.filters.threshold_isodata(img_grey)
    #img_th = cv2.threshold(img_grey,th,255,cv2.THRESH_BINARY)[1]
    img_th = cv2.threshold(img_grey,0,255,cv2.THRESH_OTSU)[1]
    
    # second morphological opening (on binary image this time)
    bin_open = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7)
    if not use_watershed:
        return cv2.connectedComponents(bin_open)[1]

    # WATERSHED
    selem=disk(disk_size)
    dil = binary_dilation(bin_open, selem)
    img_dist = ndi.distance_transform_edt(dil)
    local_maxi = peak_local_max(img_dist,
                            min_distance=min_distance,
                             indices=False,
                           exclude_border=False)
    markers = ndi.label(local_maxi)[0]
    cc = watershed(-img_dist, markers, mask=bin_open, compactness=0, watershed_line=True)

    return cc


# original, verified to be the same as ali
def parametric_pipeline_orig(img_green,
                invert_thresh_pd = 10,
                circle_size_x = 7,
                circle_size_y = 7,
                ):
    circle_size_x = np.clip(int(circle_size_x), 1, 30)
    circle_size_y = np.clip(int(circle_size_y), 1, 30)
    
    #green channel happends to produce slightly better results
    #than the grayscale image and other channels
    #morphological opening (size tuned on training data)
    circle7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(circle_size_x, circle_size_y))
    img_open=cv2.morphologyEx(img_green, cv2.MORPH_OPEN, circle7)
    #Otsu thresholding
    img_th=cv2.threshold(img_open,0,255,cv2.THRESH_OTSU)[1]
    #Invert the image in case the objects of interest are in the dark side
    if(np.sum(img_th==255)>((invert_thresh_pd/10.0)*np.sum(img_th==0))):
        img_th=cv2.bitwise_not(img_th)
    #second morphological opening (on binary image this time)
    bin_open=cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7) 
    #connected components
    cc=cv2.connectedComponents(bin_open)[1]
    #cc=segment_on_dt(bin_open,20)
    return cc

def ali_pipeline(img_green):
    #green channel happends to produce slightly better results
    #than the grayscale image and other channels
    #morphological opening (size tuned on training data)
    circle7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    img_open=cv2.morphologyEx(img_green, cv2.MORPH_OPEN, circle7)
    #Otsu thresholding
    img_th=cv2.threshold(img_open,0,255,cv2.THRESH_OTSU)[1]
    #Invert the image in case the objects of interest are in the dark side
    if(np.sum(img_th==255)>np.sum(img_th==0)):
        print 'invert ali'
        img_th=cv2.bitwise_not(img_th)
    #second morphological opening (on binary image this time)
    bin_open=cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7) 
    #connected components
    cc=cv2.connectedComponents(bin_open)[1]
    #cc=segment_on_dt(bin_open,20)
    return cc

def ali_pipeline_bk(img_green):
    #green channel happends to produce slightly better results
    #than the grayscale image and other channels
    #morphological opening (size tuned on training data)
    circle7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    img_open=cv2.morphologyEx(img_green, cv2.MORPH_OPEN, circle7)
    #Otsu thresholding
    img_th=cv2.threshold(img_open,0,255,cv2.THRESH_OTSU)[1]
    #Invert the image in case the objects of interest are in the dark side
    if(np.sum(img_th==255)>np.sum(img_th==0)):
        img_th=cv2.bitwise_not(img_th)
    #second morphological opening (on binary image this time)
    bin_open=cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7) 
    #connected components
    cc=cv2.connectedComponents(bin_open)[1]
    #cc=segment_on_dt(bin_open,20)
    return #cc