#!/usr/bin/env python
'''
Fast inplementation of Run-Length Encoding algorithm
Takes only 200 seconds to process 5635 mask files
'''

import numpy as np
import os
import skimage
from skimage.io import imread
from matplotlib import _cntr as cntr
import matplotlib.pyplot as plt

# RLE encoding

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def check_encoding():
    input_path = '../input/train'
    masks = [f for f in os.listdir(input_path) if f.endswith('_mask.tif')]
    masks = sorted(masks, key=lambda s: int(s.split('_')[0])*1000 + int(s.split('_')[1]))

    encodings = []
    N = 100     # process first N masks
    for i,m in enumerate(masks[:N]):
        if i % 10 == 0: print('{}/{}'.format(i, len(masks)))
        img = Image.open(os.path.join(input_path, m))
        x = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1])
        x = x // 255
        encodings.append(rle_encoding(x))

    #check output
    conv = lambda l: ' '.join(map(str, l)) # list -> string
    subject, img = 1, 1
    print('\n{},{},{}'.format(subject, img, conv(encodings[0])))

    # train_masks.csv:
    print('1,1,168153 9 168570 15 168984 22 169401 26 169818 30 170236 34 170654 36 171072 39 171489 42 171907 44 172325 46 172742 50 173159 53 173578 54 173997 55 174416 56 174834 58 175252 60 175670 62 176088 64 176507 65 176926 66 177345 66 177764 67 178183 67 178601 69 179020 70 179438 71 179857 71 180276 71 180694 73 181113 73 181532 73 181945 2 181950 75 182365 79 182785 79 183205 78 183625 78 184045 77 184465 76 184885 75 185305 75 185725 74 186145 73 186565 72 186985 71 187405 71 187825 70 188245 69 188665 68 189085 68 189506 66 189926 65 190346 63 190766 63 191186 62 191606 62 192026 61 192446 60 192866 59 193286 59 193706 58 194126 57 194546 56 194966 55 195387 53 195807 53 196227 51 196647 50 197067 50 197487 48 197907 47 198328 45 198749 42 199169 40 199589 39 200010 35 200431 33 200853 29 201274 27 201697 20 202120 15 202544 6')


def prob_to_rles(x, cut_off = 0.5):
    lab_img = label(x>cut_off)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)


## Eval metrics

def precision(matches):
    matches = matches.astype(int)
    true_positives = (np.sum(matches, axis=1) == 1).astype(int)   # Correct objects
    false_positives = (np.sum(matches, axis=0) == 0).astype(int)  # Extra objects
    false_negatives = (np.sum(matches, axis=1) == 0).astype(int)  # Missed objects
    over_seg = (np.sum(matches, axis=1) > 1).astype(int)   # Objects predicted multiple times
    under_seg = (np.sum(matches, axis=0) > 1).astype(int)   # Predictions overlapping multiple objects
    tp, fp, fn, oseg, useg = (np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives),
       np.sum(over_seg), np.sum(under_seg))
    return tp, fp, fn, oseg, useg

def union_intersection(labels, y_pred, exclude_bg=True):

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    #np.set_printoptions(threshold=np.nan)
    #print intersection
    
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
    
    if exclude_bg:
        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        area_true = area_true[1:,]
        area_pred = area_pred[:,1:]
        
    union[union == 0] = 1e-9



    return union, intersection, area_true, area_pred


def iou_metric(labels, y_pred, print_table=False):

    union, intersection, _, _ = union_intersection(labels, y_pred)
    
    # Compute the intersection over union
    iou = intersection.astype(float) / union

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn, _, _ = precision(iou > t)
            
        if (tp + fp + fn) > 0:
            p = 1.0 * tp / (tp + fp + fn)
        else:
            p = 0.0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


# see the SDS paper
def diagnose_errors(labels, y_pred, threshold=.5, print_message=True):

    union, intersection, area_true, area_pred = union_intersection(labels, y_pred)

    # Compute the intersection over union
    iou = intersection.astype(float) / union

    matches = (iou > threshold).astype(int)
    tp, fp, fn, oseg, useg = precision(matches)

    denom = 1.0 * (tp + fp + fn)
    p = 0.0
    p_loc = 0.0
    
    if denom > 0:
        p = tp / denom
        oseg = oseg / denom
        useg = useg / denom
    else:
        oseg = 0.0
        useg = 0.0

    # what can be achieved with locations fixed?

    # sort newly matched indices by iou
    matches_loc = np.copy(matches)
    matches0 = np.where(np.logical_and(iou > 0.1, iou < threshold))
    matches0 = sorted([(x,y,iou[x,y]) for x,y in zip(matches0[0], matches0[1])], key = lambda x: -x[2])

    for x,y,_ in matches0:
        if np.sum(matches_loc[x,:]) == 0 and np.sum(matches_loc[:,y]) == 0:
            # both parts of the pair have not been matched, assign greedily
            #print 'assigning', x, y
            matches_loc[x,y]=1.0
    
    tp_loc, fp_loc, fn_loc, _, _ = precision(matches_loc)
    #print 'loc', tp_loc, fp_loc, fn_loc
                
    #np.set_printoptions(threshold=np.nan)
    #print iou
    #print np.sum(matches,axis=0)
    #print np.sum(matches,axis=1)
    #print matches_loc[:,23]

    denom = 1.0 * (tp_loc + fp_loc + fn_loc)
    if denom > 0:
        p_loc = tp_loc / denom

    prec_thresh = 0.67
    
    # precision measure
    prec = intersection.astype(float) / np.tile(area_pred, (intersection.shape[0], 1))

    #tp_prec, fp_prec, fn_prec, _, _ = precision(prec > prec_thresh)
    #print 'prec',  tp_prec, fp_prec, fn_prec
    #denom = 1.0 * (tp_prec + fp_prec + fn_prec)
    #if denom > 0:
    #    p_prec = tp_prec / denom
    #else:
    #    p_prec = 0.0

    # recall measure
    rec = intersection.astype(float) / np.tile(area_true, (1, intersection.shape[1]))

    # this leads to duplicates, not sure how to handle it properly ...
    #tp_rec, fp_rec, fn_rec, _, _ = precision(rec > prec_thresh)
    #print 'rec',  tp_rec, fp_rec, fn_rec
     
    #denom = 1.0 * (tp_rec + fp_rec + fn_rec)
    #if denom > 0:
    #    p_rec = tp_rec / denom
    #else:
    #    p_rec = 0.0

    mean_prec = np.mean(prec[(iou > threshold)])
    mean_rec = np.mean(rec[(iou > threshold)])
    
    if print_message:
        s = 'average precision: %.3f; ignoring mislocations: %.3f; oversegmentation: %.3f; undersegmentation: %.3f' % (p, p_loc, oseg, useg)
        if mean_prec > mean_rec:
            s = s + ' segments tend to be too small:'
        else:
            s = s + ' segments tend to be too large:'
        s = s + ' pixel precision: %.3f, pixel recall: %.3f' % (mean_prec, mean_rec)
        print(s)

    return p, p_loc, mean_prec, mean_rec, oseg, useg


def read_img_join_masks(img_id, root='../../input/stage1_train/'):
    img = imread(os.path.join(root, img_id, 'images', img_id + '.png'))
    path = os.path.join(root, img_id, 'masks')
    mask = None
    i = 0
    for mask_file in next(os.walk(path))[2]:
        if mask_file.endswith('png'):
            i = i + 1
            mask_ = imread(os.path.join(path, mask_file)).astype(int)
            mask_[mask_>0] = i
            if mask is None:
                mask = mask_
            else:
                mask = np.maximum(mask, mask_)
    return img, mask

# https://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
# add contour to plot, without directly plotting!
def add_contour(z, ax):

    x, y = np.mgrid[:z.shape[0], :z.shape[1]]
    c = cntr.Cntr(x, y, z)

    # trace a contour at z == 0.5
    for at in range(1, np.amax(z)+1):
        res = c.trace(at)

        # result is a list of arrays of vertices and path codes
        # (see docs for matplotlib.path.Path)
        nseg = len(res) // 2
        segments, codes = res[:nseg], res[nseg:]

        for seg in segments:
            # for some reason, the coordinates are flipped???
            p = plt.Polygon([[x[1],x[0]] for x in seg], fill=False, color='black')
            ax.add_artist(p)

def show_img(img):
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.grid(None)
    ax.imshow(img)
    plt.tight_layout()
    plt.show()

    
def show_with_contour(img, mask):
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.grid(None)
    ax.imshow(img)
    add_contour(mask, ax)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':
    check_encoding()
   
