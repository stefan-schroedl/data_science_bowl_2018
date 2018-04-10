import cv2
import sys
from KNN import *

if len(sys.argv)<2:
    print sys.argv[0] + " f1 f2 ..."
    sys.exit(1)

fns=sys.argv[2:]

d={}

for fn in fns:
    n="_".join(fn.replace('.png','').split('_')[2:])
    print fn,n
    d[n]=cv2.imread(fn)
    if (d[n][:,:,0]==d[n][:,:,1]).mean()==1.0 and (d[n][:,:,1]==d[n][:,:,2]).mean()==1.0:
        d[n]=(d[n][:,:,0])[:,:]

print d.keys()
model=KNN()
print d['seg'].dtype,d['seg'].shape
_,super_boundary_thresh = cv2.threshold(d['super_boundary'],20,255,cv2.THRESH_BINARY)
l=model.label(super_boundary_thresh.copy(),d['seg'].copy())
print l.shape
cl=model.color_label(l)
cv2.imshow('cl',cl)
cv2.waitKey(50000)


