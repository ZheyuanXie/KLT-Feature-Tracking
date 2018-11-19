'''
  File name: getFeatures.py
  Author: 
  Date created: 2018-11-04
'''

import cv2
import numpy as np
from skimage.feature import corner_harris, corner_shi_tomasi, peak_local_max

def getFeatures(img,bbox,use_shi=False):
    n_object = np.shape(bbox)[0]
    N = 0
    temp = np.empty((n_object,),dtype=np.ndarray)   # temporary storage of x,y coordinates
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:].astype(int))
        roi = img[ymin:ymin+boxh,xmin:xmin+boxw]
        # cv2.imshow('roi',roi)
        if use_shi:
            corner_response = corner_shi_tomasi(roi)
        else:
            corner_response = corner_harris(roi)
        coordinates = peak_local_max(corner_response,num_peaks=20,exclude_border=2)
        coordinates[:,1] += xmin
        coordinates[:,0] += ymin
        temp[i] = coordinates
        if coordinates.shape[0] > N:
            N = coordinates.shape[0]
    x = np.full((N,n_object),-1)
    y = np.full((N,n_object),-1)
    for i in range(n_object):
        n_feature = temp[i].shape[0]
        x[0:n_feature,i] = temp[i][:,1]
        y[0:n_feature,i] = temp[i][:,0]
    return x,y

if __name__ == "__main__":
    cap = cv2.VideoCapture("Easy.mp4")
    ret, frame = cap.read()  # get first frame

    # This section is for generating bounding boxes
    # n_object =  int(input("Number of objects to track:"))
    # bbox = np.empty((n_object,4,2), dtype=int)
    # for i in range(n_object):
    #     (xmin, ymin, boxw, boxh) = cv2.selectROI("Select Object %d"%(i),frame)
    #     cv2.destroyWindow("Select Object %d"%(i))
    #     bbox[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]])

    # We can use fixed
    n_object = 1
    bbox = np.array([[[291,187],[405,187],[291,267],[405,267]]])

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    x,y = getFeatures(frame_gray,bbox)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()
    ax.imshow(frame_gray,cmap='gray')
    ax.scatter(x[x!=-1],y[y!=-1],color=(1,0,0))
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:])
        patch = Rectangle((xmin,ymin),boxw,boxh,fill=False,color=(0,0,0),linewidth=3)
        ax.add_patch(patch)
    plt.show()
    