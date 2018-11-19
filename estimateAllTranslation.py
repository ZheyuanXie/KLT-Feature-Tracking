'''
  File name: estimateAllTranslation.py
  Author: 
  Date created: 2018-11-05
'''

import numpy as np
import cv2
from scipy import signal

from getFeatures import getFeatures
from estimateFeatureTranslation import estimateFeatureTranslation

def estimateAllTranslation(startXs,startYs,img1,img2):
    I = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    I = cv2.GaussianBlur(I,(5,5),0.2)
    Iy, Ix = np.gradient(I.astype(float))

    startXs_flat = startXs.flatten()
    startYs_flat = startYs.flatten()
    newXs = np.full(startXs_flat.shape,-1,dtype=float)
    newYs = np.full(startYs_flat.shape,-1,dtype=float)
    for i in range(np.size(startXs)):
        if startXs_flat[i] != -1:
            newXs[i], newYs[i] = estimateFeatureTranslation(startXs_flat[i], startYs_flat[i], Ix, Iy, img1, img2)
    newXs = np.reshape(newXs, startXs.shape)
    newYs = np.reshape(newYs, startYs.shape)
    return newXs, newYs

if __name__ == "__main__":
    cap = cv2.VideoCapture("Easy.mp4")
    ret, frame1 = cap.read()  # get first frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2,cv2.COLOR_RGB2GRAY)

    n_object = 1
    bbox = np.array([[[291,187],[405,187],[291,267],[405,267]]])

    startXs,startYs = getFeatures(frame1_gray,bbox)
    newXs, newYs =  estimateAllTranslation(startXs, startYs, frame1, frame2)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()
    diff = np.subtract(frame1_gray.astype(int),frame2_gray.astype(int))
    ax.imshow(diff,cmap='gray')
    ax.scatter(newXs[newXs!=-1],newYs[newYs!=-1],color=(0,1,0))
    ax.scatter(startXs[startXs!=-1],startYs[startYs!=-1],color=(1,0,0))
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:])
        patch = Rectangle((xmin,ymin),boxw,boxh,fill=False,color=(0,0,0),linewidth=1)
        ax.add_patch(patch)
    plt.show()
