'''
  File name: estimateAllTranslation.py
  Author: 
  Date created: 2018-11-05
'''

from  utils import GaussianPDF_2D,rgb2gray
from scipy import signal
import numpy as np
import cv2
from getFeatures import getFeatures
from estimateFeatureTranslation import estimateFeatureTranslation

def estimateAllTranslation(startXs,startYs,img1,img2):
    G = GaussianPDF_2D(0, 0.5, 7, 7)
    I = rgb2gray(img1)
    # J = rgb2gray(img2)
    dx, dy = np.gradient(G, axis=(1,0))
    Iy = signal.convolve2d(I, dy, 'same')
    Ix = signal.convolve2d(I, dx, 'same')
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
    # ret, frame2 = cap.read()  # get second frame
    frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2,cv2.COLOR_RGB2GRAY)

    n_object = 1
    bbox = np.array([[[291,187],[405,187],[291,267],[405,267]]])

    
    startXs,startYs = getFeatures(frame1_gray,bbox)
    newXs, newYs =  estimateAllTranslation(startXs, startYs, frame1, frame2)
    # print(startXs - newXs)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()

    diff = np.subtract(frame1_gray.astype(int),frame2_gray.astype(int))
    # print(diff[:,40])
    # cv2.imshow("win",diff)
    # cv2.waitKey()
    ax.imshow(diff,cmap='gray')
    ax.scatter(newXs[newXs!=-1],newYs[newYs!=-1],color=(0,1,0))
    ax.scatter(startXs[startXs!=-1],startYs[startYs!=-1],color=(1,0,0))

    # for i in range(n_object):
    #     (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:])
    #     patch = Rectangle((xmin,ymin),boxw,boxh,fill=False,color=(0,0,0),linewidth=3)
    #     ax.add_patch(patch)
    plt.show()


