'''
  File name: getFeatures.py
  Author: 
  Date created: 2018-11-04
'''

import cv2
import numpy as np
from skimage.feature import corner_harris, corner_shi_tomasi, peak_local_max

def getFeatures(img,bbox,use_shi=True):
    n_object = np.shape(bbox)[0]
    N = 0
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:])
        roi = img[ymin:ymin+boxh,xmin:xmin+boxw]
        corner_response = corner_harris(roi)
        coordinates = peak_local_max(corner_response)
        if coordinates.shape[0] > N:
            N = coordinates.shape[0]
    x = np.zeros((N,n_object))
    y = np.zeros((N,n_object))
    return x,y

if __name__ == "__main__":
    cap = cv2.VideoCapture("Easy.mp4")
    ret, frame = cap.read()  # get first frame
    n_object = 1 # int(input("Number of objects to track:"))
    bbox = np.empty((n_object,4,2), dtype=int)
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.selectROI("Select Object %d"%(i),frame)
        cv2.destroyWindow("Select Object %d"%(i))
        bbox[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]])
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    x,y = getFeatures(frame_gray,bbox)

