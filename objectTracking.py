'''
  File name: objectTracking.py
  Author: 
  Date created: 2018-11-05
'''

import cv2
import numpy as np

from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation


def objectTracking(rawVideo, draw_bb=False, play_realtime=False):
    # initilize
    n_frame = 200
    frames = np.empty((n_frame,),dtype=np.ndarray)
    bboxs = np.empty((n_frame,),dtype=np.ndarray)
    for frame_idx in range(n_frame):
        _, frames[frame_idx] = cap.read()

    # draw rectangle roi for target objects, or use default objects initilization
    if draw_bb:
        n_object = int(input("Number of objects to track:"))
        bboxs[0] = np.empty((n_object,4,2), dtype=float)
        for i in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.selectROI("Select Object %d"%(i),frames[0])
            cv2.destroyWindow("Select Object %d"%(i))
            bboxs[0][i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
    else:
        n_object = 1
        bboxs[0] = np.array([[[291,187],[405,187],[291,267],[405,267]]]).astype(float)

    # Start from the first frame, do optical flow for every two consecutive frames.
    startXs,startYs = getFeatures(cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY),bboxs[0],use_shi=True)
    for i in range(1,n_frame):
        print('Processing Frame',i)
        newXs, newYs = estimateAllTranslation(startXs, startYs, frames[i-1], frames[i])
        Xs, Ys ,bboxs[i] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[i-1])
        # update feature points every other frame? (We might want to do this only on certain conditions)
        startXs = Xs
        startYs = Ys
        if np.shape(startXs)[0] < 20:
            print(i)
            startXs,startYs = getFeatures(cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY),bboxs[i])

        # draw bounding box and visualize feature point for each object
        for j in range(n_object):
            frames[i] = cv2.rectangle(frames[i], (int(bboxs[i][j,0,0]), int(bboxs[i][j,0,1])), (int(bboxs[i][j,3,0]), int(bboxs[i][j,3,1])), (255,0,0), 2)
            for k in range(newXs.shape[0]):
                frames[i] = cv2.circle(frames[i], (int(newXs[k,j]),int(newYs[k,j])),3,(0,0,255),thickness=2)
            # imshow if to play the result in real time
            if play_realtime:
                cv2.imshow("win",frames[i])
                cv2.waitKey(10)
        
        # update coordinates
        # startXs = newXs
        # startYs = newYs

    # loop the resulting video (for debugging purpose only)
    while 1:
        for i in range(1,n_frame):
            cv2.imshow("win",frames[i])
            cv2.waitKey(50)


if __name__ == "__main__":
    cap = cv2.VideoCapture("Easy.mp4")
    objectTracking(cap,draw_bb=True,play_realtime=False)
