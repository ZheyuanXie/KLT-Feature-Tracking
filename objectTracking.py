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
    n_frame = 100
    frames = np.empty((n_frame,),dtype=np.ndarray)
    bboxs = np.empty((n_frame,),dtype=np.ndarray)
    # coords = np.full((n_frame,1000,2),-1)
    for frame_idx in range(n_frame):
        _, frames[frame_idx] = cap.read()
    
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

    startXs,startYs = getFeatures(cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY),bboxs[0])
    for i in range(1,n_frame):
        newXs, newYs = estimateAllTranslation(startXs, startYs, frames[i-1], frames[i])
        Xs, Ys ,bboxs[i] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[i-1])
        # startXs,startYs = getFeatures(cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY),bboxs[i])
        print('Processing Frame',i)
        startXs = newXs
        startYs = newYs
        # coords[i,0:np.size(newXs),0:1] = newXs.flatten()
        # coords[i,0:np.size(newYs),1:2] = newYs.flatten()

        if play_realtime:
            for j in range(n_object):
                frame_bb = cv2.rectangle(frames[i], (int(bboxs[i][j,0,0]), int(bboxs[i][j,0,1])), (int(bboxs[i][j,3,0]), int(bboxs[i][j,3,1])), (255,0,0), 2)
                    # frame_bb = cv2.circle(frame_bb,tuple(coords[i,:,:].tolist()),1,(0,0,255))
                cv2.imshow("win",frame_bb)
                cv2.waitKey(10)

    while 1:
        for i in range(1,n_frame):
            for j in range(n_object):
                frame_bb = cv2.rectangle(frames[i], (int(bboxs[i][j,0,0]), int(bboxs[i][j,0,1])), (int(bboxs[i][j,3,0]), int(bboxs[i][j,3,1])), (255,0,0), 2)
            # frame_bb = cv2.circle(frame_bb,tuple(coords[i,:,:].tolist()),1,(0,0,255))
            cv2.imshow("win",frame_bb)
            cv2.waitKey(50)


if __name__ == "__main__":
    cap = cv2.VideoCapture("Easy.mp4")
    objectTracking(cap,)
