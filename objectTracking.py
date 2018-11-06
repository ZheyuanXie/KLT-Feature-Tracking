import cv2
import numpy as np

from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def objectTracking(rawVideo):
    n_frame = 100
    frames = np.empty((n_frame,),dtype=np.ndarray)
    bboxs = np.empty((n_frame,),dtype=np.ndarray)
    coords = np.full((n_frame,1000,2),-1)
    for frame_idx in range(n_frame):
        _, frames[frame_idx] = cap.read()
        # _, frames[frame_idx] = cap.read()
        # _, frames[frame_idx] = cap.read()
        # _, frames[frame_idx] = cap.read()
        # _, frames[frame_idx] = cap.read()
    
    # n_object = 1
    # bboxs[0] = np.array([[[291,187],[405,187],[291,267],[405,267]]]).astype(float)

    n_object = 3 # int(input("Number of objects to track:"))
    # This section is for generating bounding boxes
    bboxs[0] = np.empty((n_object,4,2), dtype=float)
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.selectROI("Select Object %d"%(i),frames[0])
        cv2.destroyWindow("Select Object %d"%(i))
        bboxs[0][i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)

    startXs,startYs = getFeatures(cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY),bboxs[0])
    for i in range(1,n_frame):
        print('Start',i)
        newXs, newYs = estimateAllTranslation(startXs, startYs, frames[i-1], frames[i])
        Xs, Ys ,bboxs[i] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[i-1])
        print('Finish',i)
        startXs = newXs
        startYs = newYs
        # coords[i,0:np.size(newXs),0:1] = newXs.flatten()
        # coords[i,0:np.size(newYs),1:2] = newYs.flatten()

        # fig, ax = plt.subplots()
        # # diff = np.subtract(frames[i-1].astype(int),frames[i].astype(int))
        # ax.imshow(frames[i],cmap='gray')
        # ax.scatter(newXs[newXs!=-1],newYs[newYs!=-1],color=(0,1,0))
        # ax.scatter(startXs[startXs!=-1],startYs[startYs!=-1],color=(1,0,0))
        # (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[i-1][0,:,:])
        # patch = Rectangle((xmin,ymin),boxw,boxh,fill=False,color=(1,0,0),linewidth=3)
        # ax.add_patch(patch)
        # (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[i][0,:,:])
        # patch = Rectangle((xmin,ymin),boxw,boxh,fill=False,color=(0,1,0),linewidth=3)
        # ax.add_patch(patch)
        # plt.show()

    while 1:
        for i in range(1,n_frame):
            for j in range(n_object):
                frame_bb = cv2.rectangle(frames[i], (int(bboxs[i][j,0,0]), int(bboxs[i][j,0,1])), (int(bboxs[i][j,3,0]), int(bboxs[i][j,3,1])), (255,0,0), 2)
            # frame_bb = cv2.circle(frame_bb,tuple(coords[i,:,:].tolist()),1,(0,0,255))
            cv2.imshow("win",frame_bb)
            cv2.waitKey(50)
            

if __name__ == "__main__":


    cap = cv2.VideoCapture("Easy.mp4")
    objectTracking(cap)

    # ret, frame1 = cap.read()  # get first frame
    # ret, frame2 = cap.read()  # get second frame
    # ret, frame2 = cap.read()  # get second frame
    # ret, frame2 = cap.read()  # get second frame
    # # ret, frame2 = cap.read()  # get second frame
    # frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
    # frame2_gray = cv2.cvtColor(frame2,cv2.COLOR_RGB2GRAY)

    # n_object = int(input("Number of objects to track:"))
    # # This section is for generating bounding boxes
    # bbox = np.empty((n_object,4,2), dtype=int)
    # for i in range(n_object):
    #     (xmin, ymin, boxw, boxh) = cv2.selectROI("Select Object %d"%(i),frame1)
    #     cv2.destroyWindow("Select Object %d"%(i))
    #     bbox[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]])

    # # We can use fixed
    # # n_object = 1
    # # bbox = np.array([[[291,187],[405,187],[291,267],[405,267]]])

    # startXs,startYs = getFeatures(frame1_gray,bbox)
    # newXs, newYs =  estimateAllTranslation(startXs, startYs, frame1, frame2)
    # Xs, Ys ,newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)


    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Rectangle
    # fig, ax = plt.subplots()

    # diff = np.subtract(frame1_gray.astype(int),frame2_gray.astype(int))
    # ax.imshow(diff,cmap='gray')
    # ax.scatter(newXs[newXs!=-1],newYs[newYs!=-1],color=(0,1,0))
    # ax.scatter(startXs[startXs!=-1],startYs[startYs!=-1],color=(1,0,0))

    # for i in range(n_object):
    #     (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:])
    #     patch = Rectangle((xmin,ymin),boxw,boxh,fill=False,color=(1,0,0),linewidth=3)
    #     ax.add_patch(patch)
    #     (xmin, ymin, boxw, boxh) = cv2.boundingRect(newbbox[i,:,:])
    #     patch = Rectangle((xmin,ymin),boxw,boxh,fill=False,color=(0,1,0),linewidth=3)
    #     ax.add_patch(patch)
    # plt.show()