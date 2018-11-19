'''
  File name: applyGeometricTransformation.py
  Author: 
  Date created: 2018-11-05
'''

import cv2
import numpy as np
from skimage import transform as tf

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    n_object = bbox.shape[0]
    newbbox = np.zeros_like(bbox)
    Xs = newXs.copy()
    Ys = newYs.copy()
    for obj_idx in range(n_object):
        startXs_obj = startXs[:,[obj_idx]]
        startYs_obj = startYs[:,[obj_idx]]
        newXs_obj = newXs[:,[obj_idx]]
        newYs_obj = newYs[:,[obj_idx]]
        desired_points = np.hstack((startXs_obj,startYs_obj))
        actual_points = np.hstack((newXs_obj,newYs_obj))
        t = tf.SimilarityTransform()
        t.estimate(dst=actual_points, src=desired_points)
        mat = t.params

        # estimate the new bounding box with all the feature points
        # coords = np.vstack((bbox[obj_idx,:,:].T,np.array([1,1,1,1])))
        # new_coords = mat.dot(coords)
        # newbbox[obj_idx,:,:] = new_coords[0:2,:].T

        # estimate the new bounding box with only the inliners (Added by Yongyi Wang)
        THRES = 1
        projected = mat.dot(np.vstack((desired_points.T.astype(float),np.ones([1,np.shape(desired_points)[0]]))))
        distance = np.square(projected[0:2,:].T - actual_points).sum(axis = 1)
        actual_inliers = actual_points[distance < THRES]
        desired_inliers = desired_points[distance < THRES]
        if np.shape(desired_inliers)[0]<4:
            print('too few points')
            actual_inliers = actual_points
            desired_inliers = desired_points
        t.estimate(dst=actual_inliers, src=desired_inliers)
        mat = t.params
        coords = np.vstack((bbox[obj_idx,:,:].T,np.array([1,1,1,1])))
        new_coords = mat.dot(coords)
        newbbox[obj_idx,:,:] = new_coords[0:2,:].T
        Xs[distance >= THRES, obj_idx] = -1
        Ys[distance >= THRES, obj_idx] = -1

    return Xs, Ys, newbbox

if __name__ == "__main__":
    from getFeatures import getFeatures
    from estimateAllTranslation import estimateAllTranslation

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

   
    # This section is for generating bounding boxes
    # n_object = int(input("Number of objects to track:"))
    # bbox = np.empty((n_object,4,2), dtype=int)
    # for i in range(n_object):
    #     (xmin, ymin, boxw, boxh) = cv2.selectROI("Select Object %d"%(i),frame1)
    #     cv2.destroyWindow("Select Object %d"%(i))
    #     bbox[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]])

    # We can use fixed
    n_object = 1
    bbox = np.array([[[291,187],[405,187],[291,267],[405,267]]]).astype(float)

    startXs,startYs = getFeatures(frame1_gray,bbox)
    newXs, newYs =  estimateAllTranslation(startXs, startYs, frame1, frame2)
    Xs, Ys ,newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)


    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()

    diff = np.subtract(frame1_gray.astype(int),frame2_gray.astype(int))
    ax.imshow(cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB))
    # ax.scatter(newXs[newXs!=-1],newYs[newYs!=-1],color=(0,1,0))
    # ax.scatter(startXs[startXs!=-1],startYs[startYs!=-1],color=(1,0,0))

    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:].astype(int))
        patch = Rectangle((xmin,ymin),boxw,boxh,fill=False,color=(1,0,0),linewidth=2)
        ax.add_patch(patch)
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(newbbox[i,:,:].astype(int))
        patch = Rectangle((xmin,ymin),boxw,boxh,fill=False,color=(0,1,0),linewidth=2)
        ax.add_patch(patch)
    plt.show()