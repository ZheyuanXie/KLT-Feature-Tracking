'''
  File name: estimateFeatureTranslation.py
  Author: 
  Date created: 2018-11-05
'''

import numpy as np
from numpy.linalg import inv
import cv2
from utils import interp2

WINDOW_SIZE = 25

def estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2):
    X=startX
    Y=startY
    mesh_x,mesh_y=np.meshgrid(np.arange(WINDOW_SIZE),np.arange(WINDOW_SIZE))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    mesh_x_flat_fix =mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
    mesh_y_flat_fix =mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
    coor_fix = np.vstack((mesh_x_flat_fix,mesh_y_flat_fix))
    I1_value = interp2(img1_gray, coor_fix[[0],:], coor_fix[[1],:])
    Ix_value = interp2(Ix, coor_fix[[0],:], coor_fix[[1],:])
    Iy_value = interp2(Iy, coor_fix[[0],:], coor_fix[[1],:])
    I=np.vstack((Ix_value,Iy_value))
    A=I.dot(I.T)
   

    for _ in range(15):
        mesh_x_flat=mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
        mesh_y_flat=mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
        coor=np.vstack((mesh_x_flat,mesh_y_flat))
        I2_value = interp2(img2_gray, coor[[0],:], coor[[1],:])
        Ip=(I2_value-I1_value).reshape((-1,1))
        b=-I.dot(Ip)
        solution=inv(A).dot(b)
        # solution = np.linalg.solve(A, b)
        X += solution[0,0]
        Y += solution[1,0]
    
    return X, Y