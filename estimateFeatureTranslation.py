'''
  File name: estimateFeatureTranslation.py
  Author: 
  Date created: 2018-11-05
'''

import numpy as np
from numpy.linalg import inv
import cv2
from utils import interp2

def estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2):
    X=startX
    Y=startY
    mesh_x,mesh_y=np.meshgrid(np.arange(10),np.arange(10))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mesh_x_flat_fix =mesh_x.flatten() + X - 5
    mesh_y_flat_fix =mesh_y.flatten() + Y - 5
    coor_fix = np.vstack((mesh_x_flat_fix,mesh_y_flat_fix))
    Ix_value = interp2(Ix, coor_fix[0:1,:], coor_fix[1:2,:])
    Iy_value = interp2(Iy, coor_fix[0:1,:], coor_fix[1:2,:])
    I=np.vstack((Ix_value,Iy_value))
    A=I.dot(I.T)
    I1_value = interp2(img1_gray, coor_fix[0:1,:], coor_fix[1:2,:])

    for _ in range(3):
        mesh_x_flat=mesh_x.flatten() + X - 5
        mesh_y_flat=mesh_y.flatten() + Y - 5
        coor=np.vstack((mesh_x_flat,mesh_y_flat))
        I2_value = interp2(img2_gray, coor[0:1,:], coor[1:2,:])
        Ip=(I2_value-I1_value).reshape((-1,1))
        b=-I.dot(Ip)
        solution=inv(A).dot(b)
        X += solution[0,0]
        Y += solution[1,0]
    
    return X, Y