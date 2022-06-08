#-*- coding: utf-8 -*-
from __future__ import print_function
# ---------------------------------------------------------
# imports
# ---------------------------------------------------------
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from shapely.geometry import Polygon
import math
import pyclipper

def create_mask(image,regions,shrink_ratio=0.4):
    h,w,_=image.shape
    mask=np.zeros((h,w))
        
    for region in regions:
        polygon = np.array(region).reshape((-1, 2)).tolist()
        polygon = Polygon(polygon)
        distance = polygon.area * (1 - np.power(shrink_ratio, 2)) / polygon.length
        subject = [tuple(l) for l in region]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked = padding.Execute(-distance)
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
        cv2.fillPoly(mask, [shrinked.astype(np.int32)], 1)
    return mask