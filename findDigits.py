import glob
import time

import cv2
import tensorflow as tf
from PIL import Image
from sklearn.externals import joblib
import numpy as np

import pytorch
from predict import predict_using_pytorch_cnn


cnn_clf_pt = pytorch.model

#video_in = cv2.VideoCapture(0)

if True:
    # for webcam input
    #ret, frame = video_in.read()
    # for img input
    frame = cv2.imread('testImages/digirec2.jpg')
    rs_ratio = 1000/frame.shape[0]
    if frame.shape[0] < frame.shape[1]:
        rs_ratio = 1000/frame.shape[1]
    print("RS: "+str(rs_ratio))
    frame = cv2.resize(frame, (0, 0), fx=rs_ratio, fy=rs_ratio)

    cv2.imshow("frame", frame)

    min_size = max(frame.shape[0], frame.shape[1])/20
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray_inv = 255-img_gray

    im_th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 15)

    ii, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.imshow("thr", im_th)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs if ctr.size > min_size]

    rects_ok = rects

    print("FOUND: "+str(len(rects_ok)))

    c = 1
    # For each rectangular region, calculate HOG features and predict
    # the digit using SVC.
    for rect in rects_ok:
        print("-----------------------------")
        leng = int(rect[3] * 1.2)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

        if pt1 < 0:
            pt1 = 0
        if pt2 < 0:
            pt2 = 0

        roi = 255 - im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        # roi = cv2.blur(roi, (5, 5))

        roi_inv = 255 - roi

        #roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        roi_img = Image.fromarray(roi)
        pred = predict_using_pytorch_cnn(roi_img, cnn_clf_pt)
        #print("  "+str(pred[0][0]))


        #time.sleep(3)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), 2)
        if not pred[1] < -0.2:
            cv2.putText(frame, str(pred[0])+"("+str(round(pred[1], 2))+")", (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.imshow(str(pred[0]) + "=>" + str(round(pred[1], 4)) + "(" + str(c) + ")", roi)

    cv2.imshow("frame", frame)
    cv2.imwrite("out.jpg", frame)

cv2.waitKey(0)
