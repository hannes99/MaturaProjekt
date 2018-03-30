import time

import cv2
from PIL import Image

times = []
times_conv = []
from predict import predict_using_pytorch_cnn

start_load = time.time()

import pytorch

cnn_clf_pt = pytorch.model

print("TIME it took to load cnn-model: " + str(time.time() - start_load))

file_name = "ti7.jpg"


def classify(rect, im_th, result):
    leng = int(rect[3] * 1.1)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

    if pt1 < 0:
        pt1 = 0
    if pt2 < 0:
        pt2 = 0

    roi = 255 - im_th[pt1:pt1 + leng, pt2:pt2 + leng]
    # roi = cv2.blur(roi, (5, 5))

    # roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    # start_time = time.time()
    roi_img = Image.fromarray(roi)
    # times_conv.append(time.time() - start_time)

    start_time = time.time()
    pred = predict_using_pytorch_cnn(roi_img, cnn_clf_pt)
    times.append(time.time() - start_time)

    result[rect] = pred[0], pred[1]
    return


#video_in = cv2.VideoCapture(0)
if True:
    # for webcam input
    #ret, frame = video_in.read()
    # for img input
    frame = cv2.imread('testImages/' + file_name)
    start_time = time.time()
    rs_ratio = 1000/frame.shape[0]
    if frame.shape[0] < frame.shape[1]:
        rs_ratio = 1000/frame.shape[1]

    frame = cv2.resize(frame, (0, 0), fx=rs_ratio, fy=rs_ratio)
    print("Time it took to resize img: " + str(time.time() - start_time))
    # cv2.imshow("frame", frame)

    min_size = max(frame.shape[0], frame.shape[1])/20

    start_time = time.time()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("Time it took to greyscale img: " + str(time.time() - start_time))

    # img_gray_inv = 255-img_gray
    start_time = time.time()
    im_th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 15)
    print("Time it took to Threhhold img: " + str(time.time() - start_time))

    start_time = time.time()
    ii, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("Time it took to find Contours: " + str(time.time() - start_time))

    # cv2.imshow("thr", im_th)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs if ctr.size > min_size]

    rects_ok = rects

    print("FOUND: "+str(len(rects_ok)))

    c = 1
    results = {}
    threads = []
    # For each rectangular region predict
    start_time = time.time()

    for rect in rects_ok:
        classify(rect, im_th, results)
        # t = threading.Thread(target=classify, args=(rect, im_th, results))
        # threads.append(t)
        # t.start()

    print("TIME(to start): " + str(time.time() - start_time))

    start_time = time.time()

    for t in threads:
        t.join()

    print("TIME(to join): " + str(time.time() - start_time))
    for rect in results:
        pred = results[rect]
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), 2)
        if not pred[1] < -0.3:
            cv2.putText(frame, str(pred[0]), (rect[0], rect[1]),
                        cv2.FONT_HERSHEY_DUPLEX, 0.71, (0, 0, 0), 2)
    print("AVG Time it took to classify img: " + str(sum(times) / len(times)) + "(Total: " + str(sum(times)) + ")")
    # print("AVG Time it took to convert to img: "+str(sum(times_conv)/len(times_conv))+"(Total: "+str(sum(times_conv))+")")
    # cv2.imshow("frame", frame)
    cv2.imwrite("output/out(" + file_name + ").jpg", frame)

# cv2.waitKey(0)
