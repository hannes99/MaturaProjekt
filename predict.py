import threading

import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from torch.autograd import Variable
from torchvision.transforms import transforms


def predict_using_cnn(img, classifier):
    test_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": img},
        num_epochs=1,
        shuffle=False
    )
    return list(classifier.predict(input_fn=test_input))[0]['classes']


def threaded_predict(x, clf_name, clf, all):
    pred = clf.predict_proba(x)[0]
    all[clf_name] = pred[1]


# only one used
def predict_using_pytorch_cnn(img, clsf):
    imsize = 28
    loader = transforms.Compose([transforms.Resize(imsize), transforms.CenterCrop(28), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    image = loader(img).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)

    pred = clsf(image.cuda()).data.max(1, keepdim=True)
    return pred[1][0][0], pred[0][0][0]


def predict_using_svm(img, clfs):
    roi_hog_fd = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    threads = []
    all = {}
    cv2.imshow("dbg", roi_hog_fd)
    x = np.array([roi_hog_fd], 'float64')
    for clf_name in clfs:
        clf = clfs[clf_name]
        thr = threading.Thread(target=threaded_predict, args=(x, clf_name, clf, all))
        thr.start()

    for t in threads:
        t.join()

    #if max(all.values()) > 0.5:
    return max(all, key=all.get)
    return "?"#[max(all.values()), max(all, key=all.get)]
