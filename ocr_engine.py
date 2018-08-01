import cv2
import cPickle as cp
import numpy as np
from skimage.feature import hog


def load_model():
    with open("knn-hog-100.mdl") as fp:
        clf = cp.load(fp)
    return clf


def sort_contours(cnts):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 500]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i]))
    return (cnts, boundingBoxes)


def predict_digit(im, im_th, rect):
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2],
                                           rect[1] + rect[3]), (0, 255, 0), 1)
    leng = int(rect[3] * 1.2)
    newx = rect[0] - (leng - rect[2]) // 2
    newl = leng
    newy = rect[1] - 10
    newh = rect[3] + 20
    roi = im_th[newy:newy + newh, newx:newx + newl]
    #cv2.rectangle(im, (newx, newy), (newx + newl, newy + newh), (0, 0, 255), 1)
    if len(roi) > 0:
        try:
            roi = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            to_predict = hog(roi, orientations=9, pixels_per_cell=(
                10, 10), cells_per_block=(1, 1))
            nbr = clf.predict([to_predict])
            cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            return True, nbr[0]
        except:
            return False, None
    else:
        return False, None


def predict_img(name, app=False):
    im = cv2.imread(name)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    image, ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts, rects = sort_contours(ctrs)
    if app:
        expr = ""
        tracker = 0
        for rect in rects:

            if tracker == 2:
                tracker = 0
            else:
                ret, digit = predict_digit(im, im_th, rect)
                if ret:
                    tracker += 1
                    expr += str(digit)

        ans = int(expr[:2]) * int(expr[2:])
        height, width = im.shape[:2]

        cv2.putText(im, "=" + str(ans), (width / 3, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    else:
        for rect in rects:
            predict_digit(im, im_th, rect)

    cv2.imshow("Result", im)
    cv2.waitKey()


clf = load_model()
predict_img("./test.jpg")
