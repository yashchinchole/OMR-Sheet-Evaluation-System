import re
from xml.sax import xmlreader
import cv2
import numpy as np

# TO STACK ALL THE IMAGES IN ONE WINDOW

def stackImages(imgArray, scale, Lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(
                    imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(Lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c*eachImgWidth, eachImgHeight*d), (c*eachImgWidth+len(
                    Lables[d])*13+27, 30+eachImgHeight*d), (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, Lables[d][c], (eachImgWidth*c+5, eachImgHeight *
                            d+20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 255), 1)
    return ver


def rectContours(contours,Area):

    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        # print("Area", area)
        if area > Area:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            # print("Corner Points", len(approx))

            if len(approx) == 4:
                rectCon.append(i)

    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    # print([cv2.contourArea(i) for i in rectCon])

    return rectCon


def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02*peri, True)
    return approx


def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print(myPoints)
    # print(add)

    myPointsNew[0] = myPoints[np.argmin(add)]   # [0 , 0]
    myPointsNew[3] = myPoints[np.argmax(add)]   # [w , h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]   # [width , 0]
    myPointsNew[2] = myPoints[np.argmax(diff)]  # [0 , height]
    # print(diff)
    return(myPointsNew)


def splitBoxes(img):
    rows = np.vsplit(img, 10)
    # cv2.imshow("Split Verically", rows[0])
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 4)
        for box in cols:
            boxes.append(box)
            # cv2.imshow("split", box)   # Shows Last Box
            # cv2.imshow("Test Box 1",boxes[0])
    return boxes


def splitBoxesRN(img):
    cols = np.hsplit(img, 8)
    # cv2.imshow("Split Horizontally", cols[0])
    boxes = []
    for c in cols:
        rows = np.vsplit(c, 10)
        for box in rows:
            boxes.append(box)
            # cv2.imshow("split", box)   # Shows Last Box
            # cv2.imshow("Test Box 1",boxes[4])
    return boxes


def splitBoxesSC(img):
    cols = np.hsplit(img, 3)
    # cv2.imshow("Split Horizontally", cols[0])
    boxes = []
    for c in cols:
        rows = np.vsplit(c, 10)
        for box in rows:
            boxes.append(box)
            # cv2.imshow("split", box)   # Shows Last Box
            # cv2.imshow("Test Box 1",boxes[4])
    return boxes

# def splitBoxesSN(img):
#     rows = np.vsplit(img, 4)
#     cv2.imshow("Split Horizontally", rows[0])
#     boxes = []

#     return boxes


def showAnswers(img, myIndex, grading, ans, questions, Choices):
    secW = int(img.shape[1]/Choices)
    secH = int(img.shape[0]/questions)

    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW//2
        cY = (x * secH) + secH//2

        if grading[x] == 1:
            myColor = (0, 255, 0)
        else:
            myColor = (0, 0, 255)
            coreectAns = ans[x]
            cv2.circle(img, ((coreectAns*secW) + secW // 2,
                       (x * secH) + secH // 2), 25, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (cX, cY), 25, (myColor), cv2.FILLED)

    return img


def determine_grade(scores):
    if scores >= 90 and scores <= 100:
        return 'A1'
    elif scores >= 80 and scores <= 89:
        return 'A2'
    elif scores >= 70 and scores <= 79:
        return 'B1'
    elif scores >= 60 and scores <= 69:
        return 'B2'
    elif scores >= 50 and scores <= 59:
        return 'C1'
    elif scores >= 40 and scores <= 49:
        return 'C2'
    else:
        return 'D'


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]),
             (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]),
             (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]),
             (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]),
             (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img


def nothing(x):
    pass


def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)


def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1, Threshold2
    return src
