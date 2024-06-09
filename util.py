import cv2
import numpy as np

def rectContours(contours, area):
    rectCon = []
    for i in contours:
        if cv2.contourArea(i) > area:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    return cv2.approxPolyDP(cont, 0.02*peri, True)

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def splitBoxes(img):
    rows = np.vsplit(img, 10)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 4)
        for box in cols:
            boxes.append(box)
    return boxes

def splitBoxesRN(img):
    cols = np.hsplit(img, 8)
    boxes = []
    for c in cols:
        rows = np.vsplit(c, 10)
        for box in rows:
            boxes.append(box)
    return boxes

def splitBoxesSN(img):
    cols = np.hsplit(img, 1)
    boxes = []
    for c in cols: 
        rows = np.vsplit(c, 4)
        for box in rows:
            boxes.append(box)
    return boxes

def splitBoxesSC(img):
    cols = np.hsplit(img, 3)
    boxes = []
    for c in cols:
        rows = np.vsplit(c, 10)
        for box in rows:
            boxes.append(box)
    return boxes

def showAnswers(img, myIndex, grading, ans, questions, choices):
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)
    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        if grading[x] == 1:
            myColor = (0, 255, 0)
        else:
            myColor = (0, 0, 255)
            coreectAns = ans[x]
            cv2.circle(img, ((coreectAns*secW) + secW // 2,
                       (x * secH) + secH // 2), 25, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (cX, cY), 25, (myColor), cv2.FILLED)
    return img

def determineGrade(scores):
    if 90 <= scores <= 100:
        return 'A1'
    elif 80 <= scores <= 89:
        return 'A2'
    elif 70 <= scores <= 79:
        return 'B1'
    elif 60 <= scores <= 69:
        return 'B2'
    elif 50 <= scores <= 59:
        return 'C1'
    elif 40 <= scores <= 49:
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
