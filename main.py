import os
import cv2
import numpy as np
import util
import functions

currentDir = os.getcwd()
imageDir = os.path.join(currentDir, "assets/Sample_OMR")
imagePath = os.path.join(imageDir, "OMR_20_4.jpg")
img = cv2.imread(imagePath)
widthImg = 600
heightImg = 780

marksPerQuestion = 2
choices = 4
numQuestions = 20
questionsBox1 = 10
questionsBox2 = 10
questions = [questionsBox1, questionsBox2]

ans1 = [1, 0, 2, 3, 1, 0, 3, 1, 2, 3]
ans2 = [0, 1, 0, 2, 3, 0, 2, 3, 1, 1]
ans = [ans1, ans2]

def findMarks():
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgBiggestContours = img.copy()

    img1 = functions.preProcess(img)

    contours, hierarchy = functions.findContours(img1, imgContours)

    rectCon = util.rectContours(contours, 200000)
    biggestContour1 = util.getCornerPoints(rectCon[0])

    if biggestContour1.size != 0:
        cv2.drawContours(imgBiggestContours, biggestContour1, -1, (0, 255, 0), 20)
        biggestContour1 = util.reorder(biggestContour1)

        pt1 = np.float32(biggestContour1)
        pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix1 = cv2.getPerspectiveTransform(pt1, pt2)
        imgWrap = cv2.warpPerspective(img, matrix1, (widthImg, heightImg))

        h, w, channels = imgWrap.shape
        cut = (h * 60) // 100

        top = imgWrap[:cut, :]
        bottom = imgWrap[cut:, :]

        finalImage = functions.upper(top, bottom, imgContours, questionsBox1, choices, questions, ans, marksPerQuestion)
        cv2.imshow('Final Image', finalImage)
        cv2.waitKey(0)

findMarks()
