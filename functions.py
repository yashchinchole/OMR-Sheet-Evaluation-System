import cv2
from cv2 import FILLED
import numpy as np
import util

widthImg = 600
heightImg = 780
Digit_Count = 10
RN_Digits = 8
SC_Digits = 3

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)
    return imgCanny

def findContours(img, imgContours):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -10, (255, 165, 0), 2)
    return contours, hierarchy

def upper(img, bottom, imgContours, questions_box_length, Choices, questions, ans, Marks_per_question):
    temp = preProcess(img)
    contours, hierarchy = findContours(temp, imgContours)
    rectCon = util.rectContours(contours, 0)

    RollNumber = util.getCornerPoints(rectCon[0])
    SubjectCode = util.getCornerPoints(rectCon[4])
    MarksPoints = util.getCornerPoints(rectCon[1])
    Percentage = util.getCornerPoints(rectCon[2])
    GradeContour = util.getCornerPoints(rectCon[3])

    RollNumber = util.reorder(RollNumber)
    SubjectCode = util.reorder(SubjectCode)
    MarksPoints = util.reorder(MarksPoints)
    Percentage = util.reorder(Percentage)
    GradeContour = util.reorder(GradeContour)

    RN1 = np.float32(RollNumber)
    RN2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrixRN = cv2.getPerspectiveTransform(RN1, RN2)
    imgwrapRollNumber = cv2.warpPerspective(img, matrixRN, (widthImg, heightImg))

    PS1 = np.float32(SubjectCode)
    PS2 = np.float32([[0, 0], [120, 0], [0, 300], [120, 300]])
    matrixPS = cv2.getPerspectiveTransform(PS1, PS2)
    imgwrapSubjectCode = cv2.warpPerspective(img, matrixPS, (120, 300))

    RollNumberGray = cv2.cvtColor(imgwrapRollNumber, cv2.COLOR_BGR2GRAY)
    imgThreshRollNumber = cv2.threshold(RollNumberGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    boxes = util.splitBoxesRN(imgThreshRollNumber)

    myPixelval = np.zeros((RN_Digits, Digit_Count))
    countDigit = 0
    countRN = 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelval[countRN][countDigit] = totalPixels
        countDigit += 1
        if countDigit == Digit_Count:
            countRN += 1
            countDigit = 0

    myindex = []
    for x in range(RN_Digits):
        arrRN = myPixelval[x]
        myIndexval = np.where(arrRN == np.amax(arrRN))
        myindex.append(myIndexval[0][0])
    print("\nRoll Number: ", myindex)

    SubjectCodeGray = cv2.cvtColor(imgwrapSubjectCode, cv2.COLOR_BGR2GRAY)
    imgThreshSubjectCode = cv2.threshold(SubjectCodeGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    boxes = util.splitBoxesSC(imgThreshSubjectCode)

    myPixelval = np.zeros((SC_Digits, Digit_Count))
    countDigit = 0
    countSC = 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelval[countSC][countDigit] = totalPixels
        countDigit += 1
        if countDigit == Digit_Count:
            countSC += 1
            countDigit = 0

    myindexnew = []
    for x in range(SC_Digits):
        arrSC = myPixelval[x]
        myIndexvalnew = np.where(arrSC == np.amax(arrSC))
        myindexnew.append(myIndexvalnew[0][0])
    print("\nSubject Code: ", myindexnew)

    bottom, marks_Obtained, score, Grade, marks = lower(bottom, questions_box_length, Choices, questions, ans, Marks_per_question, SubjectCode, RollNumber, imgContours)

    ptG1 = np.float32(MarksPoints)
    ptG2 = np.float32([[0, 0], [420, 0], [0, 130], [420, 130]])
    matrixM = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgMarksDisplay = cv2.warpPerspective(img, matrixM, (420, 130))

    ptP1 = np.float32(Percentage)
    ptP2 = np.float32([[0, 0], [120, 0], [0, 130], [120, 130]])
    matrixP = cv2.getPerspectiveTransform(ptP1, ptP2)
    imgPercentage = cv2.warpPerspective(img, matrixP, (120, 130))

    GRA1 = np.float32(GradeContour)
    GRA2 = np.float32([[0, 0], [120, 0], [0, 130], [120, 130]])
    matrixG = cv2.getPerspectiveTransform(GRA1, GRA2)
    imgGrade = cv2.warpPerspective(img, matrixG, (120, 130))

    final_image = np.concatenate((img, bottom), axis=0)
    h, w, channels = final_image.shape

    imgRawMarks = np.zeros_like(imgMarksDisplay)
    cv2.putText(imgRawMarks, str(int(marks_Obtained)) + "/" + str(int(marks)), (160, 90), cv2.FONT_HERSHEY_COMPLEX, 2.4, (255, 255, 0), 5, FILLED)
    invMatrixM = cv2.getPerspectiveTransform(ptG2, ptG1)
    imgInvMarksDisplay = cv2.warpPerspective(imgRawMarks, invMatrixM, (w, h))

    imgRawPercentage = np.zeros_like(imgPercentage)
    cv2.putText(imgRawPercentage, str(int(score)) + "%", (5, 90), cv2.FONT_HERSHEY_COMPLEX, 1.4, (0, 255, 255), 4, FILLED)
    invMatrixP = cv2.getPerspectiveTransform(ptP2, ptP1)
    imgInvPercentagDisplay = cv2.warpPerspective(imgRawPercentage, invMatrixP, (w, h))

    imgRawGrade = np.zeros_like(imgGrade)
    cv2.putText(imgRawGrade, str(Grade), (15, 100), cv2.FONT_HERSHEY_COMPLEX, 2.5, (255, 0, 255), 7, FILLED)
    invMatrixG = cv2.getPerspectiveTransform(GRA2, GRA1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (w, h))

    final_image = cv2.addWeighted(final_image, 0.6, imgInvMarksDisplay, 0.3, 0)
    final_image = cv2.addWeighted(final_image, 1, imgInvGradeDisplay, 0.6, 0)
    final_image = cv2.addWeighted(final_image, 1, imgInvPercentagDisplay, 1, 0)
    return final_image

def lower(img, questions_box_length, Choices, questions, ans, Marks_per_question, subject_code, roll_no, imgContours):
    temp = preProcess(img)
    contours, hierarchy = findContours(temp, imgContours)
    rectContors = util.rectContours(contours, 15000)
    rectContors = sorted(rectContors, key=lambda x: util.reorder(util.getCornerPoints(x))[0][0][0])

    points = []
    matrices = []
    allgrades = []
    warp_boxes = []
    marked_answer = []

    for index, value in enumerate(rectContors):
        firstbox = util.getCornerPoints(value)
        firstbox = util.reorder(firstbox)

        pt1 = np.float32(firstbox)
        pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix1 = cv2.getPerspectiveTransform(pt1, pt2)
        imgwrapColored1 = cv2.warpPerspective(img, matrix1, (widthImg, heightImg))

        imgwrapGray = cv2.cvtColor(imgwrapColored1, cv2.COLOR_BGR2GRAY)
        imgThresh1 = cv2.threshold(imgwrapGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

        boxes = util.splitBoxes(imgThresh1)

        myPixelval = np.zeros((10, Choices))
        countC = 0
        countR = 0

        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelval[countR][countC] = totalPixels
            countC += 1
            if countC == Choices:
                countR += 1
                countC = 0

        myindex = []
        for x in range(questions[index]):
            arr = myPixelval[x]
            myIndexval = np.where(arr == np.amax(arr))
            myindex.append(myIndexval[0][0])
        print(f"Answer Index Of Box {index + 1}\n", myindex)

        grading = []
        for x in range(questions[index]):
            if ans[index][x] == myindex[x]:
                grading.append(1)
            else:
                grading.append(0)

        allgrades.append(grading)
        points.append([pt1, pt2])
        matrices.append(matrix1)
        warp_boxes.append(imgwrapColored1)
        marked_answer.append(myindex)

    Tot_questions = sum(questions)
    print("\nTotal Number Of Questions: ", Tot_questions)

    Correctly_Marked = sum([sum(i) for i in allgrades])
    print("\nCorrectly Marked: ", Correctly_Marked)

    Wrongly_Marked = Tot_questions - Correctly_Marked
    print("\nWrongly Marked: ", Wrongly_Marked)

    marks = Marks_per_question * Tot_questions
    print("\nTotal Marks: ", marks)

    marks_Obtained = Marks_per_question * Correctly_Marked
    print("\nTotal Marks Obtained: ", marks_Obtained)

    score = (Correctly_Marked / Tot_questions) * 100
    print("\nPercentage: ", score, "%")

    Grade = util.determineGrade(score)
    print("\nGrade: ", Grade)

    h, w, channels = img.shape
    for index, value in enumerate(warp_boxes):
        imgResult1 = value.copy()
        imgResult1 = util.showAnswers(imgResult1, marked_answer[index], allgrades[index], ans[index], questions[index], Choices)

        imgRawDrawing1 = np.zeros_like(value)
        imgRawDrawing1 = util.showAnswers(imgRawDrawing1, marked_answer[index], allgrades[index], ans[index], questions[index], Choices)

        invmatrix1 = cv2.getPerspectiveTransform(points[index][1], points[index][0])
        imgInvwrap1 = cv2.warpPerspective(imgRawDrawing1, invmatrix1, (w, h))

        img = cv2.addWeighted(img, 1, imgInvwrap1, 1, 0)

    return [img, marks_Obtained, score, Grade, marks]
