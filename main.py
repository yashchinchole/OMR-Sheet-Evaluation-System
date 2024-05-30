from ctypes import util
from distutils.errors import PreprocessError
from pickletools import float8
import cv2
from cv2 import FILLED
import numpy as np
import utlis
import os
import time
 
# record start time
start = time.time()
 
############################################
path = "OMR New/PCCOE_OMR_20_4_1.jpg"
widthImg = 600
heightImg = 780
questions_box1 = 10
questions_box2 = 10
Choices = 4
Digit_Count = 10
RN_Digits = 8
SC_Digits = 3
ans1 = [1, 0, 2, 3, 1, 0, 3, 1, 2, 3]
ans2 = [0, 1, 0, 2, 3, 0, 2, 3, 1, 1]
ans = [ans1,ans2]
questions = [questions_box1,questions_box2]
Marks_per_question = 2
Negative_Marking = 0
############################################

def PreProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)
    return imgCanny 

def Find_Contours(img):
    contours, hirerarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -10,(255,165,0),2)
    return (contours, hirerarchy)


def Upper(img,bottom):
    temp = PreProcess(img)
    (contours,hirerarchy) = Find_Contours(temp)
    # cv2.drawContours(img, contours, -10,(255,165,0),2)
    # cv2.imshow('Upper image',img)
    # cv2.waitKey(0)
    rectCon = utlis.rectContours(contours,0)

    RollNumber = utlis.getCornerPoints(rectCon[0])
    SubjectCode = utlis.getCornerPoints(rectCon[4])
    MarksPoints = utlis.getCornerPoints(rectCon[1])
    Percentage = utlis.getCornerPoints(rectCon[2])
    GradeContour = utlis.getCornerPoints(rectCon[3])

    RollNumber = utlis.reorder(RollNumber)
    SubjectCode = utlis.reorder(SubjectCode)
    MarksPoints = utlis.reorder(MarksPoints)
    Percentage = utlis.reorder(Percentage)
    GradeContour = utlis.reorder(GradeContour)

    RN1 = np.float32(RollNumber)
    RN2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg],[widthImg, heightImg]])
    matrixRN = cv2.getPerspectiveTransform(RN1,RN2)
    imgwrapRollNumber = cv2.warpPerspective(img,matrixRN,(widthImg,heightImg))
    # cv2.imshow("Roll Number",imgwrapRollNumber)

    PS1 = np.float32(SubjectCode)
    PS2 = np.float32([[0, 0], [120, 0], [0, 300],[120, 300]])
    matrixPS = cv2.getPerspectiveTransform(PS1,PS2)
    imgwrapSubjectCode = cv2.warpPerspective(img,matrixPS,(120,300))
    # cv2.imshow("Paper Set",imgwrapSubjectCode)

    # Apply Threshhold On Roll Number Box
    RollNumberGray = cv2.cvtColor(imgwrapRollNumber,cv2.COLOR_BGR2GRAY)
    imgThreshRollNumber = cv2.threshold(RollNumberGray,170,255,cv2.THRESH_BINARY_INV)[1]

    # Split boxes
    boxes = utlis.splitBoxesRN(imgThreshRollNumber)
    # cv2.imshow("Test Box 1",boxes[16])

    print("###########################################################\n")
    # Getting Non-Zero pixel values of Roll Number box
    myPixelval = np.zeros((RN_Digits, Digit_Count))
    countDigit = 0
    countRN = 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelval[countRN][countDigit] = totalPixels
        countDigit +=1

        if (countDigit == Digit_Count): countRN +=1; countDigit = 0
    # print(myPixelval)

    # Finding index values Of the marking
    myindex = []
    for x in range (0,RN_Digits):
        arrRN = myPixelval[x]
        # print("arrRN", arrRN)
        myIndexval = np.where(arrRN == np.amax(arrRN))
        # print(myIndexval[0])
        myindex.append(myIndexval[0][0])
    print("\nRoll Number: ",myindex)
    
    # Apply Threshhold On Subject Code Box
    SubjectCodeGray = cv2.cvtColor(imgwrapSubjectCode,cv2.COLOR_BGR2GRAY)
    imgThreshSubjectCode = cv2.threshold(SubjectCodeGray,170,255,cv2.THRESH_BINARY_INV)[1]

    # Split boxes
    boxes = utlis.splitBoxesSC(imgThreshSubjectCode)
    # cv2.imshow("Test Box 1",boxes[16])

    # Getting Non-Zero pixel values of Set Number box
    myPixelval = np.zeros((SC_Digits, Digit_Count))
    countDigit = 0
    countSC = 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelval[countSC][countDigit] = totalPixels
        countDigit +=1

        if (countDigit == Digit_Count): countSC +=1; countDigit = 0
    
    # print(myPixelval)

    # Finding index values Of the marking
    myindexnew = []
    for x in range (0,SC_Digits):
        arrSC = myPixelval[x]
        # print("arrSC", arrSC)
        myIndexvalnew = np.where(arrSC == np.amax(arrSC))
        # print(myIndexval[0])
        myindexnew.append(myIndexvalnew[0][0])
    print("\nSubject Code: ",myindexnew)

    bottom,marks_Obtained,score,Grade,marks = Lower(bottom,SubjectCode,RollNumber)

    # After processing lower
    ptG1 = np.float32(MarksPoints)
    ptG2 = np.float32([[0, 0], [420, 0], [0, 130],[420, 130]])
    matrixM = cv2.getPerspectiveTransform(ptG1,ptG2)
    imgMarksDisplay = cv2.warpPerspective(img,matrixM,(420,130))
    # cv2.imshow("Marks Obtained",imgMarksDisplay)

    ptP1 = np.float32(Percentage)
    ptP2 = np.float32([[0, 0], [120, 0], [0, 130],[120, 130]])
    matrixP = cv2.getPerspectiveTransform(ptP1,ptP2)
    imgPercentage = cv2.warpPerspective(img,matrixP,(120,130))
    # cv2.imshow("Percentage Got",imgPercentage)
    
    GRA1 = np.float32(GradeContour)
    GRA2 = np.float32([[0, 0], [120, 0], [0, 130],[120, 130]])
    matrixG = cv2.getPerspectiveTransform(GRA1,GRA2)
    imgGrade = cv2.warpPerspective(img,matrixG,(120,130))
    # cv2.imshow("Grade",imgGrade)
    final_image = np.concatenate((img,bottom),axis=0)
    h, w, channels = final_image.shape
     # Displaying Marks
    imgRawMarks = np.zeros_like(imgMarksDisplay)
    cv2.putText(imgRawMarks, str(int(marks_Obtained)) + "/" + str(int(marks)), (160,90),cv2.FONT_HERSHEY_COMPLEX,2.4,(0,255,25),5,FILLED)
    # cv2.imshow("Marks", imgRawMarks)
    invMatrixM = cv2.getPerspectiveTransform(ptG2,ptG1)
    imgInvMarksDisplay = cv2.warpPerspective(imgRawMarks, invMatrixM, (w,h))
    # cv2.imshow("Obtained Marks",imgInvMarksDisplay)

    # Displaying Percentage
    imgRawPercentage = np.zeros_like(imgPercentage)
    cv2.putText(imgRawPercentage, str(int(score)) + "%", (5,90),cv2.FONT_HERSHEY_COMPLEX,1.4,(0,255,255),4,FILLED)
    # cv2.imshow("Percentage", imgRawPercentage)
    invMatrixP = cv2.getPerspectiveTransform(ptP2,ptP1)
    imgInvPercentagDisplay = cv2.warpPerspective(imgRawPercentage, invMatrixP, (w,h))
    # cv2.imshow("Percentage",imgInvPercentagDisplay)

    # Displaying Grade
    imgRawGrade = np.zeros_like(imgGrade)
    cv2.putText(imgRawGrade, str(Grade), (15,100),cv2.FONT_HERSHEY_COMPLEX,2.5,(255, 0, 255),7,FILLED)
    # cv2.imshow("Grade", imgRawGrade)
    invMatrixG = cv2.getPerspectiveTransform(GRA2,GRA1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (w,h))
    # cv2.imshow("Grade",imgInvGradeDisplay)
    # Adding all the images to the Final Image
    final_image = cv2.addWeighted(final_image,0.6,imgInvMarksDisplay,0.3,0)
    final_image = cv2.addWeighted(final_image,1, imgInvGradeDisplay,0.6, 0)
    final_image = cv2.addWeighted(final_image,1, imgInvPercentagDisplay,1, 0)
    return final_image



def Lower(img,subject_code,roll_no):
    temp = PreProcess(img)
    (contours,hirerarchy) = Find_Contours(temp)
    # cv2.drawContours(img, contours, -10,(255,165,0),2)
    # cv2.imshow('lower image',img)
    # cv2.waitKey(0)
    rectContors= utlis.rectContours(contours,15000)
    # cv2.drawContours(img, contours, -10,(255,165,0),2)
    
    # for i in rectContors:
    #     print(utlis.reorder(utlis.getCornerPoints(i))[0][0][0],sep=" ",end="\n")

    rectContors = sorted(rectContors,key = lambda x : utlis.reorder(utlis.getCornerPoints(x))[0][0][0])

    points = []
    matrices = []
    allgrades = []
    warp_boxes = []
    marked_answer = []

    # for i in rectContors:
    #     print(utlis.reorder(utlis.getCornerPoints(i))[0][0][0],sep=" ",end="\n")
    

    for index,value in enumerate(rectContors):
        firstbox = utlis.getCornerPoints(value)
        firstbox = utlis.reorder(firstbox)
        
        pt1 = np.float32(firstbox)
        pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg],[widthImg, heightImg]])
        matrix1 = cv2.getPerspectiveTransform(pt1,pt2)
        imgwrapColored1 = cv2.warpPerspective(img,matrix1,(widthImg,heightImg))
        # cv2.imshow("Marking Area1",imgwrapColored1)

        imgwrapGray = cv2.cvtColor(imgwrapColored1,cv2.COLOR_BGR2GRAY)
        imgThresh1 = cv2.threshold(imgwrapGray,170,255,cv2.THRESH_BINARY_INV)[1]

        imgwrapGray = cv2.cvtColor(imgwrapColored1,cv2.COLOR_BGR2GRAY)
        imgThresh1 = cv2.threshold(imgwrapGray,170,255,cv2.THRESH_BINARY_INV)[1]

        # Split boxes
        boxes = utlis.splitBoxes(imgThresh1)
        # cv2.imshow("Test Box 1",boxes[0])

        # print(cv2.countNonZero(boxes[0]),cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]),cv2.countNonZero(boxes[3]))
        # print(cv2.countNonZero(boxes[4]),cv2.countNonZero(boxes[5]),cv2.countNonZero(boxes[6]),cv2.countNonZero(boxes[7]))

        # Getting Non-Zero pixel values of each box
        myPixelval = np.zeros((questions_box1,Choices))
        countC = 0
        countR = 0

        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelval[countR][countC] = totalPixels
            countC +=1

            if (countC == Choices): countR +=1; countC = 0
        print(myPixelval)

        print("\n####################################################\n")
        # Finding index values Of the marking
        myindex = []
        for x in range (0,questions[index]):
            arr = myPixelval[x]
            # print("arr1", arr)
            myIndexval = np.where(arr == np.amax(arr))
            # print(myIndexval[0])
            myindex.append(myIndexval[0][0])
        print(f"Answer Index Of Box {index+1}\n",myindex)

        # Grading Of Box 1
        grading = []
        for x in  range(0, questions[index]):
            if ans[index][x] == myindex[x]:
                grading.append(1)
            else: grading.append(0)

        allgrades.append(grading)
        points.append([pt1,pt2])
        matrices.append(matrix1)
        warp_boxes.append(imgwrapColored1)
        marked_answer.append(myindex)
        # print("\nGrading Of Box 1: ",grading)

    # Total Number Of Questions
    Tot_questions = sum(questions)
    print("\nTotal Number Of Questions: ",Tot_questions)

    # Final Marks
    Correctly_Marked =  sum([sum(i) for i in allgrades])
    print("\nCorrectly Marked: ", Correctly_Marked)

    # Wrong Answers
    Wrongly_Marked =  (Tot_questions - Correctly_Marked)
    print("\nWrongly Marked: ",  Wrongly_Marked)

    # Total Marks
    marks = Marks_per_question * Tot_questions
    print("\nTotal Marks: ", marks)

    # Final Marks
    marks_Obtained = Marks_per_question * Correctly_Marked
    print("\nTotal Marks Obtained: ", marks_Obtained)

    # Percentage
    score = (Correctly_Marked / Tot_questions) * 100 
    print("\nPercentage: ",score,"%")

    # Grade
    Grade = utlis.determine_grade(score)
    print("\nGrade: ", Grade)
    
    # Displaying Answers For Box 1
    h, w, channels = img.shape
    for index,value in enumerate(warp_boxes):
        imgResult1 = value.copy()
        imgResult1 = utlis.showAnswers(imgResult1,marked_answer[index],allgrades[index],ans[index],questions[index],Choices)

        imgRawDrawing1 = np.zeros_like(value)
        imgRawDrawing1 = utlis.showAnswers(imgRawDrawing1,marked_answer[index],allgrades[index],ans[index],questions[index],Choices)

        invmatrix1 = cv2.getPerspectiveTransform(points[index][1],points[index][0])
        imgInvwrap1 = cv2.warpPerspective(imgRawDrawing1,invmatrix1,(w,h))

        # cv2.imshow(f"Box {index+1}", imgInvwrap1)
        # Adding all the images to the Final Image
        img = cv2.addWeighted(img,1,imgInvwrap1,1,0)
    # cv2.imshow(f"Final", img)
    return [img,marks_Obtained,score,Grade,marks]



#--------------------------------------------------#
img = cv2.imread(path)
img = cv2.resize(img, (widthImg, heightImg))
# h, w, channels = img.shape
# half2 = h//2
  
# top = img[:half2, :]
# bottom = img[half2:, :]
imgContours = img.copy()
imgBiggestContours = img.copy()
imgFinal = img.copy()
# imgBlank = np.zeros_like(img) 
#--------------------------------------------------#

# Preprocessing
img1 = PreProcess(img)

# Finding Contours
(contours, hirerarchy) = Find_Contours(img1)
# Find rectangles
rectCon = utlis.rectContours(contours,200000)
biggestContour1 = utlis.getCornerPoints(rectCon[0])


# biggestContour2 = utlis.getCornerPoints(rectCon[1])

# RollNumber = utlis.getCornerPoints(rectCon[2])
# SubjectCode = utlis.getCornerPoints(rectCon[5])

# MarksPoints = utlis.getCornerPoints(rectCon[3])
# Percentage = utlis.getCornerPoints(rectCon[4])
# GradeContour = utlis.getCornerPoints(rectCon[6])

# print(biggestContour)
if biggestContour1.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour1, -1, (0, 255, 0), 20)

    biggestContour1 = utlis.reorder(biggestContour1)

    pt1 = np.float32(biggestContour1)
    pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg],[widthImg, heightImg]])
    matrix1 = cv2.getPerspectiveTransform(pt1,pt2)
    imgwrap = cv2.warpPerspective(img,matrix1,(widthImg,heightImg))
    # cv2.imshow("Biggest Rectangle",imgwrap)

h, w, channels = imgwrap.shape
cut = (h * 60)// 100

top = imgwrap[:cut, :]
bottom = imgwrap[cut:, :]

final_image = Upper(top,bottom)
cv2.imshow('Final',final_image)

cv2.waitKey(0)