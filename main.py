import streamlit as st
from PIL import Image
import numpy as np
import cv2
import functions
import util
import style

widthImg = 600
heightImg = 780
marksPerQuestion = 2
choices = 4
questionsBox1 = 10
questionsBox2 = 10
questions = [questionsBox1, questionsBox2]
ans1 = [0, 0, 2, 3, 1, 0, 3, 1, 2, 3]
ans2 = [0, 1, 0, 2, 3, 0, 2, 3, 1, 1]
ans = [ans1, ans2]

def find_marks(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
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
        return finalImage
    else:
        return None

st.set_page_config(page_title="OMR Sheet Evaluation System", page_icon="📝", layout="centered", initial_sidebar_state="expanded")

style.apply_styling()

st.title("📝 OMR Sheet Evaluation System")
st.write("Upload your OMR sheet image below and click the **Calculate Marks** button to get your results.")

uploaded_file = st.file_uploader("Choose an OMR image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded OMR Sheet', use_column_width=True, width=300)

    if st.button('Calculate Marks'):
        with st.spinner('Processing...'):
            final_image = find_marks(image)
            if final_image is not None:
                st.image(final_image, caption='Graded OMR Sheet', use_column_width=True)
            else:
                st.error("Could not find the OMR sheet in the image. Please upload a clearer image.")
else:
    st.info("Please upload an image of your OMR sheet to get started.")
