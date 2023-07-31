import cv2 as cv
import streamlit as st
import numpy as np
import pandas as pd


class MainCV:

    def __init__(self, image=None):
        self.img = image

        if not image:
            self.img = cv.imread("OpenCVBasicFunctions/Photos/room_test.jpg")

        self.default_img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)

        self.show_basic_functions()
        self.bitwise()
        self.masking()
        self.split_merge()
        self.transformations()
        self.contours()
        self.threshold()
        self.histogram()
        self.gradients()
        self.smoothing_with_blur()
        self.color_spaces()
        self.rescale()
        self.draw()
        self.random_info()

    def show_basic_functions(self):
        with st.expander("Basic functions"):
            img = self.default_img.copy()
            st.image(img)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            st.write("Gray => cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)")
            st.image(gray)

            blur_coef = st.slider('blur_coef', min_value=1, max_value=15, value=7, step=2)
            blur = cv.GaussianBlur(img, (blur_coef, blur_coef), cv.BORDER_DEFAULT)
            st.write("Blur => cv.GaussianBlur(self.img, (blur_coef, blur_coef), cv.BORDER_DEFAULT)")

            canny_coef = st.slider('canny_coef', min_value=0, max_value=200, value=(125, 175))
            canny = cv.Canny(blur, canny_coef[0], canny_coef[1])
            st.write("Canny => cv.Canny(blur_img, canny_coef, canny_coef)")
            st.image(canny)

            dilated_coef = st.slider('dilated_coef', min_value=1, max_value=15, value=7, step=1)
            dilated_iter = st.slider('dilated_coef', min_value=1, max_value=15, value=3, step=1)
            dilated = cv.dilate(canny, (dilated_coef, dilated_coef), iterations=dilated_iter)
            st.write("Dilated => cv.dilate(canny, (dilated_coef, dilated_coef), iterations=dilated_iter)")
            st.image(dilated)

            eroded_coef = st.slider('eroded_coef', min_value=1, max_value=15, value=7, step=1)
            eroded_iter = st.slider('eroded_coef', min_value=1, max_value=15, value=3, step=1)
            eroded = cv.erode(dilated, (eroded_coef, eroded_coef), iterations=eroded_iter)
            st.write("Eroded => cv.erode(dilated, (eroded_coef, eroded_coef), iterations=eroded_coef)")
            st.image(eroded)

    def bitwise(self):
        with st.expander("Bitwise"):
            blank = np.zeros((400, 400), dtype="uint8")

            rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
            st.write("draw rectangle")
            st.image(rectangle)

            circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
            st.write("draw circle")
            st.image(circle)

            st.write("cv.bitwise_and(rectangle, circle)")
            bitwise_and = cv.bitwise_and(rectangle, circle)
            st.image(bitwise_and)

            bitwise_or = cv.bitwise_or(rectangle, circle)
            st.write("cv.bitwise_or(rectangle, circle)")
            st.image(bitwise_or)

            bitwise_xor = cv.bitwise_xor(rectangle, circle)
            st.write("cv.bitwise_xor(rectangle, circle)")
            st.image(bitwise_xor)

            bitwise_not = cv.bitwise_not(rectangle)
            st.write("cv.bitwise_not(rectangle)")
            st.image(bitwise_not)

    def contours(self):
        with st.expander("Contours"):
            img = self.default_img.copy()
            st.image(img)

            blank = np.zeros(img.shape, dtype="uint8")
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            st.write("gray => cv.cvtColor(img, cv.COLOR_BGR2GRAY)")
            st.image(gray)

            blur_coef = st.slider('blur_coef', min_value=1, max_value=15, value=5, step=2)
            blur = cv.GaussianBlur(gray, (blur_coef, blur_coef), cv.BORDER_DEFAULT)
            st.write("Blur => cv.GaussianBlur(self.img, (blur_coef, blur_coef), cv.BORDER_DEFAULT)")
            st.image(blur)

            canny_coef = st.slider('canny_coef_', min_value=0, max_value=200, value=(125, 175))
            canny = cv.Canny(blur, canny_coef[0], canny_coef[1])
            st.write("Canny => cv.Canny(blur_img, canny_coef, canny_coef)")
            st.image(canny)

            st.write("contours")
            contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
            st.image(blank)

    def draw(self):
        with st.expander("Draw"):
            blank = np.zeros((500, 500, 3), dtype="uint8")
            blank[200:300, 300:400] = 0, 0, 255
            cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=2)
            cv.circle(blank, (250, 250), 40, (0, 0, 255), thickness=-1)
            cv.line(blank, (0, 0), (250, 250), (255, 0, 255), thickness=3)
            cv.putText(blank, "Test", (325, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)

            st.image(blank)

    def gradients(self):
        with st.expander("Gradients"):
            img = self.default_img.copy()
            st.image(img)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            st.write("gray => cv.cvtColor(img, cv.COLOR_BGR2GRAY)")
            st.image(gray)

            # Laplacian

            lap = cv.Laplacian(gray, cv.CV_64F)
            lap = np.uint8(np.absolute(lap))
            st.write("Laplacian => cv.Laplacian(gray, cv.CV_64F) => np.uint8(np.absolute(lap))")
            st.image(lap)

            # Sobel
            sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0)
            sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1)

            st.write("sobel_x => cv.Sobel(gray, cv.CV_64F, 1, 0)")
            st.write("sobel_y => cv.Sobel(gray, cv.CV_64F, 0, 1)")

            combine_sobel = cv.bitwise_or(sobel_x, sobel_y)
            st.write("combine_sobel => cv.bitwise_or(sobel_x, sobel_y)")
            st.image(combine_sobel)

            canny = cv.Canny(gray, 150, 175)
            st.write("canny => cv.Canny(gray, 150, 175)")
            st.image(canny)

    def histogram(self):
        with st.expander("Histogram"):
            img = self.default_img.copy()
            st.image(img)

            blank = np.zeros(img.shape[:2], dtype="uint8")

            mask = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)
            masked = cv.bitwise_and(img, img, mask=mask)

            colors = ("blue", "green", "red")

            histograms = pd.DataFrame()

            for i, col in enumerate(colors):
                st.write(f"{col}")
                hist = cv.calcHist([img], [i], mask, [256], [0, 256])
                histograms[col] = hist.flatten()

            st.line_chart(pd.DataFrame(histograms))

    def masking(self):
        with st.expander("Masking"):
            img = self.default_img.copy()
            st.image(img)

            blank = np.zeros(img.shape[:2], dtype="uint8")

            mask = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)
            st.write("mask")
            st.image(mask)

            st.write("masked image")
            masked = cv.bitwise_and(img, img, mask=mask)
            st.image(masked)

    def threshold(self):
        with st.expander("Threshold"):
            img = self.default_img.copy()
            st.image(img)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            st.write("gray => cv.cvtColor(img, cv.COLOR_BGR2GRAY)")
            st.image(gray)

            thresh_coef_1 = st.slider('thresh_coef_1', min_value=0, max_value=255, value=(100, 255))
            threshold, thresh = cv.threshold(gray, thresh_coef_1[0], thresh_coef_1[1], cv.THRESH_BINARY)
            st.write("simplified threshold => cv.threshold(gray, thresh_coef_1[0], thresh_coef_1[1], cv.THRESH_BINARY)")
            st.write(f"{thresh_coef_1[0]} / {thresh_coef_1[1]}")
            st.image(thresh)

            thresh_coef_2 = st.slider('thresh_coef_2', min_value=0, max_value=255, value=(150, 255))
            threshold_, thresh_inv = cv.threshold(gray, thresh_coef_2[0], thresh_coef_2[1], cv.THRESH_BINARY_INV)
            st.write("threshold inv => cv.threshold(gray, thresh_coef_2[0], thresh_coef_2[1], cv.THRESH_BINARY_INV)")
            st.image(thresh_inv)

            box_size = st.slider('box_size', min_value=3, max_value=49, value=11, step=2)
            constant = st.slider('box_size', min_value=1, max_value=30, value=3)
            adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, box_size, constant)
            st.write("adaptive thresholding")
            st.image(adaptive_thresh)

    def split_merge(self):
        with st.expander("Split-merge"):
            img = self.default_img.copy()
            st.image(img)

            blank = np.zeros(img.shape[:2], dtype="uint8")

            b, g, r = cv.split(img)

            st.write("b channel")
            st.image(b)

            merged = cv.merge([b, g, r])
            st.write("merged image => cv.merge([b, g, r])")
            st.image(merged)

            color_channel = st.radio(
                "Pick color",
                ('blue', 'green', 'red'))

            if color_channel == "blue":
                st.image(cv.merge([blank, blank, b]))
            elif color_channel == "green":
                st.image(cv.merge([blank, g, blank]))
            else:
                st.image(cv.merge([r, blank, blank]))

    def transformations(self):
        with st.expander("Transformations"):
            img = self.default_img.copy()
            st.image(img)

            st.write("translation => np.float32([[1, 0, translate_coef_x], [0, 1, translate_coef_y]]) =>")
            st.write("=>(img.shape[1], img.shape[0])")
            translate_coef_x = st.slider('translate_coef_x', min_value=-200, max_value=200, value=100)
            translate_coef_y = st.slider('translate_coef_y', min_value=-200, max_value=200, value=100)
            translation_matrix = np.float32([[1, 0, translate_coef_x], [0, 1, translate_coef_y]])
            dimensions = (img.shape[1], img.shape[0])
            st.image(cv.warpAffine(img, translation_matrix, dimensions))

            st.write("rotate => cv.warpAffine(img, rotation_matrix, dimensions)")
            height, width = img.shape[:2]
            rotation_point = (width // 2, height // 2)
            rotation_angle = st.slider('rotation_angle', min_value=-360, max_value=360, value=-45)
            rotation_matrix = cv.getRotationMatrix2D(rotation_point, rotation_angle, 1.0)
            dimensions = (width, height)
            st.image(cv.warpAffine(img, rotation_matrix, dimensions))

            st.write("resize => cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)")
            width = st.slider('width', min_value=0, max_value=1000, value=300)
            height = st.slider('height', min_value=0, max_value=1000, value=300)
            st.image(cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC))

            st.write("flip => cv.flip(img, flip_val)")
            flip_val = st.slider('flip_val', min_value=-1, max_value=1, value=1)
            st.image(cv.flip(img, flip_val))

            st.write("crop => ")
            width_part = st.slider('width_part', min_value=0, max_value=img.shape[1], value=(200, 400))
            height_part = st.slider('height_part', min_value=0, max_value=img.shape[0], value=(300, 400))
            st.image(img[height_part[0]:height_part[1], width_part[0]:width_part[1]])

    def color_spaces(self):
        with st.expander("Color spaces"):
            img = self.default_img.copy()
            st.image(img)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            st.write("gray => cv.cvtColor(img, cv.COLOR_BGR2GRAY)")
            st.image(gray)

            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            st.write("HSV => cv.cvtColor(img, cv.COLOR_BGR2HSV)")
            st.image(hsv)

            lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            st.write("LAB => cv.cvtColor(img, cv.COLOR_BGR2LAB)")
            st.image(lab)

            hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            st.write("HSV --> BGR => cv.cvtColor(hsv, cv.COLOR_HSV2BGR)")
            st.image(hsv_bgr)

    def smoothing_with_blur(self):
        with st.expander("Smoothing with blur"):
            img = self.default_img.copy()
            st.image(img)

            av_blur_coef = st.slider('av_blur_coef', min_value=1, max_value=15, value=3, step=2)
            st.write("Average blur => cv.blur(img, (av_blur_coef, av_blur_coef)))")
            average_blur = cv.blur(img, (av_blur_coef, av_blur_coef))
            st.image(average_blur)

            gaus_blur_coef = st.slider('gaus_blur_coef', min_value=1, max_value=15, value=3, step=2)
            st.write("Gaussian blur => cv.GaussianBlur(img, (gaus_blur_coef, gaus_blur_coef), 0)")
            gaussian_blur = cv.GaussianBlur(img, (gaus_blur_coef, gaus_blur_coef), 0)
            st.image(gaussian_blur)

            med_blur_coef = st.slider('med_blur_coef', min_value=1, max_value=15, value=3, step=2)
            st.write("Median blur => cv.medianBlur(img, med_blur_coef)")
            median_blur = cv.medianBlur(img, med_blur_coef)
            st.image(median_blur)

            d_coef = st.slider('d', min_value=0, max_value=100, value=10)
            sigma_color_coef = st.slider('sigma_color_coef', min_value=0, max_value=100, value=35)
            sigma_space_coef = st.slider('sigma_color_coef', min_value=0, max_value=100, value=25)
            st.write("Bilateral blur => cv.bilateralFilter(img, d=d_coef, sigmaColor=sigma_color_coef, sigmaSpace=sigma_space_coef)")
            bilateral_blur = cv.bilateralFilter(img, d=d_coef, sigmaColor=sigma_color_coef, sigmaSpace=sigma_space_coef)
            st.image(bilateral_blur)

    def rescale(self):
        with st.expander("Rescale"):
            img = self.default_img.copy()
            st.image(img)

            scale = (st.slider('scale', min_value=1, max_value=100, value=50) / 100)
            width = int(img.shape[1] * scale)
            height = int(img.shape[0] * scale)
            dimensions = (width, height)
            scaled_img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
            st.write(f"Original image size = {img.shape[:2]}")
            st.write(f"Sceled image size = {scaled_img.shape[:2]}")
            st.write("Rescale => cv.resize(img, dimensions, interpolation=cv.INTER_AREA)")
            st.image(scaled_img)

    @staticmethod
    def random_info():
        with st.expander("Random info"):
            st.write("pip install opencv-python==4.5.4.60")

