import cv2
import cv2 as cv
import numpy as np


def pre_process(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(img_gray, (7, 7))
    ret, thresh = cv.threshold(img_blur, 165, 255, cv.THRESH_BINARY)

    return thresh


def find_contours(img, draw_image):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    max_area = 10_000
    contour = []
    approx_poly = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

        if abs(area) > max_area and len(approx) == 4:
            max_area = abs(area)
            contour = cnt
            approx_poly = approx

    cv.drawContours(draw_image, contour, -1, (255, 0, 0), 3)

    for point in approx_poly:
        cv.circle(draw_image, point[0], 10, (0, 255, 0), -1)

    return approx_poly


def order_poly(approx):
    points = np.array(approx)
    new_points = points.reshape((4, 2))
    sums = new_points.sum(1)

    ordered = [x for x in sorted(zip(sums, new_points), key=lambda coor: coor[0])]
    points = [x[1] for x in ordered]
    # print(points)

    return points




def create_warp(img, approx):
    polygon = order_poly(approx)
    pts1 = np.float32(polygon)
    pts2 = np.float32([[0, 0], [width, 0], [0, height+60], [width, height+60]])
    # print(pts1, pts2)

    matrix = cv.getPerspectiveTransform(pts1, pts2)
    output = cv.warpPerspective(draw_image, matrix, (width, height))
    output = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    a, output = cv.threshold(output, 150, 255, cv.THRESH_BINARY)
    output = cv.dilate(output, (7, 7), iterations=1)
    output = cv.erode(output, (5, 5), iterations=1)

    return output


# Video dimensions
proportion = 0.96
width, height = 1920, 1080

# Preparing Camera Video
capture = cv.VideoCapture(1)
capture.set(3, width)
capture.set(4, height)
capture.set(10, 250)

saved = False
while True:
    suc, image = capture.read()
    draw_image = image.copy()
    pre_processed = pre_process(image)
    poly = find_contours(pre_processed, draw_image)
    if len(poly) !=0:
        warped_image = create_warp(image, poly)
        cv.imshow('Warped', warped_image)

        # if not saved:
        #     print(1)
        #     cv.imwrite('Pdf.jpg', warped_image)


    cv.imshow('Video', image)

    if int(cv.waitKey(2)) == ord('q'):
        break

capture.release()
