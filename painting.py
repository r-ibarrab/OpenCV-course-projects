import cv2 as cv
import numpy as np
import math


def ignore_function(e):
    pass


def create_trackbars():
    cv.namedWindow('Trackbars')
    cv.createTrackbar('Hue min', 'Trackbars', 0, 179, ignore_function)
    cv.createTrackbar('Hue max', 'Trackbars', 179, 179, ignore_function)
    cv.createTrackbar('Sat min', 'Trackbars', 0, 255, ignore_function)
    cv.createTrackbar('Sat max', 'Trackbars', 255, 255, ignore_function)
    cv.createTrackbar('Val min', 'Trackbars', 0, 255, ignore_function)
    cv.createTrackbar('Val max', 'Trackbars', 255, 255, ignore_function)


def preparation_find_color(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos('Hue min', 'Trackbars')
    h_max = cv.getTrackbarPos('Hue max', 'Trackbars')
    s_min = cv.getTrackbarPos('Sat min', 'Trackbars')
    s_max = cv.getTrackbarPos('Sat max', 'Trackbars')
    v_min = cv.getTrackbarPos('Val min', 'Trackbars')
    v_max = cv.getTrackbarPos('Val max', 'Trackbars')

    min_range = np.array([h_min, s_min, v_min])
    max_range = np.array([h_max, s_max, v_max])

    mask = cv.inRange(img_hsv, min_range, max_range)

    cv.imshow('Mask', mask)

    return mask


def find_contours(img, lower, upper, color, action):
    mask = cv.inRange(img, lower, upper)
    contours, heir = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 300:
            cv.drawContours(image, cnt, -1, color, 1)
            arc = cv.arcLength(cnt, True)
            vert = cv.approxPolyDP(cnt, 0.02 * arc, True)
            x, y, w, h = cv.boundingRect(vert)
            # cv.rectangle(image, (x, y), (x + w, y + h), color, 3)
            print(action)
            cv.circle(image, (x + w//2, y), 5, color, -1)

            if action == 'd':
                for cont, point in enumerate(colored_points):
                    if abs(x + w//2 - point[0]) <= 8 and abs(y - point[1]) <= 8:
                        print('deleting')
                        del colored_points[cont]
                    cv.rectangle(image, (x + w//2 - 8, y - 8), (x + w//2 + 8, y + 8), color, 2)
            else:
                colored_points.append((x + w // 2, y, color))


def find_colors(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    for cont, color in enumerate(colors):
        find_contours(img_hsv, np.array(color[:3]), np.array(color[3:6]), contour_colors[cont], color[-1])


def paint_previous_points():
    for point in colored_points:
        cv.circle(image, (point[0], point[1]), 5, point[2], -1)


width, height = math.floor(1920 * 0.5), math.floor(1080 * 0.4)
colors = [[20, 96, 146, 42, 255, 194, 'p'],
          [92, 157, 45, 116, 255, 255, 'p'],
          [149, 149, 123, 170, 255, 255, 'd']]

contour_colors = [(64, 236, 227), (245, 219, 88), (232, 153, 241)]
colored_points = []


cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# create_trackbars()

while True:
    suc, image = cap.read()
    # preparation_find_color(image)
    image_blurry = cv.GaussianBlur(image, (5, 5), 1)

    paint_previous_points()
    find_colors(image_blurry)

    image_flip = cv.flip(image, 1)
    cv.imshow('Video', image_flip)

    if int(cv.waitKey(1)) == ord('q'):
        break

cap.release()
