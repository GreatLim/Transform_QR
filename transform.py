import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_four_points(cnt):
    start_point = cnt[0][0]
    points = [start_point, start_point, start_point, start_point]
    start_num1 = start_point[0] + start_point[1]
    start_num2 = start_point[1] - start_point[0]
    nums = [start_num1, start_num2, start_num2, start_num1]

    for point in cnt:
        num1 = point[0][0] + point[0][1]
        num2 = point[0][1] - point[0][0]
        if num1 < nums[0]:
            nums[0] = num1
            points[0] = point[0]
        if num1 > nums[3]:
            nums[3] = num1
            points[3] = point[0]
        if num2 < nums[1]:
            nums[1] = num2
            points[1] = point[0]
        if num2 > nums[2]:
            nums[2] = num2
            points[2] = point[0]

    return points


def transform(path):
    img = cv2.imread(path, 0)
    name = path.split("/")[-1]

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, kernel1, iterations=3)
    eroded = cv2.erode(dilated, kernel2, iterations=1)
    im2, contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_contour = max(contours, key=cv2.contourArea)
    points = find_four_points(max_contour)

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (300, 300))

    # plt.subplot(121)
    # plt.imshow(img)
    # plt.title('input')
    # plt.subplot(122)
    # plt.title('output')
    # plt.imshow(dst)
    # plt.show()
    cv2.imwrite(name, dst)


if __name__ == '__main__':
    path = "./problem/7.jpg"
    transform(path)
