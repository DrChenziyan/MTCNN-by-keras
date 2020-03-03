import cv2
from face_detection import face_detection

img = cv2.imread("image/reba.jpeg")
threshold = [0.5, 0.5, 0.6]
rectangles = face_detection(img, threshold)
draw = img.copy()

for rectangle in rectangles:
    if rectangle is not None:
        W = -int(rectangle[0]) + int(rectangle[2])
        H = -int(rectangle[1]) + int(rectangle[3])
        padding_H = 0.01 * W
        padding_W = 0.01 * H
        crop_img = img[int(rectangle[1] + padding_H):int(rectangle[3] - padding_H), int(rectangle[0] - padding_W):int(rectangle[2] + padding_W)]
#         crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        if crop_img is None:
            continue
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue
        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)

        for i in range(5, 15, 2):
            cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))


cv2.imwrite("image/out/reba_out.jpg", draw)

c = cv2.waitKey(0)