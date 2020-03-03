import numpy as np
import cv2
import sys
from operator import itemgetter
import matplotlib.pyplot as plt

"""
This `utils` includes the following functions:
    Function 1: Calculate the image scales to realize the image pyramid
    Function 2: Non-max-suppression
    Function 3: post process the image after P-Net and reshape the output size to (24, 24, 3) so that it can be used as the input of R-Net
    Function 4: post process the image after R-Net and reshape the output size to (48, 48, 3) so that it can be used as the input of O-Net
    Function 5: post process the image after O-Net
    Function 6: rectangles to square
"""


def rect_to_square(rectangles):
    """
    Change rectangles to squares
    :param: rectangle -- rectangle[i][0:3] is the location --> the `left upper point` and `right lower point`,
                          rectangles[i][4]  is the score --> the confidence of the bounding box
    :return: squares -- the same as the rectangles
    """
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]

    l = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T

    # return rectangles
    # core_x = rectangles[: 0] + w / 2
    # core_y = rectangles[: 1] - h / 2
    # l = np.maximum(w, h).T
    # rectangles[:, 0] = core_x - l / 2
    # rectangles[:, 1] = core_y + l / 2
    # rectangles[:, 2] = rectangles[:, 0] + l
    # rectangles[:, 3] = rectangles[:, 1] - l

    return rectangles


def non_max_suppression(rectangles, threshold):
    """
    Apply Non-max-suppression for rectangles
    :param: rectangles -- rectangle[i][0:3] is the location --> the `left upper point` and `right lower point`,
                          rectangles[i][4]  is the score --> the confidence of the bounding box
           threshold -- the threshold of IoU
    :return: result_rectangles -- the same as the input rectangles

    Notations:
    Negatives: IoU < 0.3
    Unclear gaps 0.3 - 0.4
    Part faces: 0.4 < IoU < 0.65
    Positives: IoU > 0.65
    """
    if len(rectangles) == 0:
        return rectangles

    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score = boxes[:, 4]

    area = np.multiply(x2 - x1 + 1, y1 - y2 + 1)

    order = score.argsort()[::-1]         # descend order
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])   # choose the four points of the intersection
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h                                      # compute the intersection area
        ovr = inter / (area[i] + area[order[1:]] - inter)  # compute the IoU
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
        result_rectangle = boxes[keep].tolist()
    return result_rectangle


def calculate_scales(img):
    """
    Implement image scales change to realize image pyramid
    :param:  img --  the input image
    :return: scales -- list containing the ratio of scaling
    """
    input_img = img.copy()

    # Step 1: Initialize the image px about 500px * 500px
    init_scale = 1
    h, w, _ = input_img.shape
    if min(h, w) > 500:
        init_scale = 500 / min(h, w)
        h = int(h * init_scale)
        w = int(w * init_scale)
    elif max(h, w) < 500:
        init_scale = 500 / max(h, w)
        h = int(h * init_scale)
        w = int(w * init_scale)

    # Step 2: Scale the image
    scales = []
    factor = 0.709   # after every scale, the area become the half of the prev-image
    factor_count = 0
    min_l = min(h, w)
    while min_l >= 12:
        scales.append(init_scale * pow(factor, factor_count))
        min_l *= factor
        factor_count += 1

    return scales


def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    """
    Implement postprocessing the image after P-Net
    :param cls_prob: softmax feature map for face classify
    :param roi: feature map for regression
    :param out_side: feature map's largest size
    :param scale: current input image scale in multi-scales
    :param width: image's origin width
    :param height: image's origin height
    :param threshold: 0.6 can have 99% recall rate
    :return: rectangles after P-Net and NMS -- (numbers of rectangles, 5)
    """
    in_side = 2*out_side+11
    stride = 0
    if out_side != 1:
        stride = float(in_side-12)/(out_side-1)           # compute the offset scale

    (x, y) = np.where(cls_prob >= threshold)
    bounding_box = np.array([x, y]).T                     # because of the dimensions order (m, n_C, n_H, n_W)

    bb1 = np.fix((stride * bounding_box + 0) * scale)     # left top point offset and reflex to initial image
    bb2 = np.fix((stride * bounding_box + 11) * scale)    # right down point offset and reflex to initial image

    bounding_box = np.concatenate((bb1, bb2), axis=1)     # the bounding box position in the initial image

    dx1 = roi[0][x, y]
    dx2 = roi[1][x, y]
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]
    score = np.array([cls_prob[x, y]]).T
    offset = np.array([dx1, dx2, dx3, dx4]).T

    bounding_box = bounding_box + offset * 12.0 * scale   # the four corners of the bounding box locations in the initial image
    rectangles = np.concatenate((bounding_box, score), axis=1)
    rectangles = rect_to_square(rectangles)

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return non_max_suppression(pick, 0.3)


def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    """
    Filter face position and calibrate bounding box on 12net's output
    :param cls_prob: softmax feature map for face classify
    :param roi: feature map for regression
    :param rectangles: 12net's predict
    :param width: image's origin width
    :param height: image's origin height
    :param threshold:  0.6 can have 99% recall rate
    :return: rectangles after R-Net and NMS  -- (numbers of rectangles, 5)
    """
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T
    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]
    w = x2 - x1
    h = y2 - y1

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T
    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
    rectangles = rect_to_square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])

    return non_max_suppression(pick, 0.3)


def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    """
    Implement post-process after O-Net
    :param cls_prob: cls_prob[1] is face possibility
    :param roi: roi offset
    :param pts: 5 landmarks
    :param rectangles:
    :param width: image's origin width
    :param height: image's origin height
    :param threshold:
    :return:  rectangles after R-Net and NMS  -- (numbers of rectangles, 15)
    """
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]
    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]
    w = x2 - x1
    h = y2 - y1

    pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
    pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
    pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
    pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
    pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T      # compute landmarks' location

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T
    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9), axis=1)
    # print (pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9)

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, rectangles[i][4],
                         rectangles[i][5], rectangles[i][6], rectangles[i][7],
                         rectangles[i][8], rectangles[i][9], rectangles[i][10],
                         rectangles[i][11], rectangles[i][12], rectangles[i][13],
                         rectangles[i][14]])

    return NMS(pick, 0.3)


def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w,h).T
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T
    return rectangles


def NMS(rectangles,threshold):
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle

