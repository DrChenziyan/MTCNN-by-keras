import numpy as np
import tensorflow as tf
import cv2
from MTCNN import create_Onet, create_Pnet, create_Rnet
from mtcnn_utils import rect_to_square, non_max_suppression, calculate_scales, detect_face_12net, filter_face_24net, filter_face_48net, NMS, rect2square

P_Net = create_Pnet("model weights/pnet.h5")
R_Net = create_Rnet("model weights/rnet.h5")
O_Net = create_Onet("model weights/onet.h5")


def face_detection(img, threshold):
    # image Normalization
    norm_img = (img.copy() - 127.5) / 128  # image pixels range (0, 255)
    origin_H, origin_W, origin_C = norm_img.shape

    scales = calculate_scales(norm_img)

    ### Step 1 : Run P-Net and post process
    out = []
    for scale in scales:
        temp_H = int(origin_H * scale)
        temp_W = int(origin_W * scale)
        scale_img = cv2.resize(norm_img, (temp_W, temp_H))  # resize the image
        #         print(temp_H, temp_W)
        inputs = scale_img.reshape(1, *scale_img.shape)     # add a dimension to keep the NCHW(m, n_C, n_H, n_W)
        output = P_Net.predict(inputs)
        out.append(output)

    rectangles = []  # define a list to store the output of "detect_face_12net"
    for i in range(len(scales)):
        # i = #scale, first 0 select cls score, second 0 = batchnum
        cls_prob = out[i][0][0][:, :, 1]  # the confidence of the output, which means the probablity of the face at the temp_window
        roi = out[i][1][0]                # the position of the output
        out_h, out_w = cls_prob.shape
        out_side = max(out_h, out_w)

        # dimensions change for better computation
        cls_prob = np.swapaxes(cls_prob, 0, 1)
        roi = np.swapaxes(roi, 0, 2)
        #         print(cls_prob.shape)

        rectangle = detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_W, origin_H, threshold[0])
        rectangles.extend(rectangle)

    rectangles = NMS(rectangles, 0.7)
    print("P-Net rectangles shape = " + str(np.shape(rectangles)))

    if len(rectangles) == 0:
        return rectangles

    ### Step 2: Run R-Net and post process
    predict_24_batch = []
    for rectangle in rectangles:
        # crop the image from the initial norm image
        crop_img = norm_img[int(rectangle[1]): int(rectangle[3]), int(rectangle[0]): int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (24, 24))
        predict_24_batch.append(scale_img)  # store the all crop imgaes into a list

    predict_24_batch = np.array(predict_24_batch)
    R_net_out = R_Net.predict(predict_24_batch)  # 'R_net_out' is a list

    rnet_cls_prob = R_net_out[0]  # R-Net output is [classifier, bbox_regress]
    rnet_cls_prob = np.array(rnet_cls_prob)
    rnet_roi = R_net_out[1]
    rnet_roi = np.array(rnet_roi)

    rectangles = filter_face_24net(rnet_cls_prob, rnet_roi, rectangles, origin_W, origin_H, threshold[1])
    print("R-Net rectangles shape = " + str(np.shape(rectangles)))

    if len(rectangles) == 0:
        return rectangles

    ### Step 3: Run O-Net and post process
    predict_48_batch = []
    for rectangle in rectangles:
        crop_img = norm_img[int(rectangle[1]): int(rectangle[3]), int(rectangle[0]): int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (48, 48))
        predict_48_batch.append(scale_img)

    predict_48_batch = np.array(predict_48_batch)
    O_net_out = O_Net.predict(predict_48_batch)

    # O-Net output is [classifier, bbox_regress, landmark_regress]
    onet_cls_prob = O_net_out[0]
    #     onet_cls_prob = np.array(onet_cls_prob)
    onet_roi = O_net_out[1]
    #     onet_roi = np.array(onet_roi)
    onet_pts = O_net_out[2]
    #     onet_pts = np.array(onet_pts)

    rectangles = filter_face_48net(onet_cls_prob, onet_roi, onet_pts, rectangles, origin_W, origin_H, threshold[2])
    print("O-Net rectangles shape = " + str(np.shape(rectangles)))

    return rectangles