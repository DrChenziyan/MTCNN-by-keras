import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, ZeroPadding2D, Activation, Dense, Reshape, Permute, Flatten
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential

"""
MTCNN (Multi-task Cascaded Convolution Neural Network)
step1: Resize the image into image pyramid 
step2: Use P-Net(Proposal Network) to obtain the candidate facial windows and their bounding box regression vectors. 
step3: Use R-Net(Refine Network) to reject a large number of false candidates, performs calibration with bounding box regression, and conducts NMS
step4: USe O-Net(OUtput Network) to produce the final bounding box and the landmarks

Notations:
    Activations: PReLU 

    P-Net input size: (None, None, 3) 
    R-Net input size: (24, 24, 3)
    O-Net input size: (48, 48, 3)

"""


def create_Pnet(weight_path):
    """
    Implement P-Net to obtain the candidate facial widows

    Arguments:
    weight_path: the weights path of P-Net

    Returns:
    P-Net model
    classifier -- whether the image has a face
    bbox_regress -- bounding box regression have left upper corner, height and width, including 4 elements.
    """
    # n_H and n_W could be unknown, and in paper the input size is set to (12, 12, 3) after image scaling
    X_input = Input(shape=[None, None, 3])

    # (12, 12, 3) --> (5, 5, 10)
    X = Conv2D(10, (3, 3), strides=1, padding="valid", name="conv1")(X_input)
    X = PReLU(shared_axes=[1, 2], name="PReLU1")(X)
    X = MaxPool2D(pool_size=2)(X)

    # (5, 5, 10) --> (3, 3, 16)
    X = Conv2D(16, (3, 3), strides=1, padding="valid", name="conv2")(X)
    X = PReLU(shared_axes=[1, 2], name="PReLU2")(X)

    # (3, 3, 16) --> (1, 1, 32)
    X = Conv2D(32, (3, 3), strides=1, padding="valid", name="conv3")(X)
    X = PReLU(shared_axes=[1, 2], name="PReLU3")(X)

    classifier = Conv2D(2, (1, 1), activation="softmax", name="conv4-1")(X)
    bbox_regress = Conv2D(4, (1, 1), name="conv4-2")(X)

    model = Model([X_input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)

    return model


def create_Rnet(weigth_path):
    """
    Implement R-Net to choose the candidate windows and refine the bounding box
    :param weigth_path:  the weigths of R-Net
    :return:
    R-Net model
    classifier -- face classification after FC
    bbox_regress -- refine the bounding box through FC
    """
    # input size are set to (24, 24, 3)
    X_input = Input(shape=[24, 24, 3])

    # (24, 24, 3) --> (11, 11, 28)
    X = Conv2D(28, (3, 3), strides=1, padding="valid", name="conv1")(X_input)
    X = PReLU(shared_axes=[1, 2], name="PReLU1")(X)
    X = MaxPool2D(pool_size=3, strides=2, padding="same")(X)

    # (11, 11, 28) --> (4, 4, 48)
    X = Conv2D(48, (3, 3), strides=1, padding="valid", name="conv2")(X)
    X = PReLU(shared_axes=[1, 2], name="PReLU2")(X)
    X = MaxPool2D(pool_size=3, strides=2)(X)

    # (4， 4， 48)  --> (3，3，64)
    X = Conv2D(64, (2, 2), strides=1, padding="valid", name="conv3")(X)
    X = PReLU(shared_axes=[1, 2], name="PReLU3")(X)
    X = Permute((3, 2, 1))(X)      # transpose to (n_C, n_W, n_H)
    X = Flatten()(X)

    # Fully connected layer
    X = Dense(128, name="conv4")(X)
    X = PReLU(name="PReLU4")(X)

    classifier = Dense(2, activation="softmax", name="conv5-1")(X)
    bbox_regress = Dense(4, name="conv5-2")(X)

    model = Model([X_input], [classifier, bbox_regress])
    model.load_weights(weigth_path, by_name=True)

    return model


def create_Onet(weight_path):
    """
    Implement O-Net to out put the final facial window and bounding box
    :param weight_path: the weights of O-Net
    :return:
    O-Net model
    classifier -- final classification
    bbox_regress -- final bounding box
    landmark_regress -- the 5 landmarks in the bounding box including left eye, right eye, nose, left mouth corner,
    and right mouth corner, thus landmark_regress have 10 elements
    """
    # input size are set to (48, 48, 3)
    X_input = Input(shape=[48, 48, 3])

    # (48, 48, 3) --> (23, 23, 32)
    X = Conv2D(32, (3, 3), strides=1, padding="valid", name="conv1")(X_input)
    X = PReLU(shared_axes=[1, 2], name="PReLU1")(X)
    X = MaxPool2D(pool_size=3, strides=2, padding="same")(X)

    # (23, 23, 32) --> (10, 10, 64)
    X = Conv2D(64, (3, 3), strides=1, padding="valid", name="conv2")(X)
    X = PReLU(shared_axes=[1, 2], name="PReLU2")(X)
    X = MaxPool2D(pool_size=3, strides=2)(X)

    # (10, 10, 64) --> (4, 4, 64)
    X = Conv2D(64, (3, 3), strides=1, padding="valid", name="conv3")(X)
    X = PReLU(shared_axes=[1, 2], name="PReLU3")(X)
    X = MaxPool2D(pool_size=2)(X)

    # (4, 4, 64) --> (3, 3, 128)
    X = Conv2D(128, (2, 2), strides=1, padding="valid", name="conv4")(X)
    X = PReLU(shared_axes=[1, 2], name="PReLU4")(X)

    X = Permute((3, 2, 1))(X)
    X = Flatten()(X)

    # Fully connected 256 vector
    X = Dense(256, name="conv5")(X)
    X = PReLU(name="PReLU5")(X)

    classifier = Dense(2, activation="softmax", name="conv6-1")(X)
    bbox_regress = Dense(4, name="conv6-2")(X)
    landmark_regress = Dense(10, name="conv6-3")(X)

    model = Model([X_input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model


