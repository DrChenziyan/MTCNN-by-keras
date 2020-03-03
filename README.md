# MTCNN-by-keras
Apply MTCNN through keras with pre-trained weights
******

# File Statement

1 **Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks .pdf**
This is the lecture of multi-task cascaded convolution networks(MTCNN) which declares the architecture of the networks, the training process and the comparison with other networks.

2 **image dircetory**
This dircetory contains the test images and the output images , the later are in *out directory*.

3 **model weights directory**
This directory contains the MTCNN weights including *12net.h5(pnet.h5)*, *24net.h5(rnet.h5)* and *48net.h5(onet.h5)* 

4 **MTCNN.py**
The architecture of the MTCNN, containing 3 function `create_Pnet`, `create_Rnet` and `create_Onet`

5 **mtcnn_utils.py**
Some assistant function to realize the image pyramid change, post process after each network, non-max-suppression and image shape change, et al. There are 2 functions to do NMS and image shape change, and both have similiar performance.

6 **face_detection.py**
Put the upper functions together to do the face detection. 

7 **Detector.py (Detector.ipynb)**
The detection function, you can pass a image to function `face_detection()` and then use openCV to display the final output and store it into *image/out*.

****

# Software version
python == 3.7.4
numpy == 1.18.1
opencv-python == 4.2.0.32 
tensorflow == 1.15.0
Keras == 2.1.0
