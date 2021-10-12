# 2_Image_Segmentation_using_U-net_Architecture

Ronneberger et al., 2015 presented U-net architecture (Fig. 1) to segment biomedical images [1].However, other than this specific application, U-net architecture has been found to be very useful for much broader computer vision application for semantic segmentation [2]. Input (original image) to the U-net is h * w * 3 (RGB image) and its output would be a segmented image with the same size as the original image but with the number of classes specified as the number of channels (h * w * nC) [2].

![Fig  1](https://user-images.githubusercontent.com/54812742/136703882-1a15430c-16a8-4e59-a0dc-2f52d5e19033.PNG)

U-net mainly consists of two paths: encoder and decoder [3]. The encoder consists of normal feedforward neural network including convolutional layers with occasional max pooling layers in between. The number of channels increases after applying each convolutional layer while the size (height and width) of the image decreases after applying each max pooling layer. After repeating this sequence of convolutional layers followed by max pooling layers we end up with an image with a very small height and width but with a large number of channels. This is where the decoder path starts through applying up (also called transpose) convolutional layers to increase the height and width of the image while decreasing the number of channels. In each step, the skip connection, called “copy and crop” in Fig. 1, transfers some low-level set of activations from the corresponding layer in the encoder path to the transpose convolutional layer output. Then there would be more layers of regular convolutional layers followed by RELU activation function (denoted by dark blue arrays in Fig. 1). This sequence of transpose convolutional layer to be combined with the corresponding level from encoder and then, applying some convolutional layers would be repeated. This leads to upsize the image to the original size but with the number of channels equal to the number of classes. 

In this project, Oxford Pets dataset (https://www.kaggle.com/c/oxford-iiit-pet-dataset/data), consisting of 37 breeds of cats or dogs along with their corresponding masks, was used as the training datasets. Training and validation loss and accuracies are plotted in Fig. 2a and b, respectively. 

![image](https://user-images.githubusercontent.com/54812742/136981366-18e296d4-8da4-4cd8-9d5c-2cb6473230da.png)


Also, some random images from the testing dataset are chosen and the corresponding segmented outputs are shown in  Fig. 3. The ground truth labels are also shown (in the middle) for comparison purposes. 

![image](https://user-images.githubusercontent.com/54812742/136980426-6d0207e4-d074-41da-a3db-efaebca84155.png)


For semantic segmentation, accuracy is not an appropriate metric to evaluate the model performance. And instead, intersection over union (IoU) is a more reliable criteria. The IoU for different classes were calculated as 0.77, 0.86, and 0.46 for the class 1, 2, and 3, respectively. The mean IoU is 0.69615537.

References:

[1] Ronneberger, O., Fischer, P.,  Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.

[2] Andrew Ng (2021), Convolutional neural networks Coursera course, https://www.coursera.org/learn/convolutional-neural-networks

[3] Sreenivas Bhattiprolu's youtube channel (208 - Multiclass semantic segmentation using U-Net), https://www.youtube.com/watch?v=XyX5HNuv-xE&t=413s
