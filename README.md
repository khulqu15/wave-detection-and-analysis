# wave-detection-and-analysis
Detecting waves in a video and analyzing the wave run-up height, as well as classifying the wave data at the coastal end to measure the wave reach. The code utilizes optical flow and deep learning algorithms to provide accurate results. The project is useful for coastal engineering research, environmental monitoring, and coastal management.

This program loads a dataset of wave and non-wave images, preprocesses the images, builds a convolutional neural network (CNN) model using TensorFlow's Keras API, trains the model on the dataset, evaluates the model, and saves the model.

In more detail, the program performs the following steps:

1. Loads the wave and non-wave images from the "datasets/" directory and creates a list of the image arrays and corresponding labels.
2. Resizes the images to 150x150 pixels, scales the pixel values to be between 0 and 1, and reshapes the data to have a fourth dimension for the channel.
3. Splits the data into training and validation sets, and converts the labels to one-hot encoded vectors.
4. Builds a CNN model using Keras with the following layers:
    - Reshape layer: Reshapes the input data to have a fourth dimension for the channel.
    - Conv2D layer: Applies 32 filters of size 3x3 to the input with a ReLU activation function.
    - MaxPooling2D layer: Performs max pooling with a pool size of 2x2.
    - Conv2D layer: Applies 64 filters of size 3x3 to the input with a ReLU activation function.
    - MaxPooling2D layer: Performs max pooling with a pool size of 2x2.
    - Conv2D layer: Applies 64 filters of size 3x3 to the input with a ReLU activation function.
    - Flatten layer: Flattens the output of the previous layer into a 1D vector.
    - Dense layer: Applies 64 units with a ReLU activation function.
- Dense layer: Applies a softmax activation function to produce the final output.
5. Compiles the model with the Adam optimizer, categorical cross-entropy loss function, and accuracy metric.
6. Trains the model on the training data for 10 epochs, using the validation data for validation.
7. Evaluates the model on the test data and prints the test loss and accuracy.
8. Saves the model to a file called "wave_detection_model".

The CNN model consists of three convolutional layers, each followed by a max pooling layer, and two dense layers. The first convolutional layer applies 32 filters of size 3x3 to the input with a ReLU activation function. The second convolutional layer applies 64 filters of size 3x3 to the input with a ReLU activation function. The third convolutional layer applies 64 filters of size 3x3 to the input with a ReLU activation function. The dense layers each have 64 units with a ReLU activation function and a softmax activation function, respectively.

After learning, the convolutional layers produce feature maps and the dense layers make a final classification prediction based on the learned features. The transfer functions used in the convolutional layers are ReLU, which is commonly used in CNNs to introduce non-linearity and help with the detection of complex features in images. The final layer uses a softmax activation function to produce the class probabilities.

A transfer function, in the context of neural networks, is a function that maps the inputs of a neural network to its outputs. It can be seen as a mathematical representation of the neural network's behavior.

In the code above, the transfer functions are used in the layers of the convolutional neural network (CNN). The Rectified Linear Unit (ReLU) transfer function is used in the first three convolutional layers, and the softmax function is used in the last dense layer.

The ReLU function is a simple piecewise linear function that returns the input if it is positive, and 0 if it is negative. Its output is graphed as a straight line with slope 1 for x > 0, and 0 for x <= 0.

The softmax function is a nonlinear function that maps a vector of arbitrary real values to a vector of values between 0 and 1 that sum up to 1. Its output can be graphed as a curve that starts at 0, increases gradually, and then sharply rises to 1 as the input values become large.

The output of the neural network is a probability distribution over the two classes, waves and not_waves, represented by a vector of two values between 0 and 1 that sum up to 1. The softmax function ensures that the output vector is a valid probability distribution. The predicted class is the class with the highest probability value.

# Datasets Link
https://drive.google.com/drive/folders/18M7vF-0UUneNY3roh9RdUrDTJxxfyk7-?usp=sharing

# Author
EEPIS : Mohammad Khusnul Khuluq @khulqu15