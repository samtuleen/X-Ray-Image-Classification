# X-Ray-Image-Classification

## Summary
The goal of this project is to use Deep Learning Neural Network to build an algorithm to classify a set X-rays images of pediatric patients to  help determine if whether or not pneumonia is present. The Neural Network chosen was the Convolutional Neural Network (CNN) as it is one of the preferred for image processing.

### What is Pneumonia?
Pneumonia is an infection that affects one or both lungs. It causes the air sacs, or alveoli, of the lungs to fill up with fluid or pus. Bacteria, viruses, or fungi may cause pneumonia. Symptoms can range from mild to serious and may include a cough with or without mucus (a slimy substance), fever, chills, and trouble breathing. How serious your pneumonia is depends on your age, your overall health, and what is causing your infection. 
[Source](https://www.nhlbi.nih.gov/health-topics/pneumonia)

### What is CNN?
A Convolutional Neural Network (CNN/ConvNet) is a Deep Learning algorithm which can take in an image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a CNN is much lower as compared to other classification algorithms. In simpler terms, the role of the ConvNet/CNN is to reduce the images into a form that is easier to process, without losing features that are critical for getting a good prediction. For an in-depth guide on CNNs, click [here](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

## The Data

### Obtaining the Data
The data was sourced from [kaggle.com](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and consists of 5,756 X-ray images divided into three initial sets ('train', 'test', 'val'). Each folder contains sub-folders ('NORMAL' and 'PNEUMONIA') with the labeled images. 

### Exploring the Data
Upon initially exploring the data, I noticed it contained a number of unusable files such as checkpoints and '.DS_store'. I made the decision to remove them before uploading because my functions repeatedly accessed them instead of the image files which would cause my code to break. I then loaded the data and explored the number of images per set and ensured that all of the files are readable. 

### Data Visualization
Because the X-Rays are labeled into groups it is necessary to verify their balance.
As seen above, the data is clearly unbalanced. I've left it as it is in an attempt to give the highest weight possible to the training set, given that large datasets are necessary for Deep Learning. Furthermore, the train subset will be augmented so that the number of images increases to further stabilize the model.

## Image Preprocessing
I've used two processes for preparing the images for modeling:

**Image Rescaling:** All images need to be rescaled to a fixed size before feeding them to the neural network. The larger the fixed size, the less shrinking required, which means less deformation of patterns inside the image and in turn higher chances that the model will perform well. I've rescaled all of the images to 256 colors (0 - 255).

**Data Augmentation:** The performance of Deep Learning Neural Networks often improves with the amount of data available, therefore increasing the number of data samples could result in a more skillful model. Data Augmentation is a technique to artificially create new training data from existing training data. This is done by applying domain-specific techniques to examples from the training data that create new and different training examples by simply shifting, flipping, rotating, modifying brightness, and zooming the training examples.  [Source](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)

I've augmented the training subset using the following parameters:

** zoom_range=0.3

** vertical_flip=True

# Modeling

1- Use five convolutional blocks comprised of convolutional layers, BatchNormalization, and MaxPooling.

2- Reduce over-fitting by using dropouts.

3- Use ReLu (rectified linear unit) activation function for all except the last layer. Since this is a binary classification problem, Sigmoid was used for the final layer.

4- Use the Adam optimizer and cross-entropy for the loss.

5- Add a flattened layer followed by fully connected layers. This is the last stage in CNN where the spatial dimensions of the input are collapsed into the channel dimension.

Note:
** ReLu: a piecewise linear function that outputs zero if its input is negative, and directly outputs the input otherwise.
** Sigmoid: its gradient is defined everywhere and its output is conveniently between 0 and 1 for all x.

After testing 8 models, I've chosen the following since it has offered satisfactory results based on both its validation and test accuracy.
The final model is a five Convolution Block model with twin (double) Conv2D per block and Dropout of 0.2.

All of the 8 models were initially trained with 10 and 15 epochs and was increased to 50 when the  the validation accuracy increased to more than 70%. Furthermore, an attempt was done to run both models with 100 epochs but neither was successful and ended with the laptop crashing. Once I managed to have results with 100 epochs I will update this repository with those results regardless of their nature (positive or negative).

I've also plotted a Confusion Matrix to aid in the calculation of other metrics such as F-1, recall, and precision.

