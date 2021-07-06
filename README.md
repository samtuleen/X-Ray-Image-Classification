# X-Ray-Image-Classification

## Summary
The goal of this project is to use Deep Learning Neural Network to build an algorithm to classify a set X-rays images of pediatric patients to  help determine if whether or not pneumonia is present. The Neural Network chosen was the Convolutional Neural Network (CNN) as it is one of the preferred for image processing.

## What is Pneumonia?
Pneumonia is an infection that affects one or both lungs. It causes the air sacs, or alveoli, of the lungs to fill up with fluid or pus. Bacteria, viruses, or fungi may cause pneumonia. Symptoms can range from mild to serious and may include a cough with or without mucus (a slimy substance), fever, chills, and trouble breathing. How serious your pneumonia is depends on your age, your overall health, and what is causing your infection. 
[Source](https://www.nhlbi.nih.gov/health-topics/pneumonia)

## What is CNN?
A Convolutional Neural Network (CNN/ConvNet) is a Deep Learning algorithm which can take in an image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a CNN is much lower as compared to other classification algorithms. In simpler terms, the role of the ConvNet/CNN is to reduce the images into a form that is easier to process, without losing features that are critical for getting a good prediction. For an in-depth guide on CNNs, click [here](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

## The Data

### Obtaining the Data
The data was sourced from [kaggle.com](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and consists of 5,756 X-ray images divided into three initial sets ('train', 'test', 'val'). Each folder contains sub-folders ('NORMAL' and 'PNEUMONIA') with the labeled images. 

### Exploring the Data
Upon initially exploring the data, I noticed it contained a number of unusable files such as checkpoints and '.DS_store'. I made the decision to remove them before uploading because my functions repeatedly accessed them instead of the image files which would cause my code to break. I then loaded the data and explored the number of images per set and ensured that all of the files are readable. 

### Data Visualization
Because the X-Rays are labeled into groups it is necessary to verify their balance and as seen in the following visualization it clearly is.