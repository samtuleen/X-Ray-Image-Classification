# Detecting Pediatric Pnuemonia From X-Rays Using Image Classification

## Summary
The purpose of this project is to use Deep Learning Neural Networks to build an algorithm that classifies a set X-rays images belonging to pediatric patients. This algorithm will help determine if whether or not pneumonia is present. The Neural Network chosen was the Convolutional Neural Network (CNN) as it is one of the preferred networks for image processing.

### What is Pneumonia?
Pneumonia is an infection affecting one or both lungs by causing the air sacs, or alveoli, of the lungs to fill up with fluid or pus. It is caused by bacteria, viruses, or fungi. Symptoms can range from mild to serious and may include a cough with or without mucus (a slimy substance), fever, chills, and trouble breathing. The severity of the infection depends on the age, overall health (of the infected), and the cause of the infection
[Source](https://www.nhlbi.nih.gov/health-topics/pneumonia)

### What is a CNN?
A Convolutional Neural Network (CNN/ConvNet) is a Deep Learning algorithm that can take in an image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a CNN is much lower as compared to other classification algorithms. In simpler terms, the role of the ConvNet/CNN is to reduce the images into a form that is easier to process, without losing features that are critical for getting a good prediction. For an in-depth guide on CNNs, click [here](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

## The Data

### Obtaining the Data
The data was sourced from [kaggle.com](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and consists of 5,756 X-ray images divided into three initial sets ('train', 'test', 'val'). Each folder contains sub-folders ('NORMAL' and 'PNEUMONIA') with the labeled images. 

### Exploring the Data
Upon initially exploring the data, I noticed it contained a number of unusable files such as checkpoints and '.DS_store'. I made the decision to remove them before uploading because my functions repeatedly accessed them instead of the image files which would cause my code to break. I then loaded the data and explored the number of images per set and ensured that all of the files are readable. 

**insert image**


### Data Visualization
Because the X-Rays are labeled into groups it is necessary to verify their balance.

**insert image**

As seen above, the data is clearly unbalanced. I've left it as it is in an attempt to give the highest weight possible to the training set, given that large datasets are necessary for Deep Learning. Furthermore, the train subset will be augmented so that the number of images increases to further stabilize the model.

## Image Preprocessing
I've used two processes for preparing the images for modeling:

**Image Rescaling:** All images need to be rescaled to a fixed size before feeding them to the neural network. The larger the fixed size, the less shrinking required, which means less deformation of patterns inside the image and in turn higher chances that the model will perform well. I've rescaled all of the images to 256 colors (0 - 255).

**Data Augmentation:** The performance of Deep Learning Neural Networks often improves with the amount of data available, therefore increasing the number of data samples could result in a more skillful model. Data Augmentation is a technique to artificially create new training data from existing training data. This is done by applying domain-specific techniques to examples from the training data that create new and different training examples by simply shifting, flipping, rotating, modifying brightness, and zooming the training examples.  [Source](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)

I've augmented the training subset using the following parameters:

** zoom_range=0.3

** vertical_flip=True

## Modeling

I've built my model by doing the following:

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

**insert image**

All of the 8 models were initially trained with 10 and 15 epochs and was increased to 50 when the  the validation accuracy increased to more than 70%. Additionally, I've attempted to increase the epochs to 100 however my system couldn't handle the burden.

I've also plotted a Confusion Matrix to aid in the calculation of other metrics such as F-1, recall, and precision.

## Analyzing the Model's Performance

The simplest way to analyze the perfomance of a model is to examine a visualization of its results:

**insert image**

As can be seen above, the curves of the validation accuracy and the loss indicate that the model may converge with more than 50 epochs (though it hasn't happened with the ones its been fitted with). As I've previously mentioned, I've attempted more but failed.

It's clear from the train accuracy that the model is overfitted; however the model's accuracies were still decent with the test subset. See the results below:

* Accuracy: 91.22137404580153%
* Precision: 90.38461538461539%
* Recall: 96.76470588235294%
* F1-score: 93.4659090909091

It's important to note in evaluating the model that accuracy is an inappropriat perfomance measure for imbalanced classification problems such as this. This is because the high number of samples that form the majority class (training subset) overshadows the number of examples in the minority class. This translates to accuracy scores as high as 99% for unskilled models, depending on how severe the class imbalance is. [Source](https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/)

As an alternative better option, using precision, recall and F-1 metrics can offer more reliable results. These metric concepts are are follows:

**Precision:** quantifies the number of correct positive predictions made: it calculates the accuracy for the minority class. = tp/(tp + tp)

**Recall:** quantifies the number of positive class predictions made out of all positive examples in the dataset. It provides an indication of missed positive predictions. = tp/(tp + fn)

**F1-Score:** balances both the concerns of precision and recall in one number. A good F1 score means low false positives and low false negatives  = (tp + tp)/total

**Note: tp = true positive, tn = true negative, fp = false positive, and fn = false negative**


### Confusion Matrix
I'll now examine the plotted confusion to better understand these metrics. It's important to keep in mind that the most important one is Recall, because patients prefer that the doctor misdiagnosed them with pneumonia when in fact they don't, over a healthy misdiagnosis when in reality pneumonia is present. While Precision is an important metric,  Recall for medical problems may be more accurate.

**insert image**

In viewing the matrix, we see 11 representing the fn and the 329 representing the tp from our model, we can deduct that this means that only 11 patients are misdiagnosed as *not* having pneumonia.

# Conclusion
This project has shown how to classify positive and negative pneumonia diagnosis' from a set of X-Ray images and although it's far from complete and could be improved, it is amazing to see the success of deep learning being used in real world problems.

# Recommended Next Steps
* Source more data such as the Mendeley dataset, since it contains a larger number of images and may be more suitable for Deep Learning. Doing so will allow us to process the data with a more balanced distribution, as opposed to the Kaggle dataset, where the 'val' subset contains 16 images out of 5756 (only 0.2% of the data).

* Attempt to build the models using larger datasets and running them with a greater number of epochs (such as 100 or more) if necessary to determine if there is convergence. I've unsuccessfully attempted this, however my system couldn't handle it.

* Fine tune and test other parameters to reduce overfitting.

* With this project as a base, our work can be built upon to detect more complex X-Rays, such as cancers, broken bones and more.
