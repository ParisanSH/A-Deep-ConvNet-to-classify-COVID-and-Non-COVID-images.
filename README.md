# A-Deep-ConvNet-to-classify-COVID-and-Non-COVID-images.

In this project, I have designed and trained a deep ConvNet to classify COVID and Non-COVID images. In order to do that, I have used the COVID, Non-COVID CT IMAGES dataset which is available in the below link: 
https://drive.google.com/drive/folders/1kVIe0HIYz_k9Jcjn27ViHPe51AG9y_fr?usp=sharings

# Steps:
1. First, the data has been divided into 80% for training and 20% for testing the model. Note that there is more than one image for a patient, so the images of a particular patient should not be in both training and test data.
2. Some necessary pre-processing on the images has been done, and since the images in the database have different sizes, all images should resize suitably for entering the network.
3. A deep ConvNet as a classifier to distinguish between COVID and Non-COVID CT images has been trained.
4. The precision, recall, F1-score, accuracy, and AUC criteria have been calculated and reported.
5. Finally the ROC curve has been plotted.
