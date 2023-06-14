# Epileptic-Seizure-Prediction-using-EEG-Signals
Using Machine Learning models to predict Epileptic Seizure from EEG signals.

This project was made for my Term project of Machine Learning course.

PROJECT DESCRIPTION

Epilepsy is a neurological disorder that affects approximately fifty million people according to the World Health Organization. While electroencephalography (EEG) plays important roles in monitoring the brain activity of patients with epilepsy and diagnosing epilepsy, an expert is needed to analyze all EEG recordings to detect epileptic activity. This method is obviously time-consuming and tedious, and a timely and accurate diagnosis of epilepsy is essential to initiate antiepileptic drug therapy and subsequently reduce the risk of future seizures and seizure-related complications.

In this project, we used the knowledge that we’ve gained from the Machine Learning Course to evaluate the dataset containing information about EEG signals. We have used several Machine Learning algorithms like Decision Trees, Random Forest regressor, CNN, ANN, BiLSTM etc. and gained more than 95% accuracy in predicting whether the data point represents signs of epilepsy or not.

PROJECT OBJECTIVE

* To make accurate predictions on the EEG signal dataset by using several Classifiers
* To be able to make future predictions about Epilepsy disorders.

UNDERSTANDING THE DATASET

The original dataset from the reference comprises 500 files, each file representing a single person/subject. Each recording contains the EEG signal value of brain activity for 23.6s sampled into 4096 data points. These recordings have been split into 1s windows. This results in 23 x 500 = 11500 windows of EEG data over time in 178 datapoints and each window is categorized into 5 labels:

Seizure activity
EEG recorded at tumor site
EEG recorded in healthy brain area
eyes closed during recording
eyes open during recording

Subjects labeled with classes 2-5 did not have epileptic seizures. So we converted the class labels into a binary classification of subjects suffering an epileptic seizure or not, meaning classes 1 or 0, respectively.

MODEL EVALUATION

![image](https://github.com/RushilShivade/Epileptic-Seizure-Prediction-using-EEG-Signals/assets/116446026/d31e76c0-ccb3-4a28-bcde-9bebfe65ec0c)


With this project, we have successfully classified the dataset of EEG signals and predicted with an accuracy of 95% accuracy from Bilateral LSTM. Other models like Random forest and decision tree/
