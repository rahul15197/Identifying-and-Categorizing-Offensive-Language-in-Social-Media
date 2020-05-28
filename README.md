# Identifying and Categorizing Offensive Language in Social Media
Identifying and Categorizing Offensive Language in Social Media (SemEval 2019 Task 6).

# Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Subtasks](#subtasks)
- [Models](#models)
- [Installation](#installation)
- [Requirements](#requirements)
- [Results](#results)
- [Contribution](#contribution)

## Introduction
Identifying and Categorizing Offensive language in Social Media was a SemEval 2019 Task 6. There are 3 subtasks under the project which helps to categorize the category of offensive language expressed. For the task, we have performed various classification approaches to find best possible accuracy. Detailed explanation about the project is stated under the report subsections.

## Dataset
The dataset used for the purpose of classification and categorization is OLID (Offensive Language Identification Dataset) provided officially by Codalabs and is available [here](https://sites.google.com/site/offensevalsharedtask/olid/OLIDv1.0.zip?attredirects=0&d=1). The training dataset contains tweets with their unique id and their labels for all 3 subtasks. In this project, only English language tweets are taken into account. Offensive language classification can be done for various
other languages also, which is proposed in [SemEval 2020](http://alt.qcri.org/semeval2020/).

OLID contains 13240 tweets each labelled with their id with which they were tweeted and their corresponding labels for subtask A, subtask B & subtask C.

A sample tweet is shown below

![sample_tweet](https://drive.google.com/uc?export=view&id=1-mh6adEXiZDDyC1BlnimWQdath-Ss4rA)

## Subtasks
For this task of categorization and classification of offensive tweet, there are 3 sub categories.

1. Subtask A – Offensive Language Identification [OFF (Offensive) or NOT (Not Offensive)]
To identify whether the given tweet is an offensive tweet or not.
2. Subtask B – Automatic Categorization of Offense Type [UNT (Untargeted) or TIN (Targeted insult)]
To identify whether the offensive tweet is an untargeted insult or a targeted insult tweet.
3. Subtask C – Offense Target Identification [IND (Individual) or GRP (Group) or OTH(Others)]
To identify whether the targeted insult tweet is targeted to an individual, group or others.

All the subtasks are done in the project and their corresponding results are shown in the report.

## Models
We used Naïve Bayes as our baseline model. For the project we have used Support Vector Machine, Decision Tree, K Nearest Neighbor, Logistic Regression and LSTM (Long Short-Term Memory model).


## Installation
```bash
python TaskA.py # to run task A using ML classifiers
python TaskB.py # to run task B using ML classifiers
python TaskC.py # to run task C using ML classifiers
python LSTM_TaskA.py # to run task A using LSTM
python LSTM_TaskB.py # to run task B using LSTM
python LSTM_TaskC.py # to run task C using LSTM
```

## Requirements
Following are the libraries required
* numpy
* pandas
* matplotlib
* sklearn
* keras
* tensorflow
* pickle

Also you need to install CUDA for Nvidia drivers as the project uses CuDNNLSTM that makes use of GPU for training and testing LSTM which works a lot faster than LSTM.

If you are unable to install CUDA or do not have Nvidia drivers, just replace the statements written with "CuDNNLSTM" with "LSTM" and it will work fine.

## Results
For subtask A, we found out that all classifiers gave accuracy higher than 76% and highest of 82.09% using SVM (Support Vector Machine).

For subtask B, we found out that all classifiers gave accuracy higher than 88% and highest of 90.42% using KNN (K Nearest Neighbor).

For subtask C, we found out that all classifiers gave accuracy higher than 61% and highest of 68.08% using LSTM (Long Short-Term Memory).

For detailed classification report and confusion matrix refer to Report.pdf file.

## Contribution
Project was created by [Rahul Maheshwari](mailto:rahul19027@iiitd.ac.in), [Ankit Agarwal](mailto:ankit19021@iiitd.ac.in) and [Diksha Solanki](mailto:diksha19078@iiitd.ac.in)
