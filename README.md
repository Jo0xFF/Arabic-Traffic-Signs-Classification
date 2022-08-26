# Arabic traffic signs classification
![Banner](https://i.imgur.com/TJnTUTa.jpeg)

![python version](https://img.shields.io/badge/python-3.7.13-blue&?style=for-the-badge&logo=python&color=blue&labelColor=black)
![GitHub repo size](https://img.shields.io/github/repo-size/Jo0xFF/Arabic-Traffic-Signs-Classification?logo=Github&style=for-the-badge&color=red)
![ML Problem](https://img.shields.io/badge/Type%20of%20ML-Multi--class%20Classification-red&?style=for-the-badge&logo=TensorFlow&color=darkviolet)
![License](https://img.shields.io/badge/LICENSE-MIT-green&?style=for-the-badge)
![Github Commit](https://img.shields.io/github/last-commit/Jo0xFF/Arabic-Traffic-Signs-Classification?style=for-the-badge)
![Flask logo](https://img.shields.io/badge/Flask-black&?style=for-the-badge&logo=flask&color=black)
[![Heroku logo](https://img.shields.io/badge/Heroku-Open%20in%20Heroku-black&?style=for-the-badge&logo=Heroku&logoColor=6762A6&labelColor=black&color=6762A6&link=https://arabic-traffic-signs-classify.herokuapp.com&link=https://arabic-traffic-signs-classify.herokuapp.com)](https://arabic-traffic-signs-classify.herokuapp.com/)


## Table of Contents
- [Overview](#Overview-section)
- [What kind of traffic signs](#traffic-section-imgs)
- [How it works](#how-it-works)
- [Data](#Data-section)
- [Splitting-ratio](#Splitting-ratios-section)
- [EDA](#EDA-section)
    - [Imbalance class](#Imbalance-class-section)
    - [Inspect image shapes](#Inspect-image-shapes-subsection)
- [Architectures](#Architectures-section)
    - [Baseline architecture](#Baseline-architecture-subsection)
    - [Baseline killer architecture](#Baseline-killer-architecture-subsection)
- [Wandb(Weights & Biases) Experiment monitoring](#wandb-exper-section)
- [Loss metrics](#Loss-metrics-section)
    - [Baseline losses metrics](#Baseline-losses-metrics-subsection)
    - [Baseline killer loss metrics](#Baseline-killer-loss-metrics-subsection)
- [Confusion matrix](#Confusion-matrix-section)
- [F1 score](#F1-score)
- [Predictions on test dataset](#Predictions-on-test-dataset-section)
    - [For Baseline architecture](#For-Baseline-architecture-subsection)
    - [For Baseline killer architecture](#For-Baseline-killer-architecture-subsection)
- [Deployment](#Deployment-section)
- [Libraries used](#Libraries-used-section)
- [Dev logs](#Dev-logs-section)
- [Licences and Citations](#Licences-and-Citations-section)


## Overview
Arabic traffic signs classification is one of the main components of autonomous cars. This project 
Developed for testing different CNN architectures and layers to seek more generalized predictions. The target of this project to classify 24 different traffic signs on the wild and get prediction accuracy higher than 80% for each class. 


## What kind of traffic signs
This project, based on the commonly 24 Arabic traffic signs as mentioned in a later section for more info jump to [Data](#Data-section) section


## How it works
- Prepare an arabic traffic sign or a similar to arabic traffic sign in your country .
- Browse image and click submit!

![How-it-works](https://i.imgur.com/spuBZQb.gif)

## Data
the data where this project takes from: [Dataset page](https://data.mendeley.com/datasets/4tznkn45mx) all rights to their respected authors.
- "The dataset consists of 2,718 real captured images and 57,078 augmented images for 24 Arabic traffic signs. The images are captured from three connected cities (Khobar, Dammam and Dhahran) in the Eastern Province of Saudi Arabia. The newly developed dataset consisting of 2,718 real images is randomly partitioned into 80% percent training set (2,200 images) and 20% percent testing set (518) images"

## Splitting ratios
With splitting I used an awesome library called [Split folders](https://pypi.org/project/split-folders/). The splits go with 80% for training and 20% for validation. And the Test data already split by the authors of this dataset.

## EDA
### Imbalance class
Some insights on the class distribution and check whether it's balanced or not
![Class dist](https://i.imgur.com/UWcpW9Q.png)

We can clearly see that class `20` & `22` less than the other classes, So we need to undersample it to make it balanced. We used undersampling to the minimum class number.


### Inspect image shapes
Check the images (width + height) so that in the preprocessing step of the images and before batching the dataset we need to see which numbers of width & height of majority of images:

![Images shape](https://i.imgur.com/EYSQcjo.png)

After inspecting the majority of numbers are between 0 — 1000 both (width + height), so for starter to set stone for first experiment also set baseline model i picked 300 * 300.


## Architectures
### Baseline architecture
the architecture of the baseline network used in this project are:
| Layer Name | Value |
| ---- | ---- | 
| data_augmentation_layer | 1.RandomFlip, 2.RandoZoom=0.1, 3.RandomRotation(0.2), 4.RandomTranslation(0.2, 0.2) |
| Conv2D | 1.filters= 64, 2.kernel=3 |
| MaxPool2D | default=2 |
| Dropout | drop_rate=0.2 |
| MaxPool2D | default=2 |
| Conv2D | 1.filters=128, 2. kernel=3 |
| MaxPool2D | default=2 |
| Flatten | default=None |
| Dense | units=128 |
| Dropout | drop_rate=0.2 |
| Dense | 24 output with softmax |

----
### Baseline killer architecture
CNN architecture V2 that beat the baseline architecture score and it did with 81% accuracy and 52% loss.
| Layer Name | Value |
| ---- | ---- | 
| Conv2D | Filters=32, kernel_size=5 |
| Conv2D | Filters=64, kernel_size=5 |
| MaxPool2D | default=2 |
| Conv2D | Filters=64, kernel_size=3 |
| MaxPool2D | default=2 |
| Flatten | default=None |
| Dense | units=256 |
| Dropout | drop_rate=0.5 |
| Dense | 24 output with softmax |


## Wandb(Weights & Biases) Experiment monitoring
Wandb a great MLOps platform for monitoring, experiment, tracking, versioning, evaluating model performance and sharing findings with your colleagues.

The experiments starts out of curiosity because of the tradeoffs of whether to shuffle the validation set or not. But in the long run it became useful to test it out along with different hyperparameters for batch sizes & image sizes.

The experiments goes with these values tested on 10% of data:
- Batch sizes: 16, 32, 64, 128.
- Image sizes: 30, 60, 80, 100.

Also these values looped with two main option:
- Shuffled validation.
- NOT Shuffled validation.

the format for the experiment names:
```python
ATSC_img-shape_("image size")_batch_("batch size")_shuffled
```
the last bit "shuffled" indicates that validation set is shuffled, if not it's not shuffled.

[Wandb experiment report unshuffled validation here](https://wandb.ai/joxtest/Arabic%20Traffic%20Classification%20Tests/reports/Loss-metrics-Accuracy-metrics-Unshuffled-Ver---VmlldzoyNTM0NzQ4?accessToken=65hev7berb8ommwby3u2lrqrposkk8wdqw5dmccpkq9mgu5mgal6znt3qdoitfz9)

[Wandb experiment report for Shuffled validation here](https://wandb.ai/joxtest/Arabic%20Traffic%20Classification%20Tests/reports/Loss-metrics-Accuracy-metrics-Shuffled-ver---VmlldzoyNTM0ODk0?accessToken=kyzufr2qecj2q3bi82fr6k39nvyk0o495em72qvbh2o6yofqh821lo0h89p0c79f)

For a little bit of those reports the key point here what am I want to get was the validation loss so I made a little bar chart for both Shuffled & unshuffled versions:

![Shuffled-version](https://github.com/Jo0xFF/Arabic-Traffic-Signs-Classification/blob/main/app/static/misc/Top10_val_loss_shuffled.png)

![Unshuffled-version](https://github.com/Jo0xFF/Arabic-Traffic-Signs-Classification/blob/main/app/static/misc/Top10_val_loss_not_shuffled.png)

Even though it's shown on the reports of this experiment that the shuffled version have less validation loss. I picked the unshuffled version because depending oh what [Jeremy Howard](https://youtu.be/9C06ZPF8Uuc?t=1496) said in his course  that validation set we don't need to shuffle it. but out of curiousity and the machine learning motto we need to experiment and see what result we get is it bad? or not. In the end i went with not shuffling the validation set.

**The experiment that i picked was: 30 for image shape and 16 batch size.**

## Loss metrics
### Baseline losses metrics
Check and see how the model performed during the training process for the baseline model.

![accuracy](https://i.imgur.com/gxO17Lr.png)

![loss](https://i.imgur.com/Cj8DC9P.png)

We can see that `val_loss` it's not overfitting and converging to good number but for future development we need to generalize the data more for high accuracy prediction to wild images.

---
### Baseline killer loss metrics
For `baseline killer model` it's way more optimized for validation loss.
![accuracy_baseline_killer](https://github.com/Jo0xFF/Arabic-Traffic-Signs-Classification/blob/main/app/static/misc/accuracy_baseline_killer.png)

![loss_baseline_killer](https://github.com/Jo0xFF/Arabic-Traffic-Signs-Classification/blob/main/app/static/misc/val_loss_baseline_killer.png)

This time with CNN improved it's good and more optimized with loss & accuracy. they near each other both the orange & blue line but not crossing each other in wide length which an indicators that's not overfitting and it has been generalized well.


## Confusion matrix
A good and visual way to compare the predicted labels with true labels (ground truth) in tabular way and it's flexible. The good predictions will form a line from top left to bottom right (diagonal line). The other boxes other than `diagonal line` will be FP (False positives), FN (False negative).

![Confusion_matrix](https://github.com/Jo0xFF/Arabic-Traffic-Signs-Classification/blob/main/app/static/misc/confusion_matrix_baseline_killer.png)

From this evaluation method we can interpret something which as we can see we have high confusion between "front or left" & "front or right" signs, It seems model get highly confused and challenging to differentiate between them.

Also, it happens again with ("left turn" & "right turn") + ("Narrow from left" & "Narrow from right") our model poorly predicting it. And there's slight confusion with predictions of "speed 40" sign.

Key point here: The signs which are similar to each other like `left turn, right turn` are most likely to be predicted wrong with `baseline killer model (CNN improved)`.


## F1 score
Lets check the F1 score, which is a metric that combines Recall + Precision together. And for short: 
- Recall: Model predicts 0's when it's 1
- Precision: Model predicts 1's when it's 0

That's for binary classification problems, but it's a little bit different for multi-class classification for more information on this: https://parasite.id/blog/2018-12-13-model-evaluation/

![F1 score](https://github.com/Jo0xFF/Arabic-Traffic-Signs-Classification/blob/main/app/static/misc/F1_score.png)

We can see here the bottom 6 classes are similar to each other in visual!. That's a challenge to discuss later on.


## Predictions on test dataset
### For Baseline architecture
Lets see how our model performs on the artificial test data

![test data](https://i.imgur.com/4bqnEth.png)

We can see that some of the labels the model gets it wrong and we need to improve this and generalize the model more.

---
### For Baseline killer architecture
Lets visualize the Baseline killer(CNN improved model V2) and see if it gets more correct predictions.

![test_data_cnn_improved](https://github.com/Jo0xFF/Arabic-Traffic-Signs-Classification/blob/main/app/static/misc/test_data_viz.png)

Well, as we can see 3 wrong predictions and it's fair enough for model with 52% loss rate.


## Deployment
With Flask + heroku to host as ML web app.


## Libraries used
- tensorflow-cpu==2.9.1
- Keras==2.6.0
- Pillow==9.2.0
- numpy==1.21.6
- protobuf==3.20.1
- Flask==2.1.3
- gunicorn==20.1.0

## Contribution
For contribution part feel free to do so, But could you please firstly open an issue first to discuss what to change or contribute then open a pull request.

# Dev logs
- 26 Aug 2022 - Add more metrics and visuals to the project. Add new model that beats our baseline model.
- 16 Aug 2022 - Add more Sections, pictures, visualization, badges to the readme file.


## Licences and Citations
All rights goes to their respected authors.
```
Latif, Ghazanfar; Alghazo, Jaafar; Alghmgham, Danyah A.; Alzubaidi, Loay (2020), “ArTS: Arabic Traffic Sign Dataset”, Mendeley Data, V1, doi: 10.17632/4tznkn45mx.1
```
check their paper: https://www.sciencedirect.com/science/article/pii/S1877050919321477

