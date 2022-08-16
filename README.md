# Arabic traffic signs classification
![Banner](https://i.imgur.com/TJnTUTa.jpeg)

![python version](https://img.shields.io/badge/python-3.7.13-blue&?style=for-the-badge&logo=python&color=blue&labelColor=black)
![GitHub repo size](https://img.shields.io/github/repo-size/Jo0xFF/Arabic-Traffic-Signs-Classification?logo=Github&style=for-the-badge&color=red)
![ML Problem](https://img.shields.io/badge/Type%20of%20ML-Multi--class%20Classification-red&?style=for-the-badge&logo=TensorFlow&color=darkviolet)
![License](https://img.shields.io/badge/LICENSE-MIT-green&?style=for-the-badge)
![Github Commit](https://img.shields.io/github/last-commit/Jo0xFF/Arabic-Traffic-Signs-Classification?style=for-the-badge)
![Flask logo](https://img.shields.io/badge/Flask-black&?style=for-the-badge&logo=flask&color=black)
![Heroku logo](https://img.shields.io/badge/Heroku-Open%20in%20Heroku-black&?style=for-the-badge&logo=Heroku&logoColor=6762A6&labelColor=black&color=6762A6&link=https://arabic-traffic-signs-classify.herokuapp.com&link=https://arabic-traffic-signs-classify.herokuapp.com)


## Overview
Arabic traffic signs classification is one of the main components of autonomous cars. This project 
Developed for testing different CNN architectures and layers to seek more generalized predictions. The target of this project to classify 24 different traffic signs on the wild and get prediction accuracy higher than 80% for each class. 

## Data
the data where this project takes from: [Dataset page](https://data.mendeley.com/datasets/4tznkn45mx) all rights to their respected authors.
- "The dataset consists of 2,718 real captured images and 57,078 augmented images for 24 Arabic traffic signs. The images are captured from three connected cities (Khobar, Dammam and Dhahran) in the Eastern Province of Saudi Arabia. The newly developed dataset consisting of 2,718 real images is randomly partitioned into 80% percent training set (2,200 images) and 20% percent testing set (518) images"

## EDA
Some insights on the class distribution and check whether it's balanced or not
![Class dist](https://i.imgur.com/UWcpW9Q.png)

We can clearly see that class `20` & `22` less than the other classes, So we need to undersample it to make it balanced. We used undersampling to the minimum class number.


### Inspect image shapes
Check the images (width + height) so that in the preprocessing step of the images and before batching the dataset we need to see which numbers of width & height of majority of images:

![Images shape](https://i.imgur.com/EYSQcjo.png)


## Architecture
the architecture of the network used in this project are:
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

## Loss Metrics
Check the metrics and see how the model performed during the training process

![accuracy](https://i.imgur.com/gxO17Lr.png)

![loss](https://i.imgur.com/Cj8DC9P.png)

We can see that `val_loss` it's not overfitting and converging to good number but for future development we need to generalize the data more.

## Predictions on test dataset
Lets see how our model performs on the artificial test data

![test data](https://i.imgur.com/4bqnEth.png)

We can see that some of the labels the model gets it wrong and we need to improve this and generalize the model more.


## Deployment
With Flask + heroku to host as ML web app.

## Libraries used
- tensorflow-cpu==2.9.1
- Keras==2.6.0
- Pillow==9.2.0
- numpy-1.21.6
- protobuf-3.20.1
- Flask==2.1.3
- gunicorn-20.1.0


# Dev logs
- 16 Aug 2022 - Add more Sections, pictures, visualization, badges to the readme file.


## Licences and Citations
All rights goes to their respected authors.
```
Latif, Ghazanfar; Alghazo, Jaafar; Alghmgham, Danyah A.; Alzubaidi, Loay (2020), “ArTS: Arabic Traffic Sign Dataset”, Mendeley Data, V1, doi: 10.17632/4tznkn45mx.1
```
check their paper: https://www.sciencedirect.com/science/article/pii/S1877050919321477

