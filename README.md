# Anomaly Detection method based on CW-SSIM applied to Autoencoders
## Description
This repository contains the Anomaly Detection method based on CW-SSIM applied to Autoencoders of the Andrea Bionda's Politecnico di Milano Master Thesis: *Pixelwise Anomaly Detection exploiting Steerable Filters based methods*

## Usage
Execute the software 
* ACTION: action to perform, one of: 'training' or 'evaluation'
* FILE: configuration file path

```sh
cd Anomaly_Detection_CWSSIM
python Main.py -a ACTION -f FILE 
```
It is possible to configure the training parameters in [TrainingParameters.ini](Configuration/TrainingParameters.ini) and the evaluation parameters in [EvaluationParameters.ini](Configuration/EvaluationParameters.ini). In order to replicate the experiments, it is possible to extract the pretrained autoencoder weigths file from [weights.zip](Weigths/weights.zip) and write its path in the evaluation parameters file.


## Installation
Clone and install: 
```sh
git clone https://github.com/AndreaBiondaPolimi/Anomaly_Detection_CWSSIM.git
cd Anomaly_Detection_CWSSIM
pip install -r requirements.txt
```

## Requirements
* opencv-python==4.1.1.26
* numpy>=1.16.4
* scipy==1.4.1
* matplotlib==3.1.1
* tensorflow-gpu==2.1.0
* scikit-image>=0.16.2
* scikit-learn==0.21.3
* albumentations==0.4.5
