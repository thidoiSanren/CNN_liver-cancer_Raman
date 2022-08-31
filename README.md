# Overview

To classify different types of liver tissues using Raman spectra, a VGG-16 network-based CNN model was employed. A liver tissue Raman database was established with 50 spectra per tissue sample, and a total of 12,000 spectra were obtained from 120 pairs of liver tissue samples. The spectral data ranged from 500–2000 cm-1 with 889 one-dimensional float data. The model architecture consisted of 13 one-dimensional convolutional layers, 5 pooling layers, and 3 fully connected layers, utilizing small-scale convolution kernel stacking rather than large-scale convolution kernels to reduce the parameters required for calculations. Due to the data-driven nature of deep learning, the CNN model was built and successfully employed in the distinction between Raman spectra collected from hepatic carcinoma tissues and adjacent non-tumour tissues and the recognition of different hepatic pathological tissues, including different subtypes, tumour stages, and differentiations.

# System Requirements

## Hardware Requirements

The “CNN_liver-cancer_Raman” package requires a standard computer with enough RAM to run the program. If the computer has a Nvidia graphics card that supports CUDA, this will help speed up the execution of the program.

We have tested the program based on the computer with the following specs:

RAM: 16GB

CPU: Intel(R) Core (TM) i5-9300HF CPU @ 2.40GHz

Graphics card: NVIDIA GeForce GTX 1660 Ti

## Software Requirements

### OS Requirements

The package is tested on **Windows 10 20H2** operating systems.

### Python Dependencies

The IDE is PyCharm. The installation instructions on the official website are as follows (https://www.jetbrains.com/pycharm/download/other.html):

Take version 2021.3.3 as an example:

1. Run the PyCharm-2021.3.3.exe file that starts the Installation Wizard

2. Follow all steps suggested by the wizard. Please pay special attention to the corresponding installation options

This package depends on the Python scientific stack, and the versions used for testing are:

```
Python 3.9.9
pytorch 1.10.1
numpy 1.22.0
matplotlib 3.5.1
torchvision 0.11.2
tqdm 4.62.3
pandas 1.0.4
```

# Installation Guide

Python is necessary, which can be downloaded from the official website: [Welcome to Python.org](https://www.python.org/).

PyTorch can also be downloaded from the official website: [PyTorch](https://pytorch.org/). Please confirm its version to make sure it can work with the CUDA you used.

Other packages can be installed from PyPi:

```
pip3 install numpy
pip3 install matplotlib
pip3 install torchvision
...
```

It takes about 10 minutes to install these packages on the above computer.

# DEMO

**Demo data** contains spectral data collected from liver cancer and adjacent non-tumour tissue.

In demo data, data from cancer tissues were labelled as "1", while data from paracancer tissues were labelled as "0".

## How to use the package

1. Divide the data into training, validation, and test sets (corresponding to txt files "demo data_Raman liver_train.txt", "demo data_Raman liver_validate", "demo data_Raman liver_test" respectively).

2. Run the file "Raman_training.py" to get the trained model.

3. Run the file "Raman_testing.py" to get the accuracy.

## Results

An accuracy of about 90% could be got based on the demo data, which takes about 60 minutes to run 15 epochs by the above computer.

## License

This project is covered under the **MIT License**.
