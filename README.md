[![GitHub version](https://badge.fury.io/gh/sandialabs%2FAirNet-SNL.svg)](https://badge.fury.io/gh/dennis-j-lee%2Fasnl)
[![ActionStatus](https://github.com/sandialabs/AirNet-SNL/workflows/lint%20and%20test/badge.svg)](https://github.com/sandialabs/AirNet-SNL/actions)

# AirNet-SNL: End-to-End Training of Iterative Reconstruction and Deep Neural Network Regularization for Sparse-Data XPCI CT

## Overview
AirNet-SNL is a Python package for computed tomography (CT) reconstruction with applications in X-ray phase contrast imaging (XPCI) given sparse data.
It combines iterative reconstruction and convolutional neural networks with end-to-end training.
The model reduces streak artifacts from filtered back-projection with limited data, and it trains on randomly generated shapes.

## Introduction
Few-view computed tomography (CT) reduces radiation dose and speeds acquisition time.
For example, X-ray phase contrast imaging (XPCI) CT may decimate the angular range of measurements.
Various algorithms reconstruct undersampled images.
In iterative reconstruction, the choice of priors includes total variation and compressive sensing.
A deep learning method called AirNet unrolls an iterative reconstruction algorithm.

We present a deep learning image reconstruction technique for parallel beam CT that builds on AirNet.
Our model, called AirNet-SNL, makes three key contributions: first, it trains on randomly generated shapes; second, it removes skip connections to mitigate streak artifacts, which we observe with the random shape dataset; third, and most importantly, we demonstrate CT reconstructions for all three XPCI image products: absorption, dark field, and differential phase.

We refer the reader to our conference presentation at the OSA Imaging and Applied Optics Congress for more information.

## Citing AirNet-SNL
If you find AirNet-SNL useful in your work, please consider citing

D. J. Lee, J. Mulcahy-Stanislawczyk, E. Jimenez, D. West, R. Goodner, C. Epstein, K. Thompson, and A. L. Dagel, "AirNet-SNL: End-to-End Training of Iterative Reconstruction and Deep Neural Network Regularization for Sparse-Data XPCI CT," OSA Imaging and Applied Optics Congress, 2021.

## Copyright and License

AirNet-SNL is copyright through Sandia National Laboratories. The software is distributed under the Revised BSD License. See [copyright and license](https://github.com/sandialabs/AirNet-SNL/blob/master/LICENSE) for more information.