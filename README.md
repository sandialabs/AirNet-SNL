[![GitHub version](https://badge.fury.io/gh/sandialabs%2FAirNet-SNL.svg)](https://badge.fury.io/gh/dennis-j-lee%2Fasnl)
[![ActionStatus](https://github.com/sandialabs/AirNet-SNL/workflows/lint%20and%20test/badge.svg)](https://github.com/sandialabs/AirNet-SNL/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5192883.svg)](https://doi.org/10.5281/zenodo.5192883)

# AirNet-SNL: End-to-End Training of Iterative Reconstruction and Deep Neural Network Regularization for Sparse-Data XPCI CT

## Documentation

More information about the AirNet-SNL modules is available at [readthedocs](https://airnet-snl.readthedocs.io/en/latest/).

## Overview
AirNet-SNL is a Python package for computed tomography (CT) reconstruction. It implements a machine learning model to reconstruct 2D slices from 1D projections. The AirNet-SNL model is an end-to-end neural network that combines physics-based iterative reconstruction with convolutional neural networks. It accepts sparse data as input, including decimated angles or views. The library produces different image products, including absorption, dark field, and differential or integrated phase. For example, we have demonstrated AirNet-SNL on X-ray phase contrast imaging (XPCI). As a quick start, check out the examples in the documentation for training the model and running inference.

## Citing AirNet-SNL
If you find AirNet-SNL useful in your work, please consider citing

Software citation
-----------------

Dennis J. Lee. (2021). sandialabs/AirNet-SNL: First release of AirNet-SNL (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.5192883

Conference paper citation
-------------------------

D. J. Lee, J. Mulcahy-Stanislawczyk, E. Jimenez, D. West, R. Goodner, C. Epstein, K. Thompson, and A. L. Dagel, "AirNet-SNL: End-to-End Training of Iterative Reconstruction and Deep Neural Network Regularization for Sparse-Data XPCI CT," OSA Imaging and Applied Optics Congress, 2021.

## Copyright and License

AirNet-SNL is copyright through Sandia National Laboratories. The software is distributed under the Revised BSD License. See [copyright and license](https://github.com/sandialabs/AirNet-SNL/blob/master/LICENSE) for more information.
