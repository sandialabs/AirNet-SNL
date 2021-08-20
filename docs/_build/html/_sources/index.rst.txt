Welcome to AirNet-SNL's documentation!
======================================

Introduction
============

AirNet-SNL is a Python package for computed tomography (CT) reconstruction.
It implements a machine learning model to reconstruct 2D slices from 1D projections.
The AirNet-SNL model is an end-to-end neural network that combines physics-based iterative reconstruction with convolutional neural networks.
It accepts sparse data as input, including decimated angles or views.
The library produces different image products, including absorption, dark field, and differential or integrated phase.
For example, we have demonstrated AirNet-SNL on X-ray phase contrast imaging (XPCI).
As a quick start, check out the examples for training the model and running inference.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   examples
   references
   whatsnew
   contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
