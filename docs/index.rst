irp-jm4622
======================================

This repo includes the foundational code used in training and evaluating a ResNet-based model (built in PyTorch) for the downscaling of satellite/GCM-obtained AOD data. 

All code was originally run using the `JASMIN <https://jasmin.ac.uk>`_ GPU cluster.

Installation
------------

To install pre-requisites, from the base directory run:

``$ conda env create -f environment.yml``

``$ conda activate aod-ds``


Data
-----

The ``src/train.py`` script only supports MODIS AOD data in .hdf format. A small sample of processed training data in this format can be downloaded from `here <https://drive.google.com/drive/folders/1Ds927hV1pXdDC4uDqKiyBSZMj6qzEZ7q>`_. Evaluation is supported for any AOD data; the ``data/download_scipts`` folder contains example download scripts for obtaining CAMS GCM data for evaluation.


Classes and Functions
----------------------
.. automodule:: src.models.resnet
    :members:

.. automodule:: src.data.create_training_data
    :members:

.. automodule:: src.data.dataset
    :members:

.. automodule:: src.misc.aeronet_vs_cams
    :members:

.. automodule:: src.plotting
    :members:

.. automodule:: src.train
    :members:

.. automodule:: src.eval
    :members:

.. automodule:: src.utils
    :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents: