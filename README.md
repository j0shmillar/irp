# irp
=======
## A CNN for the Spatial Downscaling of Global Aerosol Optical Depth (AOD)
[![tests](https://github.com/j0shmillar/irp/actions/workflows/tests.yml/badge.svg)](https://github.com/j0shmillar/irp/actions/workflows/tests.yml)
[![flake8](https://github.com/j0shmillar/irp/actions/workflows/flake8.yml/badge.svg)](https://github.com/j0shmillar/irp/actions/workflows/flake8.yml)
[![sphinx](https://github.com/j0shmillar/irp/actions/workflows/sphinx.yml/badge.svg)](https://github.com/j0shmillar/irp/actions/workflows/sphinx.yml)

This repo includes the foundational code used in training and evaluating a ResNet-based model (built in PyTorch) for the downscaling of satellite/GCM-obtained AOD data. All code was originally run using the [JASMIN](https://jasmin.ac.uk) GPU cluster.

Requirements
------------

To install pre-requisites, from the base directory run:
```  
$ conda env create -f environment.yml
$ conda activate aod-ds
```  

Data
-------------

The `src/data/create_training_data` script may be run via the command line to generate training data from MODIS AOD in .hdf format. The `data/download_scipts` folder contains example download scripts for obtaining CAMS GCM data. 

Instructions 
-------------

Both `src/train.py` and `src/eval.py` may also be run via the command line, on any properly formatted AOD data, with modification of the default options. See Docs (below) to the view default options, or use:

```  
$ python train.py/eval.py --help
```  

Docs
-------------

Clone repo and open `docs/html/index.html` in browser to view [documentation](https://github.com/ese-msc-2022/irp-jm4622/blob/main/docs/html/index.html).
