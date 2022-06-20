# NIVA - Automatic field delineation

This repo contains code to generate automatic contours for agricultural parcels,
given Sentinel-2 images. This code has been used ot generate contours for Lithuania, Navarra, Spain 
and the province of Castilla y Leon.

This fork allows the workflow to be run locally or on a private cluster.
You can find more information about the original project in the blog post [Parcel Boundary Detection for CAP](https://medium.com/sentinel-hub/parcel-boundary-detection-for-cap-2a316a77d2f6). 


## Introduction


The original sub-project was part of the ["New IACS Vision in Action” --- NIVA](https://www.niva4cap.eu/) project that delivers a suite of digital solutions, e-tools and good practices for e-governance and initiates an innovation ecosystem to support further development of IACS that will facilitate data and information flows.

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 842009.

Please visit the [website](https://www.niva4cap.eu) for further information. A complete list of the sub-projects made available under the NIVA project can be found on [gitlab](https://gitlab.com/nivaeu/)


### Installation

The `fd` Python 3.5+ package allows to execute the end-to-end field delineation workflow.

To install the `fd` package, clone locally the repository, and from within the repository, run the following commands:

```bash
pip install -r requirements.txt
python setup.py install --user
```

In addition, the field delineation workflow uses the following:

 * Docker containers;
 * `psql` PostgreSQL database; 
 * `gdal` geospatial processing library, version >2.4.0. 

The numbered notebooks showcase how to execute the end-to-end workflow.


## Content

This repository has the following content:

 * `fd`: modules implementing each part of the workflow;
 * `input-data`: folder storing the file defining the AOI and the consequent grid definition file;
 * `notebooks`: folder storing the example notebook to execute the end-to-end workflow.

### End2End Execution

The field delineation workflow has been designed to scale to large AOIs, by downloading data quickly and efficiently, 
and by parallelizing execution of pipelines over the tiled data.

[The End2End notebook](./notebooks/field-delineation-end2end.ipynb) showcases the entire procedure to reproduce the entire end-to-end workflow.
The following steps are executed: 

 * `Data download`: downloading the Sentinel-2 images (B-G-R-NIR) using Sentinel-Hub Batch API; 
 * `Conversion of tiffs to patches`: converts the downloaded tiff files into `EOPatches` (see [`eo-learn`](https://eo-learn.readthedocs.io/en/latest/)), 
   and computes cloud masks from cloud probabilities;
 * `Vector to raster`: adds reference vector data from a database to `EOPatches` and creates reference masks used 
   for training of the model;
 * `Patchlets sampling`: sample `EOPatches` into smaller `256x256` patchlets for each cloud-free time-stamp. The 
   sampling can be done for positive and negative examples separately;
 * `Patchlets to npz files`: the sampled patchlets are chunked and stored into multiple `.npz` files, allowing 
   to efficiently access the data during training;
 * `Create normalization stats`: compute normalisation factors for the S2 bands (e.g. B-G-R-NIR) per month. These 
   factors will be used to normalise the data before training and evaluation;
 * `Patchlets split into k-folds`: split patchlets into K-folds, allowing to perform a robust cross-validation of the models;
 * `Train model from cached npz`: train k-models, one for each left out fold. The [`ResUnet-a` architecture](https://www.sciencedirect.com/science/article/abs/pii/S0924271620300149) 
   implemented within [`eo-flow`](https://github.com/sentinel-hub/eo-flow) is used as model. A single model can be 
   derived by averaging the weights of the k-fold models; 
 * `Predict eopatches`: use the trained models to predict parcel boundary probabilities for the entire dataset;
 * `Post process predictions`: merge the predictions temporally and combine the predicted extent and boundary 
   probabilities. A time interval can be specified over which the predictions are temporally aggregated;
 * `Create vectors`: vectorise the combined field delineation probabilities; 
 * `Utm zone merging`: combine spatially vectors from multiple UTM zone if applicable.
