#!/usr/bin/env python
# coding: utf-8

# # Field delineation

#  <div class="alert alert-block alert-info"><b>This notebooks includes the training and  prediction using  a neural network. 
#     If running over a smaller area a CPU-only machine can be used, however if running on larger areas, we recommend using a machine with a GPU setup to speed up the processing time for those steps.  </b> 

# To run this  notebook out of the box, access to an Amazon AWS S3 bucket is required. Furthermore, the download of the satellite imagery is done through [Sentinel Hub](https://www.sentinel-hub.com/). To run locally (or with existing data) some modifications of the code might be required. 
# 
# The following data is a prerequisite for running this notebook:
#   - **The georeferenced file with the area of interest** 
#   - **Field boundary ground-truth data is required for training** 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import logging

import os
import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from eolearn.core import EOPatch
from sentinelhub import SHConfig
from shapely.geometry import Polygon

from fs.copy import copy_file


from fd.scripts.download import batch_download
from fd.scripts.tiffs_to_eopatches import convert_tiff_to_eopatches
from fd.scripts.vector_to_raster import rasterise_gsaa
from fd.scripts.sampling import sample_patchlets
from fd.scripts.patchlets_to_npz import patchlets_to_npz_files
from fd.scripts.normalization import calculate_normalization_factors
from fd.scripts.k_folds_split import k_fold_split
from fd.scripts.train import train_k_folds
from fd.scripts.predict import run_prediction
from fd.scripts.postprocessing import run_post_processing
from fd.scripts.vectorization import vectorise
from fd.scripts.utm_zone_merging import merge_zones

from fd.utils import BaseConfig, prepare_filesystem
from fd.utils_plot import (draw_vector_timeless, 
                           draw_true_color, 
                           draw_bbox, 
                           draw_mask, 
                           get_extent
                          )
logging.getLogger().setLevel(logging.ERROR)


# # Setup credentials 

# The procedure assumes access to a AWS S3 bucket from which it loads and stores data. 

# In[3]:


BUCKET_NAME = ''
AWS_REGION = ''
AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''

SH_CLIENT_ID = ''
SH_CLIENT_SECRET = ''


# In[4]:


sh_config = SHConfig()

sh_config.sh_client_id = SH_CLIENT_ID
sh_config.sh_client_secret = SH_CLIENT_SECRET
sh_config.aws_secret_access_key = AWS_SECRET_ACCESS_KEY
sh_config.aws_access_key_id = AWS_ACCESS_KEY_ID

sh_config.save()


# In[5]:


base_config = BaseConfig(bucket_name=BUCKET_NAME, 
                         aws_region=AWS_REGION,
                         aws_access_key_id=AWS_ACCESS_KEY_ID, 
                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

filesystem = prepare_filesystem(base_config)


# In[6]:


PROJECT_DATA_ROOT = '' # !! Local folder where project related  files are/will be stored !!  
INPUT_AOI_FILEPATH = os.path.join(PROJECT_DATA_ROOT, 'aoi.geojson') 
GRID_PATH = os.path.join(PROJECT_DATA_ROOT, 'grid.gpkg')


# In[1]:


TIME_INTERVAL = ['YYYY-MM-DD', 'YYYY-MM-DD'] # Set the time interval for which the data will be downloaded  


# Folder configs 

# In[7]:


BATCH_TIFFS_FOLDER = '' # Location on the bucket where downloaded TIFF images will be stored


# # Download of S2 images

# The images are downloaded using Sentinel Hub [Batch API](https://docs.sentinel-hub.com/api/latest/api/batch/) which provides 
# a fast and scalable way to download images across large areas. The resulting GeoTiff files are downloaded directly to the AWS S3 storage. 

# **Define bucket folder where tiffs from Batch API will be stored.** 

# In[11]:


download_config = {
    "bucket_name": BUCKET_NAME,
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION,
    "sh_client_id": SH_CLIENT_ID,
    "sh_client_secret": SH_CLIENT_SECRET,
    "description": "Field Delineation Example",
    "aoi_filename": INPUT_AOI_FILEPATH,
    "grid_filename": GRID_PATH,
    "time_interval": TIME_INTERVAL, 
    "grid_definition": {
        "grid_id": 1, 
        "resolution":10, 
        "buffer": [50, 50]
    },
    "tiles_path": BATCH_TIFFS_FOLDER,
    "bands": ["B02", "B03", "B04", "B08", "dataMask", "CLP"],
    "maxcc": 0.7,
    "mosaicking_order": "leastRecent"
}


# In[9]:


batch_download(download_config)


# Check grid 

# In[10]:


grid = gpd.read_file(GRID_PATH)
aoi = gpd.read_file(INPUT_AOI_FILEPATH)


# In[11]:


fig, ax = plt.subplots(figsize=(10, 10))
grid.boundary.plot(ax=ax, color='r')
aoi.plot(ax=ax)
ax.legend()


# # Convert to EOPatches

# **Define folder where EOPatches will be stored.** 

# In[8]:


EOPATCHES_FOLDER = '' # Location on the bucket to which EOPatches will be saved.


# In[13]:


tiffs_to_eop_config = {
    "bucket_name": BUCKET_NAME,
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION,
    "grid_filename": GRID_PATH,
    "tiffs_folder": BATCH_TIFFS_FOLDER,
    "eopatches_folder": EOPATCHES_FOLDER,
    "band_names": ["B02", "B03", "B04", "B08"],
    "mask_name": "dataMask",
    "data_name": "BANDS",
    "is_data_mask": "IS_DATA",
    "clp_name": "CLP",
    "clm_name": "CLM",
    "max_workers": 6
}


# In[ ]:


convert_tiff_to_eopatches(tiffs_to_eop_config)


# Check if EOPatches have been written

# In[15]:


eops = filesystem.listdir(EOPATCHES_FOLDER)


# In[16]:


print(eops)


# Load a sample EOPatch to check the values 

# In[17]:


eop = EOPatch.load(os.path.join(EOPATCHES_FOLDER, eops[0]), filesystem=filesystem)


# In[18]:


eop


# In[19]:


tidx = 3 # select one timestamp between 0 and number of timestamps in the EOPatch
viz_factor = 2.5

fig, axs = plt.subplots(figsize=(15, 5), ncols=3, sharey=True)
axs[0].imshow(viz_factor * eop.data['BANDS'][tidx][..., [2,1,0]]/10000)
axs[0].set_title('RGB bands')
axs[1].imshow(eop.data['CLP'][tidx].squeeze()/255, vmin=0, vmax=1)
axs[1].set_title('Cloud probability')
axs[2].imshow(eop.mask['IS_DATA'][tidx].squeeze(), vmin=0, vmax=1)
axs[2].set_title('Valid data');


# # Add GSAA reference data to patches
# 
# This step adds GSAA reference data to each EOPatch in vector and raster format.
# 
# The following features are added to eopatches:
# 
#  * original vector data
#  * raster mask from pixelated vector data, 10m buffer
#  * boundary mask (buffered raster minus raster pixelated)
#  * normalised distance transform

# ## Requirement: Import shapefile to DB for faster processing

# GSAA data for Lithuania is a rather large shapefile (~2.4G), which poses several issues:
#   * multiprocessing the eopatches with EOExecutor explodes with too high RAM usage
#   * loading the dataset takes a lot of time (even-though it happens only once, the dataset is practically copied for each process with multiprocessing, which takes time)
#   * additionally, spatial index has to be build in order to make the process of extracting geometries for each eopatch fast
#   
# Instead, we set up a local postgre database (with postgis extension), imported shapefile into it, created spatial index, and created a simple EOTask that queries the db to get the geometries intersecting the eopatch bbox.
# 
# The following was done on amazon EC2 to install everything needed, set-up the db and import shapefile. If running this locally or on an another input geometry, make sure to adjust teh username, the CRSs and of course the paths to the files:
# 
# ```bash
# # install docker, add permissions to user ubuntu to run docker commands
# sudo apt install docker.io
# sudo service docker start
# sudo usermod -a -G docker ubuntu
# 
# #so that user's permissions get picked up
# sudo reboot 
# 
# #pull the docker image with everything ready (postgre, postgis, ...)
# docker pull kartoza/postgis
# 
# #run the container
# docker run --name "postgis" -p 25431:5432 -e POSTGRES_USER=niva -e POSTGRES_PASS=n1v4 -e POSTGRES_DBNAME=gisdb -e POSTGRES_MULTIPLE_EXTENSIONS=postgis,hstore -d -t kartoza/postgis
# 
# #some more things to install in order to import shp to db
# sudo apt install postgis
# 
# #there are some issues with EPSG 3346 (Lithuanian crs), so in order to bypass them, I have converted the geometries to WGS84, using ogr2ogr, so I had to install gdal
# sudo apt install gdal-bin
# 
# #convert to WGS84
# ogr2ogr -t_srs epsg:4326 -lco ENCODING=UTF-8 -f 'Esri Shapefile' gsaa_4326.shp bucket/Declared_parcels_2019_03_S4C.shp
# 
# #create sql for import
# shp2pgsql -s 4326 -I gsaa_4326.shp gsaa > gsaa.sql
# 
# #run the sql to import
# psql -h localhost -U niva -p 25431 -d gisdb -f gsaa.sql
# ```
# 
# Now the data is in db `gisdb` in table `gsaa`. 
# 
# 

# In[20]:


rasterise_gsaa_config = {
    "bucket_name": BUCKET_NAME,
    "aws_access_key_id": AWS_ACCESS_KEY_ID, 
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION,
    "database": "gisdb",
    "user": "niva",
    "password": "n1v4",
    "host": "localhost",
    "port": "25431",
    "crs": "epsg:4326",
    "grid_filename": GRID_PATH,
    "eopatches_folder": EOPATCHES_FOLDER,
    "vector_feature": ["vector_timeless", "GSAA_ORIGINAL"],
    "extent_feature": ["mask_timeless", "EXTENT"],
    "boundary_feature": ["mask_timeless", "BOUNDARY"],
    "distance_feature": ["data_timeless", "DISTANCE"],
    "height": 1100,
    "width": 1100,
    "buffer_poly": -10,
    "no_data_value": 0,
    "disk_radius": 2,
    "max_workers": 12
}


# In[ ]:


rasterise_gsaa(rasterise_gsaa_config)


# Check the contents of the EOPatches to see if adding reference data was succesfful. 

# In[22]:


eop = EOPatch.load(os.path.join(EOPATCHES_FOLDER, eops[0]), filesystem=filesystem, lazy_loading=True)
fig, ax = plt.subplots(ncols=2, figsize=(15, 15))
draw_true_color(ax[0], eop, time_idx=15, factor=3.5/10000, feature_name='BANDS', bands=(2, 1, 0), grid=False)
draw_bbox(ax[0], eop)
draw_vector_timeless(ax[0], eop, vector_name='GSAA_ORIGINAL', alpha=.3)

draw_true_color(ax[1], eop, time_idx=15, factor=3.5/10000, feature_name='BANDS', bands=(2, 1, 0), grid=False)
draw_bbox(ax[1], eop)
draw_mask(ax[1], eop, time_idx=None, feature_name='EXTENT', alpha=.3)


# # Sample Patchlets from EOPatches
# 
# 
# This part samples image chips out of the larger `EOPatches`. A maximum number of chips is sampled randomly from the `EOPatch`, depending on the fraction of reference `EXTENT` pixels. A buffer where patchlets are not sampled from can also be specified. Image chips containing only valid image data and a cloud coverage lower than a threshold are sampled.
# 

# In[23]:


PATCHLETS_FOLDER =  '' # Location on the bucket to which sampled patchlets will be saved. 


# In[24]:


sampling_patchlets_config = {
    "bucket_name": BUCKET_NAME, 
    "aws_access_key_id": AWS_ACCESS_KEY_ID, 
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION,
    "eopatches_location": EOPATCHES_FOLDER,
    "output_path": PATCHLETS_FOLDER,
    "sample_positive": True,
    "grid_definition_file": GRID_PATH,
    "area_geometry_file": INPUT_AOI_FILEPATH,
    "sampled_feature_name": "BANDS",
    "mask_feature_name": "EXTENT",
    "buffer": 50,
    "patch_size": 256,
    "num_samples": 10,
    "max_retries": 10,
    "fraction_valid": 0.4,
    "cloud_coverage": 0.05,
    "max_workers": 12
}


# In[ ]:


sample_patchlets(sampling_patchlets_config)


# **Check what was written, take a look at one of the created patchlets to see if the contents make sense**

# In[26]:


patchlets = filesystem.listdir(PATCHLETS_FOLDER)


# In[27]:


patchlet = EOPatch.load(os.path.join(PATCHLETS_FOLDER, patchlets[11]), filesystem=filesystem)


# In[28]:


patchlet.data['BANDS'].shape


# In[29]:


fig, ax = plt.subplots(ncols=2, figsize=(15, 15))
draw_true_color(ax[0], eop, time_idx=3, factor=3.5/10000, feature_name='BANDS', bands=(2, 1, 0), grid=False)
draw_mask(ax[1], eop, time_idx=None, feature_name='EXTENT', alpha=1, grid=False)


# # Create .npz files from patchlets
# 
# This steps creates a series of `.npz` files which join the data and labels sampled in patchlets from the previous iteration. A dataframe is created to keep track of the origin of the patchlets, namely which eopatch they come from and at which position they were sampled. This dataframe is later used for the cross-validation splits. 

# In[2]:


NPZ_FILES_FOLDER = ''  # Location on the bucket to which the NPZ files will be saved. 

#  Filepath to which the metadata dataframe will be saved. 
#  Needs to be a CSV file!  
METADATA_DATAFRAME = '<DIR_ON_BUCKET>/<FILENAME>.csv' 


# In[31]:


patchlets_to_npz_config = {
    "bucket_name": BUCKET_NAME,
    "aws_access_key_id": AWS_ACCESS_KEY_ID, 
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION, 
    "patchlets_folder": PATCHLETS_FOLDER,
    "output_folder": NPZ_FILES_FOLDER, 
    "output_dataframe": METADATA_DATAFRAME,
    "chunk_size": 10,
    "max_workers": 12
}


# In[ ]:


patchlets_to_npz_files(patchlets_to_npz_config)


# Make some sanity checks on the created NPZ files

# In[33]:


filesystem.listdir(NPZ_FILES_FOLDER)


# In[34]:


npzfile = np.load(filesystem.openbin(os.path.join(NPZ_FILES_FOLDER, 'patchlets_field_delineation_0.npz')))


# In[35]:


list(npzfile.keys())


# In[36]:


pd.read_csv(filesystem.openbin(METADATA_DATAFRAME)).head()


# # Calculate normalization stats per timestamp 
# 
# This step computes the normalisation factors per band per month for the `.npz` files obtained so far. 
# These normalisation factors are saved to `.csv` file and will be used at training and validation of the model.

# In[37]:


calculate_normalization_factors_config = {
    "bucket_name": BUCKET_NAME,
    "aws_access_key_id": AWS_ACCESS_KEY_ID, 
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION,
    "npz_files_folder": NPZ_FILES_FOLDER,
    "metadata_file": METADATA_DATAFRAME,
    "max_workers": 12
}


# In[ ]:


calculate_normalization_factors(calculate_normalization_factors_config)


# Check that normalization factors were added to the metadata dataframe

# In[39]:


pd.read_csv(filesystem.openbin(METADATA_DATAFRAME)).head()


# # Split patchlets for k-fold cross-validation
# 
# This step:
# 
#  * loads the dataframe with the patchlets descriptions
#  * splits the eopatches and corresponding patchlets into k-folds
#  * updates the info csv file with fold information

# In[9]:


N_FOLDS = 3 


# In[42]:


k_fold_split_config = {
    "bucket_name": BUCKET_NAME,
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION,
    "metadata_path": METADATA_DATAFRAME,
    "npz_folder": NPZ_FILES_FOLDER,
    "n_folds": N_FOLDS,
    "seed": 2,
    "max_workers": 12
}


# In[ ]:


k_fold_split(k_fold_split_config)


# Some sanity checks: 

# In[44]:


for fold in range(N_FOLDS):
    print(f'In Fold {fold+1}:')
    print(filesystem.listdir(os.path.join(NPZ_FILES_FOLDER, f'fold_{fold+1}')))


# # Train a ResUnetA model
# 
# This step  performs training of the `ResUnetA` architecture using the `.npz` files prepared in the previous steps:
# 
# This steps:
# 
#  * creates TensorFlow datasets in a k-fold cross-validation scheme, using the npz files previously created. The datasets allow manipulation and loading on the fly, to reduce RAM load and processing of large AOIs
#  * performs training of the k-fold models  
#  * test the models predictions on a validation batch
#  
# 
# This workflow can load the `.npz` files from the S3 bucket, or locally. The training will be faster if files are copied and loaded from local disk. Change the `npz_from_s3` flag in the data loader function.

# In[10]:


LOCAL_MODEL_FOLDER = os.path.join(PROJECT_DATA_ROOT, '') # Local path to the folder where models are saved 
S3_MODEL_FOLDER = '' #  Path to the bucket folder  models are saved 
CHKPT_FOLDER=None # Path to pretrained model if exists 


# In[37]:


train_k_folds_config = {
    "bucket_name": BUCKET_NAME,  
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION,
    "wandb_id": None, 
    "npz_folder": NPZ_FILES_FOLDER,
    "npz_from_s3": True,
    "num_parallel": 100,
    "metadata_path": METADATA_DATAFRAME, 
    "model_folder": LOCAL_MODEL_FOLDER,
    "model_s3_folder": S3_MODEL_FOLDER,
    "chkpt_folder": CHKPT_FOLDER,
    "input_shape": [256, 256, 4],
    "n_classes": 2,
    "batch_size": 8,
    "iterations_per_epoch": 1500,  # Change based on the size of the AOI
    "num_epochs": 30, 
    "model_name": "resunet-a",
    "reference_names": ["extent","boundary","distance"],
    "augmentations_feature": ["flip_left_right", "flip_up_down", "rotate", "brightness"],
    "augmentations_label": ["flip_left_right", "flip_up_down", "rotate"],
    "normalize": "to_medianstd",
    "n_folds": N_FOLDS,
    "fill_value": -2,
    "seed": 42,
    "wandb_project": "",
    "model_config": {
        "learning_rate": 0.0001,
        "n_layers": 3,
        "n_classes": 2,
        "keep_prob": 0.8,
        "features_root": 32,
        "conv_size": 3,
        "conv_stride": 1,
        "dilation_rate": [1, 3, 15, 31],
        "deconv_size": 2,
        "add_dropout": True,
        "add_batch_norm": False,
        "use_bias": False,
        "bias_init": 0.0,
        "padding": "SAME",
        "pool_size": 3,
        "pool_stride": 2,
        "prediction_visualization": True,
        "class_weights": None
    }
}


# In[ ]:


train_k_folds(train_k_folds_config)


# **A model is trained for each fold and then the weights of these models are averaged out to create 
# a single model that is applied to the whole AOI.**

# In[47]:


AVG_MODELS = [x for x in os.listdir(LOCAL_MODEL_FOLDER) if 'avg' in x] 
AVG_MODELS


# In[49]:


AVG_MODEL = AVG_MODELS[0]


# # Make prediction 

# In[50]:


prediction_config = {
    "bucket_name": BUCKET_NAME, 
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION,
    "grid_filename": GRID_PATH,
    "eopatches_folder": EOPATCHES_FOLDER,
    "feature_bands": ["data", "BANDS"],
    "feature_extent": ["data", "EXTENT_PREDICTED"],
    "feature_boundary": ["data", "BOUNDARY_PREDICTED"],
    "feature_distance": ["data", "DISTANCE_PREDICTED"],
    "model_path": S3_MODEL_FOLDER,
    "model_name": AVG_MODEL,
    "model_version": "v1",
    "temp_model_path": LOCAL_MODEL_FOLDER,
    "normalise": "to_meanstd",
    "height": 1128,
    "width": 1128,
    "pad_buffer": 14,
    "crop_buffer": 26,
    "n_channels": 4,
    "n_classes": 2,
    "metadata_path": METADATA_DATAFRAME,
    "batch_size": 1
}


# In[51]:


logging.getLogger().setLevel(logging.INFO)


# In[52]:


pred = run_prediction(prediction_config)


# In[ ]:


logging.getLogger().setLevel(logging.ERROR)


# In[80]:


eop = EOPatch.load(os.path.join(EOPATCHES_FOLDER, eops[4]), filesystem=filesystem)


# In[71]:


tidx = 12 # select one timestamp
viz_factor = 2.5

fig, axs = plt.subplots(figsize=(15, 5), ncols=3, sharey=True)
axs[0].imshow(viz_factor * eop.data['BANDS'][tidx][..., [2,1,0]]/10000)
axs[0].set_title('RGB bands')
axs[1].imshow(eop.data['EXTENT_PREDICTED'][tidx].squeeze(), vmin=0, vmax=1)
axs[1].set_title('Extent')
axs[2].imshow(eop.data['BOUNDARY_PREDICTED'][tidx].squeeze(), vmin=0, vmax=1)
axs[2].set_title('Boundary');


# In[73]:


tidx = 12  # select one timestamp 
viz_factor = 3.5

fig, axs = plt.subplots(figsize=(15, 10), ncols=2, sharey=True)
axs[0].imshow(viz_factor * eop.data['BANDS'][tidx][:200, :200, [2,1,0]]/10000)
axs[0].set_title('RGB bands')
axs[0].imshow(eop.data['EXTENT_PREDICTED'][tidx].squeeze()[:200, :200], vmin=0, vmax=1, alpha=.2)
axs[0].set_title('Extent')
axs[1].imshow(viz_factor * eop.data['BANDS'][tidx][:200, :200, [2,1,0]]/10000)
axs[1].set_title('RGB bands')
axs[1].imshow(eop.data['BOUNDARY_PREDICTED'][tidx].squeeze()[:200, :200], vmin=0, vmax=1, alpha=.2)
axs[1].set_title('Boundary')


# # Postprocessing 

# Predictions for each timestamp within the requested period have been saved to the EOPatches. We need to temporally merge these predictions to get one prediction for each area.
# 
# The following steps are executed:
# * merge predictions over a time interval using percentile statistics (median is used)
# * join the extent and boundary predictions derived from previous step 
# * iteratively smooth and upscale the joined prediction for a visually smoother output
# * export the resulting map as .tif to be used for vectorisation

# **Define folder where rasterized predictions will be saved to**

# In[11]:


RASTER_RESULTS_FOLDER = '' # Location on the bucket to which the per-eopatch tiff files will be saved. 


# In[78]:


postprocessing_config = {
    "bucket_name": BUCKET_NAME, 
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION,
    "grid_filename": GRID_PATH,
    "time_intervals": {"MAY": ["2021-05-01", "2021-05-31"]}, 
    "eopatches_folder": EOPATCHES_FOLDER,
    "tiffs_folder": RASTER_RESULTS_FOLDER,
    "feature_extent": ["data", "EXTENT_PREDICTED"],
    "feature_boundary": ["data", "BOUNDARY_PREDICTED"],
    "model_version": "v1",
    "max_cloud_coverage": 0.10,
    "percentile": 50,
    "scale_factor": 2,
    "disk_size": 2,
    "max_workers": 12 
}


# In[79]:


run_post_processing(postprocessing_config)


# # Creating vectors

# The following steps are executed to vectorise and spatially merge the vectors:
# * Create a weights.tiff file, based on upscaled tiff dimension and overlap size that is used to assign gradual weights to enable seemless merging across EOPatches. 
# * list tiffs that should be in the vrt (all tiffs should be from same UTM zone)
# * create vrt file
# * run process over whole area, split into small tiles:
#   * for each row run:
#      * extract small tiff file from vrt (gdal_transform)
#      * contour it (gdal_contour)
#      * merge extracted contours with existing merged tiles of the row
#   * (run rows in multiprocess mode to speed up processing)
#   * merge rows using the same approach
# 
# 

# In[12]:


vectorize_config = {
    "bucket_name": BUCKET_NAME, 
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION,
    "tiffs_folder": RASTER_RESULTS_FOLDER,
    "time_intervals": ["MAY"],
    "utms": ["32634"], # List all the different UTM zones within the AOI
    "shape": [4400, 4400],
    "buffer": [200, 200],
    "weights_file": os.path.join(PROJECT_DATA_ROOT, "weights.tiff"),
    "vrt_dir": PROJECT_DATA_ROOT,
    "predictions_dir": os.path.join(PROJECT_DATA_ROOT, "fd-predictions"),
    "contours_dir": os.path.join(PROJECT_DATA_ROOT, "fd-contours"),
    "max_workers": 8,
    "chunk_size": 500,
    "chunk_overlap": 10,
    "threshold": 0.6,
    "cleanup": True,
    "skip_existing": True,
    "rows_merging": True
}


# In[ ]:


vectorise(vectorize_config)


# Check the vector file 

# In[16]:


vectors = gpd.read_file(os.path.join(PROJECT_DATA_ROOT, 'fd-contours', 'merged_MAY_32634.gpkg'))


# In[18]:


fig, ax = plt.subplots(figsize=(15, 15))
vectors.plot(ax=ax)


# # Merge UTM zones

#  <div class="alert alert-block alert-info"><b>This step is only needed if the AOI spans multiple UTM zones</b> 

# The procedure outline is:
# * define geometries for two UTM zones and their overlap
# * load the two single-UTM-zone vector predictions
# * split them into parts: non-overlapping (completely within UTM zone) and overlapping
# * merge the overlaps by:
#   * transform them to single CRS (WGS84)
#   * spatial join of the overlapping geodataframes from the two zones
#   * finding geometries that do not overlap (and keeping them)
#   * unary_union-ize the polygons that intersect and merge them to the geometries from previous step
# * transform everything to resulting (common) CRS
# * clean up the results (remove geometries with area larger than X * max allowed size of polygon)
# * simplify geometries
# 

# In[27]:


utm_merging_config = {
    "bucket_name": BUCKET_NAME, 
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region": AWS_REGION,
    "grid_definition_file": GRID_PATH,
    "time_intervals": ["MAY"],
    "utms": ["32634", "32635"],
    "contours_dir": os.path.join(PROJECT_DATA_ROOT, "fd-contours"), 
    "resulting_crs": "",  # !! Choose an appropriate meter-based CRS  that covers the AOI !! 
    "max_area": 4153834.1, 
    "simplify_tolerance": 2.5, 
    "n_workers": 16,
    "overlap_buffer": -0.0001,
    "zone_buffer": 0.00001
}


# In[ ]:


merge_zones(utm_merging_config)

