import os 
import json
import logging
from functools import reduce
from datetime import datetime
from typing import Callable, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

from eoflow.models.metrics import MCCMetric
from eoflow.models.segmentation_base import segmentation_metrics
from eoflow.models.losses import JaccardDistanceLoss, TanimotoDistanceLoss

from eoflow.models.segmentation_unets import ResUnetA
from tqdm.auto import tqdm 

from fd.tf_viz_utils import ExtentBoundDistVisualizationCallback
from fd.training import TrainingConfig, get_dataset
from fd.utils import prepare_filesystem#
from fd.tf_model_utils import load_model_from_checkpoints

BUCKET_NAME = ''
AWS_REGION = ''
AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
SH_CLIENT_ID = ''
SH_CLIENT_SECRET = ''
PROJECT_DATA_ROOT = 'field-delineation/input-data'  # Local folder where project related  files are/will be stored !!
INPUT_AOI_FILEPATH = os.path.join(PROJECT_DATA_ROOT, 'aoi.geojson')
GRID_PATH = os.path.join(PROJECT_DATA_ROOT, 'grid.gpkg')
REFERENCE_DATA_FILEPATH = os.path.join(PROJECT_DATA_ROOT, 'fields.gpkg')
TIME_INTERVAL = ['2021-08-03', '2021-11-19']  # Set the time interval for which the data will be downloaded YYYY-MM-DD
EOPATCHES_FOLDER = os.path.join(PROJECT_DATA_ROOT, 'eopatches')  # Location on the bucket to which EOPatches will be saved.
BATCH_TIFFS_FOLDER = os.path.join(PROJECT_DATA_ROOT, 'tiffs') # Location on the bucket where downloaded TIFF images will be stored
PATCHLETS_FOLDER = os.path.join(PROJECT_DATA_ROOT, 'patchlets')  # Location on the bucket to which sampled patchlets will be saved.
NPZ_FILES_FOLDER = os.path.join(PROJECT_DATA_ROOT, 'patchlets_npz')  # Location on the bucket to which the NPZ files will be saved.
METADATA_DATAFRAME = os.path.join(PROJECT_DATA_ROOT, 'patchlet-info.csv')  # Filepath to which the metadata dataframe will be saved as a CSV
LOCAL_MODEL_FOLDER = os.path.join(PROJECT_DATA_ROOT, 'niva-cyl-models')  # Local path to the folder where models are saved
S3_MODEL_FOLDER = PROJECT_DATA_ROOT  # Path to the bucket folder  models are saved
N_FOLDS = 3  # number of folds to use for cross validation
RASTER_RESULTS_FOLDER = os.path.join(PROJECT_DATA_ROOT, 'results', 'Spain')  # Define folder where rasterized predictions will be saved to
MAX_WORKERS = os.cpu_count() - 2  # Try to avoid saturating all my cpu cores

CHKPT_FOLDER = None

config = {
        "bucket_name": BUCKET_NAME,
        "aws_access_key_id": AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
        "aws_region": AWS_REGION,
        "wandb_id": None,
        "npz_folder": NPZ_FILES_FOLDER,
        #"npz_from_s3": True,
        "npz_from_s3": False,  # TODO - check this is appropriate
        #"num_parallel": 100,
        "num_parallel": MAX_WORKERS,  # TODO - check this is appropriate
        "metadata_path": METADATA_DATAFRAME,
        "model_folder": LOCAL_MODEL_FOLDER,
        "model_s3_folder": S3_MODEL_FOLDER,
        "chkpt_folder": CHKPT_FOLDER,
        "input_shape": [256, 256, 4],
        "n_classes": 2,
        "batch_size": 2,
        #"iterations_per_epoch": 1500,  # Change based on the size of the AOI
        "iterations_per_epoch": 400,  # TODO what does this define?
        #"num_epochs": 30,
        "num_epochs": 25,  # TODO what does this define?
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
            "learning_rate": 0.005,
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


training_config = TrainingConfig(
        bucket_name=config['bucket_name'],
        aws_access_key_id=config['aws_access_key_id'], 
        aws_secret_access_key=config['aws_secret_access_key'],
        aws_region=config['aws_region'],
        wandb_id=config['wandb_id'], 
        npz_folder=config['npz_folder'],
        metadata_path=config['metadata_path'],
        model_folder=config['model_folder'],
        model_s3_folder=config['model_s3_folder'],
        chkpt_folder=config['chkpt_folder'],
        input_shape=tuple(config['input_shape']),
        n_classes=config['n_classes'],
        batch_size=config['batch_size'],
        iterations_per_epoch=config['iterations_per_epoch'], 
        num_epochs=config['num_epochs'],
        model_name=config['model_name'],
        reference_names=config['reference_names'],
        augmentations_feature=config['augmentations_feature'],
        augmentations_label=config['augmentations_label'],
        normalize=config['normalize'],
        n_folds=config['n_folds'],
        model_config=config['model_config'],
        fill_value=config['fill_value'],
        seed=config['seed']
    )

batch_size = 2

ds_folds = [get_dataset(training_config, fold=fold, augment=True, randomize=True,
                        num_parallel=100, npz_from_s3=False) 
            for fold in tqdm(range(1, 4))]


test_batch = next(iter(ds_folds[0].batch(batch_size)))


#Import model from checkpoint
model_1 = load_model_from_checkpoints('field-delineation/input-data/niva-cyl-models', 'resunet-a_fold-1_2022-05-14-18-22-58', ResUnetA, build_shape=dict(features=[None] + list(training_config.input_shape)))

avg_model = model = load_model_from_checkpoints('field-delineation/input-data/niva-cyl-models', 'resunet-a_avg_2022-05-16-11-40-33', ResUnetA, build_shape=dict(features=[None] + list(training_config.input_shape)))


#Predict on validation data
predictions = model_1.net.predict(test_batch[0]['features'].numpy())

#print(test_batch[0]['features'].numpy())
#print(predictions)

n_images = 2

fig, axs = plt.subplots(nrows=n_images, ncols=5, 
                        sharex='all', sharey='all', 
                        figsize=(15, 3*n_images))

for nb in np.arange(n_images):
    axs[nb][0].imshow(test_batch[0]['features'].numpy()[nb][...,[2,1,0]])
    axs[nb][1].imshow(predictions[0][nb][..., 1])
    axs[nb][2].imshow(predictions[1][nb][..., 1])
    axs[nb][3].imshow(predictions[2][nb][..., 1])
    axs[nb][4].imshow(test_batch[1]['extent'].numpy()[nb][..., 1])
    
plt.tight_layout()

plt.savefig('field-delineation/predict_val_1.jpg')


#Predict on validation using average model
predictions = avg_model.net.predict(test_batch[0]['features'].numpy())

n_images = 2

fig, axs = plt.subplots(nrows=n_images, ncols=5, 
                        sharex='all', sharey='all', 
                        figsize=(15, 3*n_images))

for nb in np.arange(n_images):
    axs[nb][0].imshow(test_batch[0]['features'].numpy()[nb][...,[2,1,0]])
    axs[nb][1].imshow(predictions[0][nb][..., 1])
    axs[nb][2].imshow(predictions[1][nb][..., 1])
    axs[nb][3].imshow(predictions[2][nb][..., 1])
    axs[nb][4].imshow(test_batch[1]['extent'].numpy()[nb][..., 1])
    
plt.tight_layout()

plt.savefig('field-delineation/predict_val_ave.jpg')