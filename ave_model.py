#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#
#####
# Load models from disk
#####

import os 
import sys
import json
import logging
import argparse
import shutil
from datetime import datetime
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow import keras

from fs.copy import copy_dir
from tqdm.auto import tqdm 

from eoflow.models.segmentation_base import segmentation_metrics
from eoflow.models.losses import TanimotoDistanceLoss

from eoflow.models.segmentation_unets import ResUnetA

from fd.training import TrainingConfig, get_dataset, initialise_model, initialise_callbacks
from fd.utils import prepare_filesystem, LogFileFilter
from fd.tf_model_utils import load_model_from_checkpoints


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(LogFileFilter())
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

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
N_FOLDS = 2 # number of folds to use for cross validation
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

#ds_folds = [get_dataset(training_config, fold=fold, augment=True, randomize=True,
#                        num_parallel=100, npz_from_s3=False) 
#            for fold in tqdm(range(1, 3))]


#test_batch = next(iter(ds_folds[0].batch(batch_size)))


LOGGER.info('Create K TF datasets')

ds_folds = [get_dataset(training_config, fold=fold, augment=True, randomize=True,
                        num_parallel=100, npz_from_s3=False) 
            for fold in tqdm(range(1, 3))]

    #ds_folds = [get_dataset(training_config, fold=fold, augment=True, randomize=True,
    #                        num_parallel=config['num_parallel'], npz_from_s3=False)
    #            for fold in tqdm(range(1, training_config.n_folds + 1))]

#model_1 = load_model_from_checkpoints('field-delineation/input-data/niva-cyl-models', 'resunet-a_fold-1_2022-05-14-18-22-58', ResUnetA, build_shape=dict(features=[None] + list(training_config.input_shape)))
#model_2 = load_model_from_checkpoints('field-delineation/input-data/niva-cyl-models', 'resunet-a_avg_2022-05-16-11-40-33', ResUnetA, build_shape=dict(features=[None] + list(training_config.input_shape)))
#model_3 = load_model_from_checkpoints('field-delineation/input-data/niva-cyl-models', 'resunet-a_avg_2022-05-16-11-40-33', ResUnetA, build_shape=dict(features=[None] + list(training_config.input_shape)))


models = []
    #model_paths = []

LOGGER.info('Create model and load weights')
    #filesystem = prepare_filesystem(training_config)

model_paths = [item for item in os.listdir(f'{training_config.model_folder}') if os.path.isdir(os.path.join(f'{training_config.model_folder}', item))]

for path in model_paths:

    model = load_model_from_checkpoints('field-delineation/input-data/niva-cyl-models', os.path.basename(os.path.normpath(path)), ResUnetA, build_shape=dict(features=[None] + list(training_config.input_shape)))
    models.append(model)


LOGGER.info('Prepare for averaging models')
for model_path in model_paths:
    model_name = os.path.basename(model_path)
    #filesystem.makedirs(f'{training_config.model_s3_folder}/{model_name}', recreate=True)
    m_pth = f'{training_config.model_s3_folder}/{model_name}'
    if not os.path.exists(m_pth):
        os.makedirs(m_pth)
    else:
        shutil.rmtree(m_pth)
        os.makedirs(m_pth)

        # copy_dir(training_config.model_folder,
        #          f'{model_name}',
        #          filesystem,
        #          f'{training_config.model_s3_folder}/{model_name}')

    copy_dir(training_config.model_folder,
             f'{model_name}',
             f'{training_config.model_s3_folder}',
             f'{model_name}')

LOGGER.info('Create average model')

    #model = initialise_model(training_config, chkpt_folder=training_config.chkpt_folder)
weights = [model.net.get_weights() for model in models]

avg_weights = list()
for weights_list_tuple in zip(*weights):
    avg_weights.append(np.array([np.array(weights_).mean(axis=0) 
                            for weights_ in zip(*weights_list_tuple)]))

avg_model = initialise_model(training_config)
avg_model.net.set_weights(avg_weights)

now = datetime.now().isoformat(sep='-', timespec='seconds').replace(':', '-')
model_path = f'{training_config.model_folder}/{training_config.model_name}_avg_{now}'

LOGGER.info('Save average model to local path')
if not os.path.exists(model_path):
    os.makedirs(model_path)

checkpoints_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')
with open(f'{model_path}/model_cfg.json', 'w') as jfile:
    json.dump(training_config.model_config, jfile)
avg_model.net.save_weights(checkpoints_path)

      
#left_out_fold = testing_id[0]+1

#    LOGGER.info(f'Evaluating model on left-out fold {left_out_fold}')
#    model = models[testing_id[0]]
#    model.net.evaluate(ds_folds[testing_id[0]].batch(training_config.batch_size))

#    LOGGER.info(f'Evaluating average model on left-out fold {left_out_fold}')
#    avg_model.net.evaluate(ds_folds[testing_id[0]].batch(training_config.batch_size))
#    LOGGER.info('\n\n')


#if __name__ == '__main__':
    #LOGGER.info(f'Reading configuration from {args.config}')

    #parser = argparse.ArgumentParser(description="Train models in a k-fold cross-validation.\n")

    #parser.add_argument(
    #    "--config", 
    #    type=str, 
    #    help="Path to config file with k-fold training parameters", 
    #    required=True
   # )
    #args = parser.parse_args()


    #with open(args.config, 'r') as jfile:
    #    cfg_dict = json.load(jfile)

#    ave_model_checkpoint(train_k_folds_config)
