import os
import rasterio
import numpy as np
import pickle
import csv
import sys
sys.path.append('/exports/eddie3_homes_local/sfraval/field-delineation/input-data/eopatches')
from eolearn.core import EOPatch
from fd.utils_plot import draw_true_color, draw_bbox, draw_vector_timeless, draw_mask
import matplotlib.pyplot as plt
import pprint

def display_eopatch_predictions(eop):
    tidx = 0  # select one timestamp 
    viz_factor = 3.5

    fig, axs = plt.subplots(figsize=(15, 5), ncols=3, sharey=True)
    #axs[0].imshow(viz_factor * my_eopatch_w_predictions.data['BANDS'][tidx][:500, :500, [2,1,0]]/10000)
    axs[0].imshow(viz_factor * eop.data['BANDS'][tidx][:200, :200, [2,1,0]]/10000)
    axs[0].set_title('RGB bands')
    #axs[1].imshow(my_eopatch_w_predictions.data['EXTENT_PREDICTED'][tidx].squeeze()[:500, :500], vmin=0, vmax=1, alpha=.2)
    axs[1].imshow(eop.data['EXTENT_PREDICTED'][tidx].squeeze()[:200, :200])
    axs[1].set_title('Extent')
    #axs[2].imshow(my_eopatch_w_predictions.data['BOUNDARY_PREDICTED'][tidx].squeeze()[:500, :500], vmin=0, vmax=1, alpha=.2)
    axs[2].imshow(eop.data['BOUNDARY_PREDICTED'][tidx].squeeze()[:200, :200])
    axs[2].set_title('Boundary')    
    plt.savefig('/exports/eddie3_homes_local/sfraval/field-delineation/EO_summary.jpg')

eopatch_fn='/exports/eddie3_homes_local/sfraval/field-delineation/input-data/eopatches/ESP_NAV43'

my_eopatch_w_predictions = None
if os.path.exists(eopatch_fn):
    my_eopatch_w_predictions = EOPatch.load(
        eopatch_fn
    )
    print(my_eopatch_w_predictions)
	
	
if my_eopatch_w_predictions is not None:
    display_eopatch_predictions(my_eopatch_w_predictions)

#check that softmax is applied to the estimated arrays, and the correct logits are used in the calculation of the loss. You can check this by looking at the two channels of the predicted arrays (extent/boundary) and verify they sum to 1.
print(my_eopatch_w_predictions.data['EXTENT_PREDICTED'] + my_eopatch_w_predictions.data['BOUNDARY_PREDICTED'])