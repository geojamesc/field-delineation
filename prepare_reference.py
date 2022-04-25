import argparse
import numpy as np
import fiona
from shapely.geometry import shape
import matplotlib.pyplot as plt
import pprint


def filter_reference_data(file_path, file, LU_col, out_file, area_threshold, ratio_threshold, land_use):
    #Open the file using fiona.
    with fiona.open(file_path + file) as f:
        
        #pprint.pprint(f.schema)
        #!print(np.unique(f[LU_col]))

        with fiona.open('input-data/' + out_file, 'w', driver = 'GPKG', crs = f.crs, schema = f.schema) as output:
            if land_use != "all":
                for feature in f:
                        if feature['properties'][LU_col] in land_use:
                            area = np.array([shape(feature['geometry']).area])
                            #width = np.array([shape(feature['geometry']).width for feature in f])
                            #height = np.array([shape(feature['geometry']).height for feature in f])
                            #ratio = np.array(width)/np.array(height)
                            if area > area_threshold: #and ratio > ratio_threshold:
                                output.write(feature)

            else:
                for i in range(feature.size):
                    area = np.array([shape(feature['geometry']).area])
                    #width = np.array([shape(feature['geometry']).width for feature in f])
                    #height = np.array([shape(feature['geometry']).height for feature in f])
                    #ratio = np.array(width)/np.array(height)
                    if area > area_threshold: #and ratio[i] > ratio_threshold:
                        output.write(feature)



if __name__ == "__main__":

#    parser = argparse.ArgumentParser(description = "Prepare reference data for use in the model.")

#    parser.add_argument("-f", "--file_path", help = "Path to the folder containing the reference data.", required = True)
#    parser.add_argument("-o", "--output_path", help = "Path to the folder to save the prepared reference data.", required = True)
#    parser.add_argument("-l", "--land_use", help = "Land use to filter by. Default is crops.", default = ["crops"])
#    parser.add_argument("-a", "--area_threshold", help = "Area threshold in m2 - whole number)
#    parser.add_argument("-r", "--ratio_threshold", help = "Ratio threshold. Default is 0.5.", default = 0.5)
#    parser.add_argument("-c", "--column_name", help = "Column name to filter by. Default is LU_Col.", default = "LU_Col")

#    args = parser.parse_args()
    filter_reference_data('reference-data/', 'fields.gpkg', 'USO21', 'fields.gpkg', area_threshold=2, ratio_threshold=10, land_use = ['TIERRA ARABLE'])
    





