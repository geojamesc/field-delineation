import glob
import os
import subprocess
import rasterio


def split_spanish_data_into_quads():
    """
        take large 30TWN_1 spanish image and split into 4 smaller 1100x1100 sub-images
    """
    quad_coords = {
        1: {'xoff': 0, 'yoff': 0, 'xsize': 1100, 'ysize': 1100},
        2: {'xoff': 1100, 'yoff': 0, 'xsize': 1100, 'ysize': 1100},
        3: {'xoff': 0, 'yoff': 1100, 'xsize': 1100, 'ysize': 1100},
        4: {'xoff': 1100, 'yoff': 1100, 'xsize': 1100, 'ysize': 1100}
    }

    src_folder = '/home/james/Work/FieldBoundaries/input_data_110322/tiffs/30TWN_1'

    for src_tif_fn in glob.glob(os.path.join(src_folder, '*.tif')):
        for quad in quad_coords:
            dst_folder = os.path.join('/home/james/Desktop/30TWN_{0}'.format(str(quad)))
            dst_tif_fn = os.path.join(dst_folder, os.path.split(src_tif_fn)[-1])
            # gdal cmd to clip an image using pixel coord offsets from UL corner
            # we clip into 4 sub-quads
            cmd = 'gdal_translate -srcwin {0} {1} {2} {3} {4} {5}'.format(
                quad_coords[quad]['xoff'],
                quad_coords[quad]['yoff'],
                quad_coords[quad]['xsize'],
                quad_coords[quad]['ysize'],
                src_tif_fn,
                dst_tif_fn
            )
            print(cmd)
            subprocess.call(cmd, shell=True)

            # TODO - not sure if this is needed or if indeed doing so will break things later on but at least doing this
            #   means band names will be internally consistent

            # use rasterio to update band name descriptions
            # so in case of quad 3 i.e. T30TWN_3
            #
            # from: [
            # '20210831T105619_20210831T105909_T30TWN_B2',
            # '20210905T105621_20210905T110445_T30TWN_B2',
            # '20211119T110249_20211119T110407_T30TWN_B2'
            # ]
            #
            # to: [
            # '20210831T105619_20210831T105909_30TWN_3_B2',
            # '20210905T105621_20210905T110445_30TWN_3_B2',
            # '20211119T110249_20211119T110407_30TWN_3_B2'
            # ]

            with rasterio.open(dst_tif_fn, 'r+') as ds:
                old_descriptions = ds.descriptions
                new_descriptions = [i.replace('T30TWN', ''.join(['30TWN_', str(quad)])) for i in old_descriptions]
                ds.descriptions = tuple(
                    new_descriptions
                )

            with rasterio.open(dst_tif_fn, 'r') as ds:
                print(dst_tif_fn)
                print('\tUpdated Descriptions: ', ds.descriptions)


def main():
    split_spanish_data_into_quads()


if __name__ == "__main__":
    main()





