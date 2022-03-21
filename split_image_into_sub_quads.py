import glob
import os
import subprocess
import rasterio
import shutil
import fiona
from fiona.crs import from_epsg
from shapely.geometry import mapping
from shapely.wkt import loads
import rasterio


def get_wkt(minx, miny, maxx, maxy):

    wkt = """POLYGON(({0} {1},{2} {3},{4} {5},{6} {7},{0} {1}))""".format(
        minx,
        miny,
        maxx,
        miny,
        maxx,
        maxy,
        minx,
        maxy
    )

    return wkt


def generate_grid(base_dst_folder, out_gpkg_fn):
    my_schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "int",
            "name": "str"
        }
    }

    my_driver = "GPKG"
    my_crs = from_epsg(32630)

    with fiona.open(out_gpkg_fn, "w", driver=my_driver, crs=my_crs, schema=my_schema) as my_collection:
        geoid = 1
        for root, folders, files in os.walk(base_dst_folder):
            for fn in files:
                if os.path.split(fn)[-1] == 'B02.tif':
                    region = os.path.split(root)[-1]
                    b2_tif_fn = os.path.join(root, fn)
                    with rasterio.open(b2_tif_fn) as ds:
                        rst_bounds = ds.bounds
                        xmin, ymin, xmax, ymax = rst_bounds.left, rst_bounds.bottom, rst_bounds.right, rst_bounds.top
                        rst_bounds_wkt = get_wkt(xmin, ymin, xmax, ymax)
                        my_collection.write({
                            "geometry": mapping(loads(rst_bounds_wkt)),
                            "properties": {
                                "id": geoid,
                                "name": region
                            }
                        })
                    geoid += 1


def generate_aoi_from_grid_gpkg(grid_gpkg_fn, out_geojson_fn):
    my_schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "int",
            "name": "str"
        }
    }

    my_driver = "GeoJSON"
    my_crs = from_epsg(32630)

    with fiona.open(out_geojson_fn, "w", driver=my_driver, crs=my_crs, schema=my_schema) as my_collection:
        geoid = 1
        with fiona.open(grid_gpkg_fn, 'r') as src:
            (xmin, ymin, xmax, ymax) = src.bounds
            rst_bounds_wkt = get_wkt(xmin, ymin, xmax, ymax)
            my_collection.write({
                "geometry": mapping(loads(rst_bounds_wkt)),
                "properties": {
                    "id": geoid,
                    "name": "30TWN"
                }
            })


def split_spanish_data_into_quads(base_dst_folder):
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
            dst_folder = os.path.join(
                base_dst_folder,
                '30TWN_{0}'.format(str(quad))
            )
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


def copy_date_files(base_dst_folder='/home/james/Work/FieldBoundaries/spain_split_up_data'):
    # cp ~/Work/FieldBoundaries/input_data_110322/tiffs/30TWN_1/userdata.json 30TWN_1/userdata.json etc
    pass


def main():
    #[1] split one of the tifs into 4 1100x1100 sub-images
    #split_spanish_data_into_quads()

    #[2] need to copy userdata.json from the tiff folder to each of the new sub-folders

    #[3] generate a geopackage that describes these
    # generate_grid(
    #     base_dst_folder='/home/james/Work/FieldBoundaries/spain_split_up_data',
    #     out_gpkg_fn='/home/james/Work/FieldBoundaries/spain_split_up_data/grid.gpkg'
    # )

    #[4] generate a GeoJSON file that describes the extent of the sub-images/grid based on grid created at [3]
    generate_aoi_from_grid_gpkg(
        grid_gpkg_fn='/home/james/Work/FieldBoundaries/spain_split_up_data/grid.gpkg',
        out_geojson_fn='/home/james/Work/FieldBoundaries/spain_split_up_data/aoi.geojson'
    )

    #[5] clip fields to out new 4 quad aoi
    # fields_32630.gpkg is /home/james/Work/FieldBoundaries/input_data_110322/fields.gpkg projected to 32630 from wgs84
    # james@pentlands:~/Work/FieldBoundaries/spain_split_up_data$ ogr2ogr -clipsrc aoi.geojson fields.gpkg fields_32630.gpkg
    # so: ~/Work/FieldBoundaries/spain_split_up_data/fields.gpkg is fields that reflect our AOI
    # note when run the above ogr2ogr cmd get a bunch of:
    # ERROR 1: TopologyException: Input geom 0 is invalid: Self-intersection at or near point (!?)


if __name__ == "__main__":
    main()





