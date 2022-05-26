#######
#
#Script to prepare and download data for roi divided into a grid of features
#
#Downloads 3 stacked time periods for each S2 band and S1 output (first, mid and last)
######

import ee

#ee.Authenticate()

ee.Initialize()

geometry_grid = ee.FeatureCollection("users/Sfrav/download-grid_ESP_NAV")


#geometry = ee.FeatureCollection("users/Sfrav/Navarra")
start_date = '2021-3-1'
end_date = '2021-6-15'
mid_1 = '2021-4-1'
mid_2 = '2021-5-15'
CLOUD_FILTER = 50
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50
start_range = 0 #what ID to start downloading from 
end_range = 200 # can be overridden by grid square count below

############
#Cloud and shadow masking helper functions
def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    #not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img.addBands(is_cld_shdw)


##############
#Additional helper function to be used within the loop
def clip_geom(image):
  return image.clip(geometry.geometry())

def area(image):
    return image.set({'area': image.geometry().area()})

def area_nonzero(image): #https://spatialthoughts.com/2020/06/19/calculating-area-gee/
    zero = image.gt(0)
    areaImage = zero.multiply(ee.Image.pixelArea())
    areaNonzero = (areaImage.reduceRegion(
      reducer = ee.Reducer.sum(),
      geometry = image.geometry(), #can take geometry from higher scope?
      scale = 10,
      maxPixels = 1e10
    ))
    return image.set({'area': areaNonzero.get('B2')}) #

def format_date(image):
    date = ee.Date(image.get('system:time_start'))
    return image.set({'date': date})


def export_QA(image):
  return image.expression('QA == 0', {'QA': s2_collection.first().select('QA10')})
###############
#Additional helper functions for S1 data processing
def addRatio(image):
    date = ee.Date(image.get('segmentStartTime'))
    years = date.difference(ee.Date('1970-01-01'), 'year')

    #Create and add bands #!Clean - remove date, years and timeImage
    ratio = image.select('VV').divide(image.select('VH')).rename('VV/VH').copyProperties(image)

    timeImage = ee.Image(years).rename('t').copyProperties(image)
    return image.addBands(ratio).set({'system:time_start': image.get('segmentStartTime')})


#for 16 day composites
def filter_s1_comp(image):
    timeImage = image.metadata('system:time_start').rename('timestamp')
    timeImageMasked = timeImage.updateMask(image.mask().select(0))
    return image.addBands(timeImageMasked)


def smooth_s1(image):
    volume = ee.ImageCollection.fromImages(image.get('images'))#.copyProperties(image)
    return ee.Image(image).addBands(volume.mean())
    #.set({'system:time_start': image.get('segmentStartTime')})

############

geometry_count = geometry_grid.size().getInfo()

print(geometry_count)

end_range = geometry_count

#iterate through geometry_grid list and filter sentinel 2 image collection for each feature
i = start_range
for i in range(start_range, end_range): #geometry_count
  geometry = geometry_grid.filter('id == '+ str(i))
  grid_ref = geometry.aggregate_array('name').get(0).getInfo()
  #create buffered area to filter non overlapping tiles
  grid_square = geometry.geometry().buffer(-1).area() #@What are the units of buffer?
  

  s2_collection = (ee.ImageCollection('COPERNICUS/S2')
                  .filterBounds(geometry)
                  .filterDate(start_date, end_date)
                  #.filter(ee.Filter.stringContains('PRODUCT_ID','30TWN'))
                  # Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
                  )
  

  #print(s2_collection)
  s2_collection = s2_collection.map(clip_geom)
  s2_collection = s2_collection.map(area_nonzero)

  #print(s2_collection.aggregate_array('area').getInfo())
  #print(s2_collection.size().getInfo())

  #subset image collection to those fully covering the geometry - tested and works !fail with mid in NAV2
  s2_collection = s2_collection.filter(ee.Filter.greaterThanOrEquals('area', grid_square)) 
  #print(s2_collection.size().getInfo())
  #print(s2_collection.aggregate_array('area').getInfo())


  s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
          .filterBounds(geometry)
          .filterDate(start_date, end_date))

  #print(s2_cloudless_col)

  s2_collection = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
          'primary': s2_collection,
          'secondary': s2_cloudless_col,
          'condition': ee.Filter.equals(**{
              'leftField': 'system:index',
              'rightField': 'system:index'
          })
      }))


  s2_collection = s2_collection.map(add_cld_shdw_mask)

  first = s2_collection.first()
  last = s2_collection.sort('system:time_start', False)
  last = last.limit(1, 'system:time_start', False).first()


  s2_collection = (ee.ImageCollection('COPERNICUS/S2')
                    .filterBounds(geometry)
                    .filterDate(mid_1, mid_2)
                    #.filter(ee.Filter.inList('PRODUCT_ID', tiles))
                    # Pre-filter to get less cloudy granules.
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
                    )

                    
  s2_collection = s2_collection.map(clip_geom)
  s2_collection = s2_collection.map(area_nonzero)
  s2_collection = s2_collection.filter(ee.Filter.greaterThanOrEquals('area', grid_square))

  s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
          .filterBounds(geometry)
          #.filter(ee.Filter.inList('PRODUCT_ID', tiles))
          .filterDate(start_date, end_date))
          
  #s2_cloudless_col = s2_cloudless_col.map(area)        
  #s2_cloudless_col = s2_cloudless_col.filter(ee.Filter.greaterThanOrEquals('area', grid_square))

  s2_collection = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
          'primary': s2_collection,
          'secondary': s2_cloudless_col,
          'condition': ee.Filter.equals(**{
              'leftField': 'system:index',
              'rightField': 'system:index'
          })
      }))
      
  s2_collection = s2_collection.map(add_cld_shdw_mask)

  mid = s2_collection.first()



  ##
  # Combine S2 ready for export
  s2_collection3 = ee.ImageCollection([first, mid, last])
  
  s2_collection3 = s2_collection3.map(format_date)

  task = ee.batch.Export.image.toDrive(
    image=s2_collection3.select('B2').toBands(), #.clipToBoundsAndScale(geometry.geometry(), 2500, 2500)
    folder=grid_ref,
    description='B02',
    fileNamePrefix='B02',
    scale=10,
    region=geometry.geometry())
  task.start()
    

  task = ee.batch.Export.image.toDrive(
    image=s2_collection3.select('B3').toBands(),
    folder=grid_ref,
    description='B03',
    fileNamePrefix='B03',
    scale=10,
    region=geometry.geometry())
  task.start()
    
  task = ee.batch.Export.image.toDrive(
    image=s2_collection3.select('B4').toBands(),
    folder=grid_ref,
    description='B04',
    fileNamePrefix='B04',
    scale=10,
    region=geometry.geometry())
  task.start()

  task = ee.batch.Export.image.toDrive(
    image=s2_collection3.select('B8').toBands(),
    folder=grid_ref,
    description='B08',
    fileNamePrefix='B08',
    scale=10,
    region=geometry.geometry())
  task.start()

  #task = ee.batch.Export.image.toDrive(
  #  image = s1_s2_collection.select('VV/VH').toBands(),
  #  folder = grid_ref,
  #  description ='S1_ratio',
  #  fileNamePrefix = 'S1_ratio',
  #  scale = 10,
  #  region = geometry.geometry())
  #task.start() 
    
  #tmp to match sentinel-hub outputs
  task = ee.batch.Export.image.toDrive(
    image=s2_collection3.map(export_QA).toBands(), #s2_collection.first().expression('QA == 0', {'QA'=s2_collection.first().select('QA10')}),
    folder=grid_ref,
    description='dataMask',
    fileNamePrefix='dataMask',
    scale=10,
    region=geometry.geometry())
  task.start()

  task = ee.batch.Export.image.toDrive(
    image=s2_collection3.select(['cloudmask']).toBands(),
    folder=grid_ref,
    description='CLP',
    fileNamePrefix='CLP',
    scale=10,
    region=geometry.geometry())
  task.start()

  
  task = ee.batch.Export.table.toDrive(collection=s2_collection3, folder=grid_ref, fileNamePrefix='dates', selectors='PRODUCT_ID, date')
  task.start()
  i += 1

##@ Check progress at https:#code.earthengine.google.com/tasks OR earthengine task list
##@ Cancel all earthengine task cancel all
##@ Need approx 5 gb of storage for Navarra, Spain
##@ Will take ~16 hrs to download if no interuptions. Based on download time of 63 grid squares
##@Useful reference for gee to python conversion (found too late) https://github.com/gee-community/ee-js-to-python-syntax/blob/master/eejs2python/main.py
##! Todo: check task limit and wait if needed

##Transfer to cluster using wget. eg.
https://drive.google.com/drive/folders/1bUToVy0AR3yKyMzv-HKgcPaFujOOpgn6?usp=sharing
### wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bUToVy0AR3yKyMzv-HKgcPaFujOOpgn6?usp=sharing' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bUToVy0AR3yKyMzv-HKgcPaFujOOpgn6?usp=sharing"  && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bUToVy0AR3yKyMzv-HKgcPaFujOOpgn6' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=1bUToVy0AR3yKyMzv-HKgcPaFujOOpgn6" -O- && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bUToVy0AR3yKyMzv-HKgcPaFujOOpgn6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bUToVy0AR3yKyMzv-HKgcPaFujOOpgn6" -O "GEE" && rm -rf /tmp/cookies.txt
#unzip '*.zip'
#! Not working - find field-delination/input-data/tiffs -type d -name -exec cp field-delineation/input-data/tiffs/ESP_NAV119/userdata.json \;

##!?Quicker to use gee_tools? https://github.com/gee-community/gee_tools