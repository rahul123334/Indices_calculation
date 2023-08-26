#!/usr/bin/env python
# coding: utf-8
# author : Rahul Kumar


import warnings
warnings.filterwarnings("ignore")

import ee
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import box, shape
from matplotlib import colors
from matplotlib.patches import Patch
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta





ee.Initialize()




PATH_INFO = "/home/satyukt/Desktop/RAHUL/output/info.csv"
OUTH_PATH = "/home/satyukt/Desktop/RAHUL/output/NDVI_S2_L8_L9/"
info_df = pd.read_csv(PATH_INFO)




# generating custom colormap
levels = [0, 0.2, 0.4, 0.6, 0.8, 1]
clrs = [(0, "#ff0000"), 
        (0.13, "#e9331f"), 
        (0.26, "#fe591e"), 
        (0.39, "#dcde4c"), 
        (0.52, "#ffff00"), 
        (0.653846, "#94f25d"), 
        (0.78, "#3eca6a"), 
        (0.9, "#006d2c"), 
        (1, "#055005")]
legend_elements = [Patch(facecolor="#055005", edgecolor='black',label='Very High'), 
                   Patch(facecolor="#3eca6a", edgecolor='black',label='High'), 
                   Patch(facecolor="#ffff00", edgecolor='black',label='Medium'), 
                   Patch(facecolor="#fe591e", edgecolor='black',label='Low'), 
                   Patch(facecolor="#ff0000", edgecolor='black',label='Very Low')]
cmap = colors.LinearSegmentedColormap.from_list('rsm', clrs, N=256)




def wrapper(farm, farm_id) : 
    def ndviS2image(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    def landsatndvi(image):
        ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    def maskLandsatClouds(image):
        qa = image.select('QA_PIXEL')
        cloudMask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 5).eq(0))
        return image.updateMask(cloudMask)


    # SCL band cloud mask
    def sclCloudMask(image):
        ndvi = image.select('NDVI')
        crs = (ndvi.projection()).crs()
        scl = image.select('SCL')
        reScl = scl.resample('bilinear').reproject(crs = crs, scale = 10)
        mask = reScl.gt(3) and reScl.lte(7)
        # mask = mask.lte(7) # values which are not cloud
        maskedNdvi = ndvi.updateMask(mask)
        return ee.Image(maskedNdvi)

    # def reduce_resolution(image) :
    #     crs = (image.select('NDVI').projection()).crs()
    #     return image.reproject(crs = crs, scale = scale_res)

    # create sample rectangles
    def mapRectangle(image) :
        return ee.Image(image).sampleRectangle(region = AOI, defaultValue = float(-1))

    def get_arrays(feature) :
        arr = np.array(feature['properties']['NDVI']).astype(float)
        arr[arr < 0] = np.nan

        return arr

    get_s2_dates = np.vectorize(lambda x : x['id'][:8])
    get_landsat_dates = np.vectorize(lambda x : x['id'][-8:])

    crs = {'init': 'epsg:4326'}
    buff = 0.00008
    extent = shape(farm).bounds
    x_axis = [extent[0]-buff, extent[2]+buff]
    y_axis = [extent[1]-buff, extent[3]+buff]
    polygon = gpd.GeoDataFrame(index = [0], 
                                crs = crs, 
                                geometry = [shape(farm)]) 
    polygon_bound = gpd.GeoDataFrame(index = [0], 
                                        crs = crs, 
                                        geometry = [box(*shape(farm).bounds, 
                                                    ccw = True)])
    polygon_bound_geom = polygon_bound.buffer(buff).geometry
    polygon_bound_buff = gpd.GeoDataFrame(geometry = polygon_bound_geom)
    diff_poly = polygon.overlay(polygon_bound_buff, 
                                how = 'symmetric_difference')
    
    sdate = (datetime.now() - timedelta(days = 365)).strftime('%Y-%m-%d')
    edate = (datetime.now()).strftime('%Y-%m-%d')

    
  
    cords =  farm['coordinates'][0]
    AOI = ee.Geometry.Polygon(cords)

    s2dataset = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterDate(sdate, edate)
                .filterBounds(AOI)
                .map(ndviS2image)
                .map(sclCloudMask)
                .select('NDVI')
                .map(mapRectangle))

    l9dataset = (ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA")
                .filterBounds(AOI)
                .filterDate(sdate, edate))

    
    l8dataset = (ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
                .filterBounds(AOI)
                .filterDate(sdate, edate))

    
    landsat_merge_dataset = l9dataset.merge(l8dataset)
    landsat_dataset = (landsat_merge_dataset
                        .map(maskLandsatClouds)
                        .map(landsatndvi)
                        .select('NDVI')
                        .map(mapRectangle))
            
    s2features = s2dataset.getInfo()['features']
    s2_dates = get_s2_dates(s2features)
    landsat_features = landsat_dataset.getInfo()['features']
    landsat_dates = get_landsat_dates(landsat_features)
    merged_dates = np.concatenate((s2_dates, landsat_dates))
    unique_dates = np.unique(merged_dates)

    print(len(unique_dates))
    feature_arrays = []
    for date in unique_dates :
        if date in s2_dates :
            feature_arrays.append(get_arrays(s2features[np.where(s2_dates == date)[0][0]]))
        elif (date in landsat_dates) and (date not in s2_dates):
            feature_arrays.append(get_arrays(landsat_features[np.where(landsat_dates == date)[0][0]]))

        else :
            pass

    fig, ax = plt.subplots(figsize=(5,5), facecolor = "white")
    im = ax.imshow(feature_arrays[0], 
                    cmap = cmap, 
                    vmin = 0, 
                    vmax = 1, 
                    extent = [extent[0], extent[2], 
                                extent[1], extent[3]])
    diff_poly.plot(ax = ax, facecolor = "white")
    polygon.plot(ax = ax, facecolor = 'none', edgecolor = "black")
    ax.axis('off')
    plt.colorbar(mappable = im, 
                    orientation="vertical", 
                    ticks = levels, 
                    fraction = 0.03)
    plt.legend(handles=legend_elements, 
                loc='lower center', 
                ncol = 5, bbox_to_anchor = (0.5,0), 
                bbox_transform = plt.gcf().transFigure,  
                prop = {'size': 8})
    ax.set_xlim(x_axis)
    ax.set_ylim(y_axis)

    plt.subplots_adjust(bottom = 0.1, right = 0.8, 
                        top = 0.9)

    for index, date in enumerate(unique_dates) :
        if ~np.isnan(feature_arrays[index]).all() and date in s2_dates:
            s2_date = s2_dates[np.where(s2_dates == date)[0][0]]
            fig.suptitle(f"CURRENT NDVI IMAGES\n{s2_date}", 
                    ha = 'center', fontsize = 12, 
                    fontweight = 'bold')
            im.set_data(feature_arrays[index])
            fig.savefig(f"{OUTH_PATH}{farm_id}_S2_{s2_date}.png")
            print(f'S2_{s2_date}')
        elif ~np.isnan(feature_arrays[index]).all() and date in landsat_dates and date not in s2_dates:
            l9_date = landsat_dates[np.where(landsat_dates == date)[0][0]]
            fig.suptitle(f"CURRENT NDVI IMAGES\n{l9_date}", 
                    ha = 'center', fontsize = 12, 
                    fontweight = 'bold')
            im.set_data(feature_arrays[index])
            fig.savefig(f"{OUTH_PATH}{farm_id}_L_{l9_date}.png")
            print(f'l9_{l9_date}')
        else :
            pass
            
    fig.clf()



# farm_id = #####
index = info_df.loc[info_df['farm_id'] == farm_id].index[0]
farm = eval(info_df.loc[index].polyinfo)['geo_json']['geometry']
wrapper(farm, farm_id)

