// this script can be run on Google Earth Engine Code Editor
// https://code.earthengine.google.com

var countries = ['KE', 'MX'];
for (var i = 0; i < 2; i++) {

  // area of interest
  var aoi = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
              .filter(ee.Filter.eq('country_co', countries[i]))
              .geometry()
              .convexHull();

  // load raster dataset
  var dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
                  .filter(ee.Filter.date('2019-01-01', '2019-12-31'));

  // take the mean across multiple months
  var dataset = dataset.reduce(ee.Reducer.mean());

  // select the average radiance band
  var dataset = dataset.select('avg_rad_mean');

  // print summary stats
  print(dataset.reduceRegion({
    reducer: ee.Reducer.percentile([0, 2, 50, 98, 100]),
    scale: 463,
    maxPixels: 4e8,
    geometry: aoi,
  }));

  // visualize
  // Map.centerObject(aoi);
  // Map.addLayer(aoi, {color: 'FFFFFF'}, 'AOI');
  // var viz_params = {min: 0.0, max: 1.0,
  //                   palette: ['000000', '1A237E', '42B3D5', 'DCEDC8', 'FFFFFF']};
  // Map.addLayer(dataset, viz_params, 'Lights at Night (VIIRS)');

  // export
  print('exporting...');
  Export.image.toDrive({
    image: dataset,
    description: 'VIIRS_DNB_' + countries[i] + '_2019',
    scale: 463,
    maxPixels: 4e8,
    region: aoi,
    crs: 'EPSG:4326',
  });

}