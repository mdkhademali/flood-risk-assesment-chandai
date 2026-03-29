// ===============================
// 1. Load Study Area
// ===============================
var studyArea = ee.FeatureCollection("projects/khadem-gis-rs/assets/chandai_union");

Map.centerObject(studyArea, 12);
Map.addLayer(studyArea, {}, 'Chandai Union');

// ===============================
// 2. Time Range (Monsoon)
// ===============================
var startDate = '2023-06-01';
var endDate   = '2023-09-30';

// ===============================
// 3. Sentinel-2 (NDWI)
// ===============================
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(startDate, endDate)
            .filterBounds(studyArea)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .select(['B3','B8'])
            .median();

var ndwi = s2.normalizedDifference(['B3','B8'])
              .rename('NDWI')
              .clip(studyArea);

// ===============================
// 4. DEM + Slope
// ===============================
var dem = ee.Image('USGS/SRTMGL1_003')
            .rename('elevation')
            .clip(studyArea);

var slope = ee.Terrain.slope(dem).rename('slope');

// ===============================
// 5. Normalize (0–1)
// ===============================
var ndwi_n  = ndwi.unitScale(-0.5, 0.5);
var dem_n   = dem.unitScale(0, 100);
var slope_n = slope.unitScale(0, 10);

// ===============================
// 6. Flood Risk Model (Weighted)
// ===============================
var floodRisk = ndwi_n.multiply(0.5)
  .add(dem_n.multiply(-0.3))
  .add(slope_n.multiply(-0.2))
  .rename('FloodRisk');

// ===============================
// 7. Visualization
// ===============================
var floodVis = {min: 0, max: 1, palette: ['green','yellow','red']};
Map.addLayer(ndwi, {min:-0.5, max:0.5, palette:['brown','white','blue']}, 'NDWI');
Map.addLayer(floodRisk, floodVis, 'Flood Risk');

// ===============================
// 8. Sampling (for Graph + ML)
// ===============================
var sample = floodRisk.addBands(ndwi)
  .addBands(dem)
  .addBands(slope)
  .sample({
    region: studyArea,
    scale: 30,
    numPixels: 4000,
    geometries: true
  });

// ===============================
// 9. Correlation
// ===============================
var corr = sample.reduceColumns({
  reducer: ee.Reducer.pearsonsCorrelation(),
  selectors: ['NDWI', 'FloodRisk']
});
print('NDWI vs FloodRisk Correlation:', corr);

// ===============================
// 10. Scatter & Histogram Charts
// ===============================
var chart1 = ui.Chart.feature.byFeature(sample, 'NDWI', ['FloodRisk'])
  .setChartType('ScatterChart')
  .setOptions({
    title: 'NDWI vs Flood Risk',
    hAxis: {title: 'NDWI'},
    vAxis: {title: 'Flood Risk'},
    pointSize: 3,
    trendlines: {0: {color: 'red'}}
  });
print(chart1);

var chart2 = ui.Chart.feature.byFeature(sample, 'elevation', ['FloodRisk'])
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Elevation vs Flood Risk',
    hAxis: {title: 'Elevation'},
    vAxis: {title: 'Flood Risk'},
    pointSize: 3,
    trendlines: {0: {color: 'blue'}}
  });
print(chart2);

var chart3 = ui.Chart.feature.byFeature(sample, 'slope', ['FloodRisk'])
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Slope vs Flood Risk',
    hAxis: {title: 'Slope'},
    vAxis: {title: 'Flood Risk'},
    pointSize: 3,
    trendlines: {0: {color: 'green'}}
  });
print(chart3);

var hist = ui.Chart.feature.histogram(sample, 'FloodRisk')
  .setOptions({title: 'Flood Risk Distribution'});
print(hist);

// ===============================
// 11. ML Classification
// ===============================
var stats = floodRisk.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: studyArea,
  scale: 30,
  maxPixels: 1e13
});
var meanRisk = ee.Number(stats.get('FloodRisk'));
print('Mean Flood Risk:', meanRisk);

var training = sample.map(function(f) {
  return f.set('FloodClass', ee.Number(f.get('FloodRisk')).gt(meanRisk).toInt());
});
print('Class check:', training.aggregate_histogram('FloodClass'));

// Train classifier
var classifier = ee.Classifier.smileRandomForest(100)
  .train({
    features: training,
    classProperty: 'FloodClass',
    inputProperties: ['NDWI','elevation','slope']
  });

// Apply
var classified = floodRisk.addBands(ndwi).addBands(dem).addBands(slope)
  .classify(classifier);
Map.addLayer(classified, {min:0, max:1, palette:['green','red']}, 'ML Flood Map');

// ===============================
// 12. Accuracy Assessment
// ===============================
var withRandom = training.randomColumn('random');
var trainSet = withRandom.filter(ee.Filter.lt('random', 0.7));
var testSet  = withRandom.filter(ee.Filter.gte('random', 0.7));

var trainedClassifier = ee.Classifier.smileRandomForest(100)
  .train({
    features: trainSet,
    classProperty: 'FloodClass',
    inputProperties: ['NDWI','elevation','slope']
  });

var testClassified = testSet.classify(trainedClassifier);
var cm = testClassified.errorMatrix('FloodClass', 'classification');
print('Confusion Matrix:', cm);
print('Overall Accuracy:', cm.accuracy());
print('Kappa Coefficient:', cm.kappa());

// ===============================
// 13. Feature Importance
// ===============================
var importance = ee.Dictionary(trainedClassifier.explain().get('importance'));
var importanceFc = ee.FeatureCollection(
  importance.keys().map(function(key) {
    return ee.Feature(null, {variable: key, importance: importance.get(key)});
  })
);
var importanceChart = ui.Chart.feature.byFeature(importanceFc, 'variable', ['importance'])
  .setChartType('ColumnChart')
  .setOptions({
    title: 'Feature Importance (Random Forest)',
    hAxis: {title: 'Variables'},
    vAxis: {title: 'Importance'},
    legend: {position: 'none'},
    colors: ['#1a73e8']
  });
print(importanceChart);

// ===============================
// 14. ROC Curve + AUC (Fixed)
// ===============================

// Train probability classifier
var probClassifier = ee.Classifier.smileRandomForest(100)
  .setOutputMode('PROBABILITY')
  .train({
    features: trainSet,
    classProperty: 'FloodClass',
    inputProperties: ['NDWI','elevation','slope']
  });

// Classify test set with probability output
var testWithProb = testSet.classify(probClassifier).map(function(f) {
  return f.set('prob', f.get('classification'));
});

// Threshold list
var thresholds = ee.List.sequence(0, 1, 0.05);

// FIX: Use errorMatrix correctly via .array() to extract TP/FP/TN/FN
var rocPoints = thresholds.map(function(th) {
  th = ee.Number(th);

  var predicted = testWithProb.map(function(f) {
    var pred = ee.Number(f.get('prob')).gte(th).toInt();
    return f.set('pred', pred);
  });

  // errorMatrix returns a ConfusionMatrix; use .array() to access cells
  var matrix = predicted.errorMatrix('FloodClass', 'pred').array();

  // matrix layout (rows = actual, cols = predicted):
  // [TN, FP]   row 0 (actual=0)
  // [FN, TP]   row 1 (actual=1)
  var TN = matrix.get([0, 0]);
  var FP = matrix.get([0, 1]);
  var FN = matrix.get([1, 0]);
  var TP = matrix.get([1, 1]);

  var TPR = ee.Number(TP).divide(ee.Number(TP).add(ee.Number(FN)).max(1)); // Sensitivity
  var FPR = ee.Number(FP).divide(ee.Number(FP).add(ee.Number(TN)).max(1)); // Fall-out

  return ee.Feature(null, {
    threshold: th,
    TPR: TPR,
    FPR: FPR
  });
});

var rocFc = ee.FeatureCollection(rocPoints);

// ROC Chart
var rocChart = ui.Chart.feature.byFeature(rocFc, 'FPR', ['TPR'])
  .setChartType('LineChart')
  .setOptions({
    title: 'ROC Curve',
    hAxis: {title: 'False Positive Rate (FPR)', viewWindow: {min:0, max:1}},
    vAxis: {title: 'True Positive Rate (TPR)', viewWindow: {min:0, max:1}},
    lineWidth: 2,
    colors: ['#e53935'],
    series: {0: {labelInLegend: 'ROC'}}
  });
print(rocChart);

// FIX: AUC via server-side ee.List.sequence + iterate (no client-side loop)
var fprList = rocFc.aggregate_array('FPR');
var tprList = rocFc.aggregate_array('TPR');

// Trapezoidal AUC using ee.List.sequence iterate
var n = thresholds.size();
var indices = ee.List.sequence(1, n.subtract(1));

var auc = ee.Number(indices.iterate(function(i, prev) {
  i = ee.Number(i).toInt();
  var x1 = ee.Number(fprList.get(i.subtract(1)));
  var x2 = ee.Number(fprList.get(i));
  var y1 = ee.Number(tprList.get(i.subtract(1)));
  var y2 = ee.Number(tprList.get(i));
  var trapezoid = x2.subtract(x1).abs().multiply(y1.add(y2).divide(2));
  return ee.Number(prev).add(trapezoid);
}, ee.Number(0)));

print('AUC Score:', auc);

// ===============================
// 15. Export Flood Risk Map
// ===============================
Export.image.toDrive({
  image: floodRisk,
  description: 'FloodRisk_Chandai',
  region: studyArea.geometry(),
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// ===============================
// 16. Export ML Flood Map
// ===============================
Export.image.toDrive({
  image: classified.select('classification'),
  description: 'ML_Flood_Map_Chandai',
  region: studyArea.geometry(),
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

Export.table.toDrive({
  collection: training,
  description: 'sample_points',
  fileFormat: 'CSV'
});