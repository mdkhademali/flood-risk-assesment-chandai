## Flood Risk Assessment: Chandai Union, Natore

> A multi-source geospatial flood risk model combining **Sentinel-2 NDWI**, **SRTM DEM**, **Random Forest ML**, and **ROC/AUC evaluation** — built entirely in **Google Earth Engine** with Python/Colab visualization support.

---

## Study Area

| Field | Details |
|---|---|
| Location | Chandai Union, Natore, Rajshahi Division, Bangladesh |
| Season | Monsoon (June – September 2023) |
| Coordinate System | EPSG:4326 |
| Asset | `projects/khadem-gis-rs/assets/chandai_union` |

---

## Methodology

```
Sentinel-2 SR  ──►  NDWI
SRTM DEM       ──►  Elevation + Slope
                         │
                    Normalize (0–1)
                         │
              Weighted Overlay Model
              (NDWI×0.5, DEM×-0.3, Slope×-0.2)
                         │
                   Flood Risk Map
                         │
              ┌──────────┴──────────┐
         Sampling (4000 pts)   Visualization
              │
      Random Forest (100 trees)
              │
      ┌───────┴────────┐
  Classification    Accuracy Assessment
  (ML Flood Map)    (Confusion Matrix, Kappa)
                         │
                   ROC Curve + AUC
```

---

## Data Sources

| Dataset | Source | Resolution | Usage |
|---|---|---|---|
| Sentinel-2 SR Harmonized | `COPERNICUS/S2_SR_HARMONIZED` | 10 m | NDWI calculation |
| SRTM Digital Elevation Model | `USGS/SRTMGL1_003` | 30 m | Elevation & Slope |
| Study Area Boundary | GEE Asset (custom) | — | Clip & region mask |

---

## Flood Risk Model

The weighted overlay formula used:

$$\text{FloodRisk} = \text{NDWI}_n \times 0.5 + \text{Elevation}_n \times (-0.3) + \text{Slope}_n \times (-0.2)$$

| Factor | Weight | Rationale |
|---|---|---|
| NDWI (normalized) | +0.5 | Higher water index → higher flood risk |
| Elevation (normalized) | −0.3 | Lower elevation → higher flood risk |
| Slope (normalized) | −0.2 | Flatter terrain → higher flood risk |

---

## Machine Learning: Random Forest

- **Algorithm:** `smileRandomForest` (100 trees)
- **Features:** `NDWI`, `elevation`, `slope`
- **Label:** `FloodClass` (0 = non-flood, 1 = flood; split at mean risk)
- **Split:** 70% training / 30% testing (random column)
- **Output mode:** Binary classification + Probability (for ROC)

### Results

| Metric | Value |
|---|---|
| Overall Accuracy | **98.35%** |
| Kappa Coefficient | **0.967** |
| AUC Score | **0.998** |

### Confusion Matrix

```
             Predicted 0   Predicted 1
Actual 0   [    648    ,       8    ]
Actual 1   [     12    ,     541    ]
```

### Feature Importance

| Feature | Importance |
|---|---|
| NDWI | 268.99 |
| Slope | 137.93 |
| Elevation | 61.88 |

---

## Outputs & Visualizations

### Flood Risk Map (Weighted Overlay)
Color ramp: Green (Low) → Yellow (Moderate) → Red (High)

### ML Flood Map (Random Forest)
Binary classification: Green = Non-Flood | Red = Flood

### ROC Curve
- Red curve = model performance
- Gray dashed = random classifier baseline
- AUC = **0.9985** (near-perfect discrimination)

### Charts Produced
- `NDWI vs Flood Risk` — Scatter with trendline
- `Elevation vs Flood Risk` — Scatter with trendline
- `Slope vs Flood Risk` — Scatter with trendline
- `Flood Risk Distribution` — Histogram
- `Feature Importance` — Column chart
- `ROC Curve + AUC` — Line chart with reference diagonal

---

## References

- Gao, B.C. (1996). NDWI — A normalized difference water index for remote sensing of vegetation liquid water from space. *Remote Sensing of Environment*, 58(3), 257–266.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
- USGS SRTMGL1 — [lpdaac.usgs.gov](https://lpdaac.usgs.gov/products/srtmgl1v003/)
- ESA Sentinel-2 — [sentinel.esa.int](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- Google Earth Engine — [earthengine.google.com](https://earthengine.google.com)

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Built with** Google Earth Engine &nbsp;|&nbsp; Python &nbsp;|&nbsp; Remote Sensing

</div>
