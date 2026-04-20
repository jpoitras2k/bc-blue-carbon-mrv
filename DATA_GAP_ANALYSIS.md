# Data Gap Analysis — BC Blue Carbon (Eelgrass)

**Project:** BC Blue Carbon ML — Kaggle Uncharted Data Challenge  
**Prepared by:** Engineer (Paperclip [KEE-25](/KEE/issues/KEE-25))  
**Date:** 2026-04-12  
**Source inventory:** `PROJECT_LOG.md` | **Schema:** `BC_BLUE_CARBON_SCHEMA.md` | **Dataset:** `unified_bc_blue_carbon.csv`

---

## 1. Carbon Density Gap Summary

Of the **20 distinct site records** compiled in `unified_bc_blue_carbon.csv`:

| Metric | Count |
|--------|-------|
| Sites with measured `carbon_density_gCm2` | 4 (academic) |
| Sites with estimated `carbon_density_gCm2` | 1 (regional proxy) |
| Sites flagged `CARBON_GAP` (location/extent only, no carbon density data) | **15** |
| Sites with any sequestration rate | 4 |

**75% of catalogued sites lack direct sediment carbon density measurements.** The measured sites come from peer-reviewed academic literature (Douglas 2022, Postlethwaite 2018, Lin 2024, Lutz 2018), highlighting that BC provincial and federal monitoring programs currently lack integrated carbon stock quantification.

---

## 2. Regional Coverage vs. Data Sparsity

### Best-Covered Regions (spatial data richest)

| Region | Sites in dataset | Sites with carbon density data |
|--------|-----------------|----------------------|
| Salish Sea | 14 | 4 (Cowichan Estuary, Portage Inlet, Skagit County, Proxy) |
| West Coast VI | 3 | 1 (Clayoquot Sound) |
| Central Coast | 2 | 0 |
| NE Pacific | 1 | 0 (rate measured, density gap) |
| North Coast / Haida Gwaii | 0 | 0 |

**Salish Sea** has the most spatial coverage (GIS polygons from Islands Trust, BC Data Catalogue CRIMS, Hakai Institute, SeaChange MCS), and recently added academic studies (Lin 2024, Lutz 2018) have doubled the number of measured carbon density points in this region. However, a 71% gap still remains within this best-covered region.

### Most Data-Sparse Regions (critical gaps)

1. **North Coast / Haida Gwaii** — zero eelgrass sites inventoried. No GIS data, no carbon data, no monitoring programs discovered in the source inventory. This remains the largest uncharted blind spot.

2. **Central Coast** — two records (Hakai time-series extent monitoring), both are `CARBON_GAP`. No sediment cores or published carbon density studies were found despite established eelgrass presence.

3. **West Coast Vancouver Island (outside Clayoquot Sound)** — large coastline (Barkley Sound, Nootka Sound, Kyuquot Sound) with no indexed carbon or spatial records beyond Bamfield Inlet (GIS only) and the Postlethwaite 2018 Clayoquot study.

4. **Cowichan Bay / Saanich Inlet Z. japonica beds** — dwarf eelgrass (BC-EEL-017) is known to have different carbon dynamics than *Z. marina*, but no measurements exist; this sub-habitat is entirely un-quantified.

---

## 3. The "Uncharted" Value Proposition for Kaggle

British Columbia's eelgrass meadows represent one of the largest inventoried blue carbon ecosystems on North America's Pacific coast, yet **direct carbon density measurements exist for only a handful of the province's identified eelgrass sites**. Comprehensive GIS distribution layers show *where* eelgrass grows across thousands of kilometres of coastline, but the corresponding sediment carbon measurements are missing for 75% of known sites. This spatial-carbon mismatch creates a genuine prediction challenge: can a model trained on a few measured regions plus regional academic averages accurately estimate carbon stocks for the 15+ unsampled sites across the Central Coast and Salish Sea, using only habitat type, sediment type, and geographic region as features?

---

## 4. Recommended ML Approach

**Target variable:** `carbon_density_gCm2`

**Input features (Mechanistic Drivers):**
- `region` (categorical - for regional grouping)
- `sediment_type` (categorical)
- `habitat_type` (categorical)
- `buoy_temperature_c` (continuous - 12mo physical mean)
- `buoy_salinity_pss` (continuous - 12mo physical mean)
- `anthropogenic_stress_index` (continuous - port proximity)
- `measurement_depth_cm` (continuous)

**Recommended approach (Mechanistic shift):**

1. **Feature Engineering over Feature Space** — Focus on transforming `latitude`/`longitude` into physical environmental drivers (temperature, salinity, depth, stress) rather than using coordinates directly. This prevents geographic overfitting and improves scientific generalizability.

1. **Gradient Boosted Trees (XGBoost / LightGBM)** — robust to mixed data types and effective for small datasets when augmented with regional proxies.

2. **Bayesian Linear Regression** — allows incorporating NE Pacific literature (Prentice et al. 2020) as informative priors to regularize predictions for unsampled sites.

3. **Spatial Kriging** — to generate a continuous carbon stock surface across the BC coast, providing a baseline to compare against feature-based ML predictions.

---

## 5. Python Library Installation Status

| Library | Status |
|---------|--------|
| `geopandas` | ✅ Installed (0.14.3) |
| `rasterio` | ✅ Installed (1.3.9) |
| `shapely` | ✅ Installed (2.0.3) |

**Status:** All required Python libraries for spatial processing and GIS integration are now available in the environment.

---

## 6. Outputs Produced

| File | Description |
|------|-------------|
| `unified_bc_blue_carbon.csv` | 20-row unified dataset; 14 columns; 15 CARBON_GAP sites flagged |
| `DATA_GAP_ANALYSIS.md` | This document (updated with new academic sources) |
| `BC_BLUE_CARBON_SCHEMA.md` | Schema definition |
| `PROJECT_LOG.md` | Source inventory |
