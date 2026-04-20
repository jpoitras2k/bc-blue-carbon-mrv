import os
import pandas as pd
import numpy as np
import requests
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from shapely.geometry import shape, Point, Polygon
from shapely.ops import nearest_points
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def load_and_validate_data(file_path="unified_bc_blue_carbon.csv", df=None):
    """
    Loads the blue carbon data, applies schema, and performs basic validation.
    """
    if df is None:
        df = pd.read_csv(file_path)

    # Apply schema data types
    schema = {
        "site_id": str,
        "site_name": str,
        "latitude": float,
        "longitude": float,
        "region": str,
        "habitat_type": str,
        "sediment_type": str,
        "carbon_density_gCm2": float,
        "sequestration_rate_gCm2yr": float,
        "measurement_depth_cm": "Int64",  # Use Int64 to allow NaN
        "survey_year": "Int64",  # Use Int64 to allow NaN
        "data_source": str,
        "access_type": str,
        "notes": str,
    }

    # Why? Standard pandas integer columns historically drop to floats when they encounter NaN.
    # By carefully forcing 'Int64' (capital I), we allow integer columns (like survey_year and depth)
    # to store null values without corrupting the downstream scikit-learn models.
    for col, dtype in schema.items():
        if dtype == str:
            df[col] = (
                df[col].astype(str).replace("nan", np.nan)
            )  # Convert 'nan' string to actual NaN
        elif dtype == "Int64":  # For nullable integers
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif dtype == float:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(dtype)

    # Spatial validation (Latitude and Longitude ranges)
    # Range: latitude (48.0–60.0), longitude (-140.0–-114.0)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    if not ((df["latitude"] >= 48.0) & (df["latitude"] <= 60.0)).all():
        out_of_range_lat = df[(df["latitude"] < 48.0) | (df["latitude"] > 60.0)]
        print(
            "Warning: "
            + str(len(out_of_range_lat))
            + " latitudes are out of the 48.0-60.0 range. Samples:\n"
            + str(out_of_range_lat[["site_id", "latitude", "notes"]].head())
        )
    if not ((df["longitude"] >= -140.0) & (df["longitude"] <= -114.0)).all():
        out_of_range_lon = df[(df["longitude"] < -140.0) | (df["longitude"] > -114.0)]
        print(
            "Warning: "
            + str(len(out_of_range_lon))
            + " longitudes are out of the -140.0--114.0 range. Samples:\n"
            + str(out_of_range_lon[["site_id", "longitude", "notes"]].head())
        )

    # Validate categorical columns
    valid_regions = [
        "Salish Sea",
        "West Coast VI",
        "Central Coast",
        "North Coast",
        "Haida Gwaii",
        "NE Pacific",
    ]
    valid_habitat_types = [
        "eelgrass",
        "eelgrass_dwarf",
        "kelp",
        "saltmarsh",
        "mixed_seagrass",
    ]
    valid_sediment_types = [
        "mud",
        "sand",
        "mud_sand",
        "gravel",
        "rocky",
        "organic",
        "unknown",
    ]
    valid_access_types = [
        "open",
        "restricted",
        "grey_literature",
        "academic",
        "estimated",
    ]

    invalid_regions = df[~df["region"].isin(valid_regions)]
    if not invalid_regions.empty:
        print(
            f"Warning: {len(invalid_regions)} regions are not in the valid list. Invalid values: {invalid_regions['region'].unique()}"
        )

    invalid_habitat_types = df[~df["habitat_type"].isin(valid_habitat_types)]
    if not invalid_habitat_types.empty:
        print(
            f"Warning: {len(invalid_habitat_types)} habitat_types are not in the valid list. Invalid values: {invalid_habitat_types['habitat_type'].unique()}"
        )

    # Fill NaN sediment_type with 'unknown' before validation
    df["sediment_type"] = df["sediment_type"].fillna("unknown")
    invalid_sediment_types = df[~df["sediment_type"].isin(valid_sediment_types)]
    if not invalid_sediment_types.empty:
        print(
            f"Warning: {len(invalid_sediment_types)} sediment_types are not in the valid list. Invalid values: {invalid_sediment_types['sediment_type'].unique()}"
        )

    invalid_access_types = df[~df["access_type"].isin(valid_access_types)]
    if not invalid_access_types.empty:
        print(
            f"Warning: {len(invalid_access_types)} access_types are not in the valid list. Invalid values: {invalid_access_types['access_type'].unique()}"
        )

    # Check for unique site_id
    if not df["site_id"].is_unique:
        print("Warning: Duplicate site_id found.")

    # Identify CARBON_GAP sites
    df["is_carbon_gap"] = (
        df["notes"].fillna("").str.contains("CARBON_GAP:")
        | df["carbon_density_gCm2"].isna()
    )

    return df


def fetch_geojson_data(url, local_cache_path=None):
    """
    Fetches GeoJSON data from a given URL, with optional local caching.
    """
    if local_cache_path and os.path.exists(local_cache_path):
        print(f"Loading cached data from: {local_cache_path}")
        with open(local_cache_path, "r") as f:
            return json.load(f)

    print(f"Fetching data from URL: {url}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if local_cache_path:
        os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)
        with open(local_cache_path, "w") as f:
            json.dump(data, f)
        print(f"Saved data to cache: {local_cache_path}")

    return data


ERDDAP_CACHE_FILE = "data/ocean_cache.json"


def load_ocean_cache():
    if os.path.exists(ERDDAP_CACHE_FILE):
        with open(ERDDAP_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_ocean_cache(cache):
    os.makedirs(os.path.dirname(ERDDAP_CACHE_FILE), exist_ok=True)
    with open(ERDDAP_CACHE_FILE, "w") as f:
        json.dump(cache, f)


# Mapping of regions to nearest Hakai telemetered buoys/stations
# Used for fetching mechanistic physical drivers (SST, Salinity)
BUOY_MAPPING = {
    "Salish Sea": {
        "dataset_id": "HakaiQuadraLimpet5min",
        "temp_var": "WaterTemp_Avg",
        "sal_var": "WaterSalinity_Avg",
    },
    "West Coast VI": {
        "dataset_id": "HakaiBamfieldBoL5min",
        "temp_var": "TSG_T_Avg",
        "sal_var": "TSG_S_Avg",
    },
    "Central Coast": {
        "dataset_id": "HakaiKCBuoy1hour",
        "temp_var": "WaterTemp_Avg",
        "sal_var": "WaterSalinity_Avg",
    },
    "NE Pacific": {
        "dataset_id": "HakaiBamfieldBoL5min",
        "temp_var": "TSG_T_Avg",
        "sal_var": "TSG_S_Avg",
    },
}


def fetch_hakai_buoy_data(region, cache):
    """
    Fetches Temperature and Salinity from Hakai ERDDAP for a given region.
    Returns 12-month averages for mechanistic modeling.
    """
    if region not in BUOY_MAPPING:
        return np.nan, np.nan

    mapping = BUOY_MAPPING[region]
    ds_id = mapping["dataset_id"]
    temp_var = mapping["temp_var"]
    sal_var = mapping["sal_var"]

    key = f"hakai_{ds_id}_12mo"
    if key in cache:
        return cache[key].get("temp", np.nan), cache[key].get("sal", np.nan)

    from datetime import datetime, timedelta

    try:
        from erddapy import ERDDAP

        e = ERDDAP(server="https://catalogue.hakai.org/erddap", protocol="tabledap")
        e.dataset_id = ds_id

        # Calculate 12 months ago
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=365)

        e.constraints = {
            "time>=": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time<=": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        # Variable names in the download might differ slightly from internal metadata
        e.variables = [temp_var, sal_var]

        df_e = e.to_pandas()
        if not df_e.empty:
            # Drop any potential 'time' column if it was included
            cols = [c for c in df_e.columns if "time" not in c.lower()]
            # Find the columns that match our requested variables (case insensitive or partial match)
            t_col = [c for c in df_e.columns if temp_var in c][0]
            s_col = [c for c in df_e.columns if sal_var in c][0]

            avg_temp = df_e[t_col].mean()
            avg_sal = df_e[s_col].mean()
            results = {"temp": float(avg_temp), "sal": float(avg_sal)}
            print(f"Fetched Hakai buoy data for {region} ({ds_id}): T={avg_temp:.2f}, S={avg_sal:.2f}")
        else:
            results = {"temp": np.nan, "sal": np.nan}

        cache[key] = results
        save_ocean_cache(cache)
        return results["temp"], results["sal"]
    except Exception as err:
        print(f"Warning: Could not fetch Hakai buoy data for {region} ({ds_id}): {err}")
        cache[key] = {"temp": np.nan, "sal": np.nan}
        return np.nan, np.nan


def fetch_bio_oracle_ocean_data(lat, lon, cache):
    """
    Fetches SST and Salinity from Bio-ORACLE ERDDAP.
    Caches requests to rounded (2 decimal places) lat/lon combinations (~1.1 km).
    """
    if pd.isna(lat) or pd.isna(lon):
        return np.nan, np.nan

    lat_r = round(lat, 2)
    lon_r = round(lon, 2)
    key = f"{lat_r},{lon_r}"

    if key in cache:
        return cache[key].get("sst", np.nan), cache[key].get("sss", np.nan)

    import time

    time.sleep(1.0)  # Slow down the bot clicks to avoid rate limiting

    try:
        from erddapy import ERDDAP

        e = ERDDAP(server="https://erddap.bio-oracle.org/erddap", protocol="griddap")
        e.requests_kwargs = {
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
            }
        }

        datasets = {
            "sst": "thetao_baseline_2000_2019_depthsurf",
            "sss": "so_baseline_2000_2019_depthsurf",
        }

        results = {}
        for var, ds_id in datasets.items():
            e.dataset_id = ds_id
            e.griddap_initialize()  # Load grid bounds

            # Since ERDDAP requires time parameter for bio-oracle even if it's a baseline, we let erddapy handle constraints
            # We strictly update the spatial boundaries, preserving time/depth natively required by griddap
            e.constraints["latitude>="] = lat_r - 0.05
            e.constraints["latitude<="] = lat_r + 0.05
            e.constraints["longitude>="] = lon_r - 0.05
            e.constraints["longitude<="] = lon_r + 0.05

            # Attempt to extract flattened pandas grid slice
            df_e = e.to_pandas()
            # The actual value is usually in the last column
            if not df_e.empty:
                results[var] = float(df_e.iloc[:, -1].mean())
            else:
                results[var] = np.nan

        cache[key] = results
        # Save incrementally for absolute safety on massive datasets
        save_ocean_cache(cache)
        return results.get("sst", np.nan), results.get("sss", np.nan)
    except Exception as err:
        cache[key] = {"sst": np.nan, "sss": np.nan}
        return np.nan, np.nan


def process_crd_eelgrass_data(eelgrass_geojson, sediment_geojson, existing_df):
    """
    Processes CRD eelgrass GeoJSON data, extracts relevant features,
    and performs spatial join with sediment data.
    """
    import geopandas as gpd

    # Load into GeoDataFrames
    eelgrass_gdf = gpd.GeoDataFrame.from_features(
        eelgrass_geojson["features"], crs="EPSG:4326"
    )
    sediment_gdf = gpd.GeoDataFrame.from_features(
        sediment_geojson["features"], crs="EPSG:4326"
    )
    # Project to EPSG:3857 (Web Mercator) to calculate accurate geometric features and centroids without warnings
    eelgrass_proj = eelgrass_gdf.to_crs("EPSG:3857")
    sediment_proj = sediment_gdf.to_crs("EPSG:3857")

    # Calculate centroids on the flat projected plane
    eelgrass_proj["centroid"] = eelgrass_proj.geometry.centroid

    # Extract pristine latitudes/longitudes by casting just the centroids back to geographic EPSG:4326
    centroids_4326 = gpd.GeoSeries(eelgrass_proj["centroid"], crs="EPSG:3857").to_crs(
        "EPSG:4326"
    )
    eelgrass_proj["latitude"] = centroids_4326.y
    eelgrass_proj["longitude"] = centroids_4326.x

    if "Shape.STArea()" in eelgrass_proj.columns:
        eelgrass_proj["area_ha"] = eelgrass_proj["Shape.STArea()"] / 10000.0
    else:
        eelgrass_proj["area_ha"] = eelgrass_proj.area / 10000.0

    # Ensure Geometry column is set to Centroid for the proxy/sediment spatial join
    eelgrass_proj = eelgrass_proj.set_geometry("centroid")

    sediment_type_mapping = {
        "mud/sand": "mud_sand",
        "gravelly mud/san": "mud_sand",
        "vegetation": "organic",
        "gravelly sand": "sand",
        "no survey": "unknown",
    }

    # Vectorized Spatial Join to find nearest Sediment polygon
    joined_gdf = gpd.sjoin_nearest(
        eelgrass_proj, sediment_proj, how="left", distance_col="dist"
    )
    if "SEDIMENT" in joined_gdf.columns:
        joined_gdf["sediment_type"] = (
            joined_gdf["SEDIMENT"].map(sediment_type_mapping).fillna("unknown")
        )
    else:
        joined_gdf["sediment_type"] = "unknown"

    # Drop duplicates generated by equidistant sediment polygons
    joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep="first")]

    crd_data = []

    # Filter the measured proxy dataset once
    measured_eelgrass = None
    if existing_df is not None and not existing_df.empty:
        measured_eelgrass = existing_df[
            (existing_df["habitat_type"] == "eelgrass")
            & (existing_df["carbon_density_gCm2"].notna())
        ]

    for i, row in joined_gdf.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]

        # Dynamically find nearest measured eelgrass site for carbon density proxy
        carbon_density = np.nan
        if measured_eelgrass is not None and not measured_eelgrass.empty:
            distances = measured_eelgrass.apply(
                lambda ms: haversine_distance(
                    lat, lon, ms["latitude"], ms["longitude"]
                ),
                axis=1,
            )
            nearest_idx = distances.idxmin()
            carbon_density = measured_eelgrass.loc[nearest_idx, "carbon_density_gCm2"]

        crd_data.append(
            {
                "site_id": f"CRD-EEL-{i+1:03d}",
                "site_name": f"CRD Eelgrass Site {i+1}",
                "latitude": lat,
                "longitude": lon,
                "region": "Salish Sea",
                "habitat_type": "eelgrass",
                "sediment_type": row["sediment_type"],
                "carbon_density_gCm2": carbon_density,  # CARBON_GAP
                "sequestration_rate_gCm2yr": np.nan,
                "measurement_depth_cm": np.nan,
                "survey_year": 2024,
                "data_source": "CRD Harbours Atlas (ArcGIS MapServer Layer 25)",
                "access_type": "open",
                "notes": "CRD_ATTRIBUTION: used with permission from Capital Regional District; CARBON_GAP",
                "area_ha": row["area_ha"],  # Retain for engineer_features
            }
        )

    return pd.DataFrame(crd_data)


def process_janousek_data(file_path):
    df_janousek = pd.read_csv(file_path)

    # Filter for seagrass meadow rows (Ecosystem code 'SG')
    df_janousek_seagrass = df_janousek[df_janousek["Ecosystem"] == "SG"].copy()

    # Standardize habitat_type to 'eelgrass' for Janousek seagrass data
    df_janousek_seagrass["habitat_type"] = "eelgrass"

    # Add a note about the data source
    if "data_source" not in df_janousek_seagrass.columns:
        df_janousek_seagrass["data_source"] = ""
    df_janousek_seagrass["data_source"] = (
        df_janousek_seagrass["data_source"].fillna("") + "; Janousek 2025 Data Ingest"
    )
    df_janousek_seagrass["access_type"] = "academic"

    # Ensure all columns expected by the unified schema are present. Add missing ones with NaN if needed.
    expected_columns = [
        "site_id",
        "site_name",
        "latitude",
        "longitude",
        "region",
        "habitat_type",
        "sediment_type",
        "carbon_density_gCm2",
        "sequestration_rate_gCm2yr",
        "measurement_depth_cm",
        "survey_year",
        "data_source",
        "access_type",
        "notes",
    ]
    for col in expected_columns:
        if col not in df_janousek_seagrass.columns:
            df_janousek_seagrass[col] = np.nan

    return df_janousek_seagrass[expected_columns]


def process_hakai_data(file_path):
    df_hakai = pd.read_csv(file_path)
    # Ensure 'access_type' column exists before use
    if "access_type" not in df_hakai.columns:
        df_hakai["access_type"] = np.nan  # Initialize with NaN if missing
    print(
        "Hakai df access_type immediately after read_csv:",
        df_hakai["access_type"].unique() if not df_hakai.empty else "Empty",
    )

    # No specific filtering mentioned for Hakai, so all rows will be processed.

    # Add a note about the data source
    if "data_source" not in df_hakai.columns:
        df_hakai["data_source"] = ""
    df_hakai["data_source"] = (
        df_hakai["data_source"].fillna("") + "; Hakai / Prentice 2020 Data Ingest"
    )
    df_hakai["access_type"] = "open"  # Fixed: direct assignment

    # Ensure all columns expected by the unified schema are present. Add missing ones with NaN if needed.
    expected_columns = [
        "site_id",
        "site_name",
        "latitude",
        "longitude",
        "region",
        "habitat_type",
        "sediment_type",
        "carbon_density_gCm2",
        "sequestration_rate_gCm2yr",
        "measurement_depth_cm",
        "survey_year",
        "data_source",
        "access_type",
        "notes",
    ]
    for col in expected_columns:
        if col not in df_hakai.columns:
            df_hakai[col] = np.nan

    return df_hakai[expected_columns]


def process_hakai_grain_size_data(file_path, df_unified):
    """
    Processes hakai_grain_size.csv to extract percent_fines and percent_oc,
    and merges it into the unified dataframe.
    """
    df_grain_size = pd.read_csv(file_path)

    # Filter for British Columbia and Vegetated cover
    df_grain_size_bc_veg = df_grain_size[
        (df_grain_size["region"] == "British Columbia")
        & (df_grain_size["cover"] == "Vegetated")
    ].copy()

    # We average the percent_fines and percent_oc by site.
    # Why? Grain size data is often collected from multiple cores within a single site.
    # We need a single representative mean per site to join against our unified site list.
    grain_size_site_means = (
        df_grain_size_bc_veg.groupby(["site", "region"])[
            ["percent_fines", "percent_oc"]
        ]
        .mean()
        .reset_index()
    )

    # Calculate regional means for imputation fallback
    grain_size_regional_means = (
        df_grain_size_bc_veg.groupby(["region"])[["percent_fines", "percent_oc"]]
        .mean()
        .reset_index()
    )
    grain_size_regional_means.rename(
        columns={
            "percent_fines": "regional_percent_fines_mean",
            "percent_oc": "regional_percent_oc_mean",
        },
        inplace=True,
    )

    # Calculate overall BC vegetated mean for imputation fallback
    bc_vegetated_overall_mean_fines = 1.92  # Provided in the task
    bc_vegetated_overall_mean_oc = df_grain_size_bc_veg[
        "percent_oc"
    ].mean()  # Dynamically calculate

    # Merge into the unified dataframe based on site and region
    df_unified = pd.merge(
        df_unified,
        grain_size_site_means,
        how="left",
        left_on=["site_name", "region"],
        right_on=["site", "region"],
        suffixes=("", "_grain_size"),
    )
    if "site_grain_size" in df_unified.columns:
        df_unified.drop(
            columns=["site_grain_size"], inplace=True
        )  # Drop the redundant site column

    # Identify BC vegetated sites in the unified dataframe for targeted imputation
    # This assumes 'eelgrass' is a primary 'vegetated' type. We can refine if needed.
    bc_regions = [
        "Salish Sea",
        "West Coast VI",
        "Central Coast",
        "North Coast",
        "Haida Gwaii",
        "NE Pacific",
    ]
    vegetated_habitat_types = [
        "eelgrass",
        "eelgrass_dwarf",
        "saltmarsh",
        "mixed_seagrass",
    ]
    is_bc_vegetated_site = df_unified["region"].isin(bc_regions) & df_unified[
        "habitat_type"
    ].isin(vegetated_habitat_types)

    # Impute missing percent_fines for BC vegetated sites
    missing_fines_mask = is_bc_vegetated_site & df_unified["percent_fines"].isna()
    if missing_fines_mask.any():
        # First try to impute with regional means
        df_unified = pd.merge(
            df_unified,
            grain_size_regional_means,
            how="left",
            on="region",
            suffixes=("", "_regional_mean"),
        )
        df_unified.loc[missing_fines_mask, "percent_fines"] = df_unified.loc[
            missing_fines_mask, "regional_percent_fines_mean"
        ].fillna(bc_vegetated_overall_mean_fines)
        if "regional_percent_fines_mean" in df_unified.columns:
            df_unified.drop(columns=["regional_percent_fines_mean"], inplace=True)

    # Impute missing percent_oc for BC vegetated sites
    missing_oc_mask = is_bc_vegetated_site & df_unified["percent_oc"].isna()
    if missing_oc_mask.any():
        # First try to impute with regional means
        if (
            "regional_percent_oc_mean" not in df_unified.columns
        ):  # Ensure it's there if not merged for fines
            df_unified = pd.merge(
                df_unified,
                grain_size_regional_means,
                how="left",
                on="region",
                suffixes=("", "_regional_mean"),
            )
        df_unified.loc[missing_oc_mask, "percent_oc"] = df_unified.loc[
            missing_oc_mask, "regional_percent_oc_mean"
        ].fillna(bc_vegetated_overall_mean_oc)
        if "regional_percent_oc_mean" in df_unified.columns:
            df_unified.drop(columns=["regional_percent_oc_mean"], inplace=True)

    # For any remaining NaNs in percent_fines or percent_oc (e.g., non-BC vegetated sites, or if regional means were NaN), fill with a global mean if appropriate for model
    # Use the overall mean from the df_unified itself for these remaining NaNs for robustness
    df_unified["percent_fines"] = df_unified["percent_fines"].fillna(
        df_unified["percent_fines"].mean()
    )
    df_unified["percent_oc"] = df_unified["percent_oc"].fillna(
        df_unified["percent_oc"].mean()
    )

    return df_unified


def process_hakai_isotope_data(file_path, df_unified):
    """
    Processes hakai_isotope_data.csv to extract d13C and d15N,
    and merges it into the unified dataframe.
    """
    df_isotope = pd.read_csv(file_path)

    # Rename columns to match expected names
    df_isotope.rename(
        columns={"site": "site_name", "avg_d13C": "d13C", "avg_d15N": "d15N"},
        inplace=True,
    )

    # Standardize site names for better merging
    site_name_mapping = {
        "Padilla Bay": "Padilla Bay, WA",
        "Skagit Bay": "Skagit County (3 Bays)",
        # Add other specific mappings if they become apparent and necessary
    }
    df_isotope["site_name"] = df_isotope["site_name"].replace(site_name_mapping)

    # Merge into the unified dataframe based on site_name
    df_unified = pd.merge(
        df_unified,
        df_isotope[["site_name", "d13C", "d15N"]],
        how="left",
        on=["site_name"],
        suffixes=("", "_isotope"),
    )

    # Impute missing d13C and d15N with the mean if necessary
    df_unified["d13C"] = df_unified["d13C"].fillna(df_unified["d13C"].mean())
    df_unified["d15N"] = df_unified["d15N"].fillna(df_unified["d15N"].mean())

    return df_unified


def engineer_features(df):
    """
    Performs feature engineering on the loaded data.
    """
    # One-hot encode categorical features for ML without destroying human-readable originals
    categorical_cols = ["region", "habitat_type", "sediment_type"]
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)

    # Impute missing numerical features with median
    numerical_cols_to_impute = ["measurement_depth_cm", "survey_year"]
    for col in numerical_cols_to_impute:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Derived Feature 1: Habitat Area (ha)
    # For CRD data, area_ha is already calculated in process_crd_eelgrass_data.
    # For other sites, if not available, set to NaN.
    # Assuming 'area_ha' is a temporary column from CRD processing. We'll formalize it.
    if "area_ha" not in df.columns:
        df["habitat_area_ha"] = np.nan
    else:
        df["habitat_area_ha"] = df["area_ha"]
        df = df.drop(columns=["area_ha"])

    # Derived Feature 2: Anthropogenic Stress Index (Proximity to major ports)
    # Why? Blue carbon ecosystems closer to major industrial ports often face higher
    # environmental degradation, which might negatively correlate with their carbon storage potential.
    # Major ports (approximate coordinates: lat, lon)
    ports = {
        "Victoria": (48.4284, -123.3698),
        "Vancouver": (49.2827, -123.1207),
        "Prince Rupert": (54.3164, -130.3259),
    }

    df["min_distance_to_port_km"] = df.apply(
        lambda row: (
            min(
                [
                    haversine_distance(
                        row["latitude"],
                        row["longitude"],
                        port_coords[0],
                        port_coords[1],
                    )
                    for port_coords in ports.values()
                ]
            )
            if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
            else np.nan
        ),
        axis=1,
    )
    # Inverse of distance as a simple stress index (closer = higher stress)
    df["anthropogenic_stress_index"] = 1 / (
        df["min_distance_to_port_km"] + 1
    )  # Add 1 to avoid division by zero

    # Derived Feature 3: Oceanic Sea Surface Temperature (SST) & Salinity (SSS)
    print(
        "Fetching oceanic baseline features (SST & Salinity) from Bio-ORACLE via ERDDAP..."
    )
    ocean_cache = load_ocean_cache()

    sst_vals = []
    sss_vals = []
    buoy_sst_vals = []
    buoy_sss_vals = []

    for _, row in df.iterrows():
        # 1. Fetch Bio-ORACLE baselines (Fallback)
        sst, sss = fetch_bio_oracle_ocean_data(
            row["latitude"], row["longitude"], ocean_cache
        )
        sst_vals.append(sst)
        sss_vals.append(sss)

        # 2. Fetch Hakai Buoy data (High-fidelity mechanistic drivers)
        b_sst, b_sss = fetch_hakai_buoy_data(row["region"], ocean_cache)
        buoy_sst_vals.append(b_sst)
        buoy_sss_vals.append(b_sss)

    df["sea_surface_temperature_c"] = sst_vals
    df["sea_surface_salinity_pss"] = sss_vals
    df["buoy_temperature_c"] = buoy_sst_vals
    df["buoy_salinity_pss"] = buoy_sss_vals

    # Mechanistic Fallback: Use Bio-ORACLE if Buoy data is missing
    df["buoy_temperature_c"] = df["buoy_temperature_c"].fillna(
        df["sea_surface_temperature_c"]
    )
    df["buoy_salinity_pss"] = df["buoy_salinity_pss"].fillna(
        df["sea_surface_salinity_pss"]
    )

    # Derived Feature 4: Neighbor Density (15km radius)
    from sklearn.neighbors import BallTree

    print("Calculating deep spatial metrics (Neighbor Density & Regional KMeans)...")
    valid_coords = df[["latitude", "longitude"]].dropna()
    df["neighbor_density_15km"] = np.nan
    df["spatial_cluster_id"] = np.nan

    if not valid_coords.empty:
        # BallTree expects radians for haversine
        coords_rad = np.radians(valid_coords)
        tree = BallTree(coords_rad, metric="haversine")
        # 15km / 6371.0088 (Earth radius in km) = radians
        radius = 15.0 / 6371.0088
        counts = tree.query_radius(coords_rad, r=radius, count_only=True)
        # Minus 1 so we don't count the site itself
        df.loc[valid_coords.index, "neighbor_density_15km"] = counts - 1

        # Derived Feature 5: Spatial Clustering (KMeans k=6)
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(valid_coords)
        df.loc[valid_coords.index, "spatial_cluster_id"] = clusters

    # One-hot encode spatial clusters
    if "spatial_cluster_id" in df.columns:
        dummies = pd.get_dummies(
            df["spatial_cluster_id"], prefix="spatial_cluster", drop_first=False
        )
        df = pd.concat([df, dummies], axis=1)

    # Impute remaining NaNs in all numerical features before model selection
    numerical_features_to_impute_final = [
        "latitude",
        "longitude",
        "habitat_area_ha",
        "anthropogenic_stress_index",
        "percent_fines",
        "percent_oc",
        "d13C",
        "d15N",
        "measurement_depth_cm",
        "survey_year",
        "sea_surface_temperature_c",
        "sea_surface_salinity_pss",
        "neighbor_density_15km",
    ]
    for col in numerical_features_to_impute_final:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Select features for the model. Exclude target and other non-feature columns.
    # Select features for the model.
    # CRITICAL: We exclude raw 'latitude' and 'longitude' to prevent geographic overfitting.
    # We use mechanistic physical drivers (buoy data, SST, salinity) and distance-to-port instead.
    features = [
        col
        for col in df.columns
        if col.startswith(
            (
                "region_",
                "habitat_type_",
                "habitat_area_ha",
                "anthropogenic_stress_index",
                "percent_fines",
                "percent_oc",
                "d13C",
                "d15N",
                "sea_surface_temperature",
                "sea_surface_salinity",
                "buoy_temperature",
                "buoy_salinity",
                "neighbor_density_15km",
                "spatial_cluster_",
            )
        )
        and col not in ["latitude", "longitude", "spatial_cluster_id"]
    ]

    return (
        df[features],
        df,
    )  # Return features DataFrame and the full DataFrame with engineered features


from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


def train_and_predict_model(features_df, df_engineered):
    """
    Trains a regression model (Voting Ensemble by default) and predicts carbon density for gap sites.
    """

    # We train the model ONLY on locations where we have actual lab measurements (not 'is_carbon_gap').
    # Then we use those trained relationships to predict the density for the gap sites.
    # Separate data into training and prediction sets
    train_df = df_engineered[~df_engineered["is_carbon_gap"]].copy()
    predict_df = df_engineered[df_engineered["is_carbon_gap"]].copy()

    X_train = features_df.loc[train_df.index].astype(float).values
    y_train = train_df["carbon_density_gCm2"].astype(float).values

    if len(X_train) == 0:
        print("Warning: No data available for training the model.")
        df_engineered["predicted_carbon_density_gCm2"] = np.nan
        return df_engineered

    # Initialize and train the Ensemble (Pure tree boosting ensemble: XGBoost, LightGBM, CatBoost)
    estimators = []
    if XGBRegressor is not None:
        xgb = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=100, random_state=42, objective="reg:squarederror"
            ),
        )
        estimators.append(("xgb", xgb))
    if LGBMRegressor is not None:
        lgb = make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        )
        estimators.append(("lgb", lgb))
    if CatBoostRegressor is not None:
        cat = make_pipeline(
            SimpleImputer(strategy="median"),
            CatBoostRegressor(n_estimators=100, random_state=42, verbose=0),
        )
        estimators.append(("cat", cat))

    model = VotingRegressor(estimators)
    model.fit(X_train, y_train)
    print("\nVoting Ensemble Regressor trained successfully.")

    # Predict carbon density for gap sites
    if not predict_df.empty:
        X_predict = features_df.loc[predict_df.index].astype(float).values
        df_engineered.loc[
            df_engineered["is_carbon_gap"], "predicted_carbon_density_gCm2"
        ] = model.predict(X_predict)
    else:
        df_engineered["predicted_carbon_density_gCm2"] = np.nan

    return df_engineered


def evaluate_models(features_df, df_engineered):
    """
    Evaluates multiple models using Leave-One-Out Cross-Validation (LOOCV) and calculates RMSE and R².
    """

    measured_sites = df_engineered[~df_engineered["is_carbon_gap"]].copy()
    if len(measured_sites) < 2:  # Need at least 2 for LOOCV
        print("Warning: Not enough measured sites for meaningful LOOCV evaluation.")
        return {}

    # Features and target for measured sites
    X_measured = features_df.loc[measured_sites.index].astype(float).values
    y_measured = measured_sites["carbon_density_gCm2"].astype(float).values

    loo = LeaveOneOut()

    # Define the models suite
    models = {}
    estimators = []
    if XGBRegressor is not None:
        xgb = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=100, random_state=42, objective="reg:squarederror"
            ),
        )
        models["XGBoost"] = xgb
        estimators.append(("xgb", xgb))
    if LGBMRegressor is not None:
        lgb = make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        )
        models["LightGBM"] = lgb
        estimators.append(("lgb", lgb))
    if CatBoostRegressor is not None:
        cat = make_pipeline(
            SimpleImputer(strategy="median"),
            CatBoostRegressor(n_estimators=100, random_state=42, verbose=0),
        )
        models["CatBoost"] = cat
        estimators.append(("cat", cat))

    models["Voting Ensemble (Boosters)"] = VotingRegressor(estimators)

    results = {}

    print("\n--- Starting Model Evaluations (LOOCV) ---")
    for name, model in models.items():
        print(f"Evaluating {name}...")
        try:
            y_pred_loocv = cross_val_predict(
                model, X_measured, y_measured, cv=loo, n_jobs=-1
            )
            rmse = np.sqrt(mean_squared_error(y_measured, y_pred_loocv))
            r2 = r2_score(y_measured, y_pred_loocv)
            results[name] = {"rmse": rmse, "r2": r2}
        except Exception as e:
            print(f"Failed on {name}: {e}")
            results[name] = {"rmse": np.nan, "r2": np.nan}

    return results


if __name__ == "__main__":
    # Load and validate existing data
    df = load_and_validate_data()
    print("Data loaded and validated successfully!")

    # Fetch CRD GeoJSON data
    EELGRASS_URL = "https://mapservices.crd.bc.ca/arcgis/rest/services/Harbours/MapServer/25/query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=geojson"
    SEDIMENT_URL = "https://mapservices.crd.bc.ca/arcgis/rest/services/Harbours/MapServer/52/query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=geojson"

    eelgrass_geojson = fetch_geojson_data(
        EELGRASS_URL, local_cache_path="data/raw/crd_eelgrass.geojson"
    )
    sediment_geojson = fetch_geojson_data(
        SEDIMENT_URL, local_cache_path="data/raw/crd_sediment.geojson"
    )

    # Process CRD data
    crd_df = process_crd_eelgrass_data(eelgrass_geojson, sediment_geojson, df)
    print(f"Processed {len(crd_df)} CRD eelgrass sites.")

    # Process Janousek data
    janousek_file = "data/Janousek_et_al_2025_cores.csv"
    processed_janousek_df = process_janousek_data(janousek_file)
    print(f"Processed {len(processed_janousek_df)} Janousek eelgrass sites.")

    # Process Hakai data
    hakai_file = "data/hakai_prentice_eelgrass_sediment_carbon.csv"
    processed_hakai_df = process_hakai_data(hakai_file)
    print(f"Processed {len(processed_hakai_df)} Hakai sites.")

    # Combine all dataframes
    df_combined = pd.concat(
        [df, crd_df, processed_janousek_df, processed_hakai_df], ignore_index=True
    )

    # Remove duplicate site_ids, keeping the last (newest) entry if duplicates exist
    df_combined.drop_duplicates(subset=["site_id"], keep="last", inplace=True)

    # Process Hakai grain size data and merge
    df_combined = process_hakai_grain_size_data(
        "data/hakai_grain_size.csv", df_combined
    )
    print(f"Processed Hakai grain size data.")

    # Process Hakai isotope data and merge
    df_combined = process_hakai_isotope_data("data/hakai_isotope_data.csv", df_combined)
    print(f"Processed Hakai isotope data.")

    # Re-validate combined data
    df_combined = load_and_validate_data(df=df_combined)
    print("Combined data loaded and re-validated successfully!")

    features_df, df_engineered = engineer_features(df_combined.copy())
    print("Features engineered successfully!")

    df_final = train_and_predict_model(features_df, df_engineered.copy())
    print("Model trained and predictions made successfully!")
    print("\nDataFrame with predictions (first 5 rows):\n" + str(df_final.head()))
    print(
        "\nCarbon gap sites with predictions:\n"
        + str(
            df_final[df_final["is_carbon_gap"]][
                ["site_id", "carbon_density_gCm2", "predicted_carbon_density_gCm2"]
            ]
        )
    )

    evaluation_results = evaluate_models(features_df, df_final.copy())

    # Pretty print the model comparisons
    print("\nModel Comparison Results (LOOCV):")
    comparison_df = pd.DataFrame(evaluation_results).T
    print(comparison_df.sort_values(by="r2", ascending=False))

    # Calculate and display Feature Importances using Random Forest specifically
    model_for_importance = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=100, random_state=42),
    )
    # Retrain model on full training data to get feature importances
    train_df_for_importance = df_engineered[~df_engineered["is_carbon_gap"]].copy()
    X_train_for_importance = features_df.loc[train_df_for_importance.index].astype(
        float
    )
    y_train_for_importance = train_df_for_importance["carbon_density_gCm2"].astype(
        float
    )

    if not X_train_for_importance.empty:
        model_for_importance.fit(X_train_for_importance, y_train_for_importance)
        # Extract the RF model from the pipeline to get feature importances
        imputer_step = model_for_importance.named_steps["simpleimputer"]
        rf_step = model_for_importance.named_steps["randomforestregressor"]

        # Get the actual features that survived the imputer (i.e. drops habitat_area_ha)
        feature_names = imputer_step.get_feature_names_out(
            X_train_for_importance.columns
        )

        feature_importances = pd.DataFrame(
            {"feature": feature_names, "importance": rf_step.feature_importances_}
        ).sort_values(by="importance", ascending=False)
        print("\nFeature Importance Table:\n" + str(feature_importances))
    else:
        print("\nCannot calculate feature importances: No training data available.")

    # Extract final dataset without Machine Learning backend matrices
    df_final["carbon_density_gCm2_final"] = df_final["carbon_density_gCm2"].fillna(
        df_final["predicted_carbon_density_gCm2"]
    )
    cols_to_drop = [
        c
        for c in df_final.columns
        if c.startswith("region_")
        or c.startswith("habitat_type_")
        or c.startswith("sediment_type_")
        or c == "predicted_carbon_density_gCm2"
    ]
    df_final.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Save the polished Kaggle CSV
    output_file = "unified_bc_blue_carbon_filled.csv"
    df_final.to_csv(output_file, index=False)
    print("\nFilled data saved to " + str(output_file))
