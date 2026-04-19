import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict


def generate_map():
    print("Generating map_coverage.png...")
    df = pd.read_csv("unified_bc_blue_carbon_filled.csv")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )

    import contextily as cx

    gdf_mercator = gdf.to_crs(epsg=3857)

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot our specific data colored by Data Source
    gdf_mercator.plot(
        column="data_source",
        ax=ax,
        cmap="viridis",
        markersize=60,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        legend_kwds={"title": "Data Source", "loc": "lower left"},
    )

    # Add beautiful contextily basemap
    cx.add_basemap(
        ax, crs=gdf_mercator.crs.to_string(), source=cx.providers.CartoDB.Positron
    )

    plt.title(
        "Spatial Coverage of Uncharted BC Blue Carbon Dataset", fontsize=16, pad=15
    )
    ax.set_axis_off()  # Hide lat/lon axes for a cleaner map look

    plt.tight_layout()
    plt.savefig("map_coverage.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_model_plots():
    print("Generating model_scatter.png and model_residuals.png...")
    # Read the full engineered dataset from earlier, or we can just reproduce a clean subset
    # Since we dropped ML categories in pipeline output, we can't easily reproduce the model blindly
    # without running engineer_features.

    # Let's import the pipeline functions!
    from pipeline import (
        load_and_validate_data,
        process_janousek_data,
        process_hakai_data,
        engineer_features,
    )

    # Alternatively, the easiest way to generate a strong Kaggle plot without re-running the huge pipeline
    # is to fetch 'unified_bc_blue_carbon.csv', engineer it here minimally, and run RF.

    df = pd.read_csv("unified_bc_blue_carbon_filled.csv")
    df = df[df["habitat_type"] == "eelgrass"]
    df = df.dropna(subset=["carbon_density_gCm2"])

    # Clean simple features + new oceanic features + new spatial features
    # Instead of manual tracking, dynamically grab all active ML numerical/encoded features from the csv
    features = [
        c
        for c in df.columns
        if c.startswith(
            (
                "latitude",
                "longitude",
                "anthropogenic_stress",
                "sea_surface_temperature",
                "sea_surface_salinity",
                "neighbor_density_15km",
                "spatial_cluster_",
            )
        )
    ]
    X = df[features].copy()
    # Fill NA for the simple visualizer just in case
    X.fillna(X.median(), inplace=True)
    y = df["carbon_density_gCm2"]

    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(n_estimators=100, random_state=42)
    except ImportError:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Cross Validation
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)

    # Scatter Plot - Premium styling
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 7))
    plt.scatter(y, y_pred, alpha=0.7, edgecolors="white", s=80, color="#00A86B")

    # 1:1 Line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="#2C3E50",
        linestyle="--",
        lw=2,
        label="1:1 Perfect Prediction",
    )

    # Add Metrics Box
    from sklearn.metrics import mean_squared_error, r2_score

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
    plt.gca().text(
        0.05,
        0.95,
        f"Model: XGBoost (LOOCV)\nRMSE: {rmse:.2f}\nR²: {r2:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    plt.title(
        "Predicted vs Actual Blue Carbon Sequestration",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Actual Carbon Density (gC/m²)", fontsize=12)
    plt.ylabel("Predicted Carbon Density (gC/m²)", fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("model_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Residual Plot - Premium KDE with dynamic fills
    residuals = y - y_pred
    plt.figure(figsize=(9, 6))

    ax = sns.histplot(
        residuals, kde=True, color="#FF6F61", bins=40, edgecolor="white", alpha=0.7
    )

    # Add vertical lines for Mean and Zero
    plt.axvline(
        x=0, color="#2C3E50", linestyle="--", lw=2, label="Zero Error (Perfect)"
    )
    mean_resid = np.mean(residuals)
    plt.axvline(
        x=mean_resid,
        color="#3498DB",
        linestyle="-.",
        lw=2,
        label=f"Mean Error: {mean_resid:.2f}",
    )

    plt.title(
        "Model Residuals (Prediction Error Distribution)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Prediction Error (gC/m²)", fontsize=12)
    plt.ylabel("Frequency density", fontsize=12)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("model_residuals.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    generate_map()
    generate_model_plots()
    print("Done!")
