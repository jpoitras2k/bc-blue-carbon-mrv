import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict

def generate_map():
    print("Generating map_coverage.png...")
    df = pd.read_csv('unified_bc_blue_carbon_filled.csv')
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot our specific data colored by Data Source
    sns.scatterplot(
        x='longitude', y='latitude',
        hue='data_source',
        data=df,
        ax=ax,
        palette='viridis',
        s=40,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Zoom in on BC/Salish Sea
    ax.set_xlim([-135, -122])
    ax.set_ylim([48, 56])
    
    plt.title("Spatial Coverage of Uncharted BC Blue Carbon Dataset", fontsize=14, pad=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Data Source", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('map_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_model_plots():
    print("Generating model_scatter.png and model_residuals.png...")
    # Read the full engineered dataset from earlier, or we can just reproduce a clean subset
    # Since we dropped ML categories in pipeline output, we can't easily reproduce the model blindly 
    # without running engineer_features.
    
    # Let's import the pipeline functions!
    from pipeline import load_and_validate_data, process_janousek_data, process_hakai_data, engineer_features
    # Alternatively, the easiest way to generate a strong Kaggle plot without re-running the huge pipeline 
    # is to fetch 'unified_bc_blue_carbon.csv', engineer it here minimally, and run RF.
    
    df = pd.read_csv('unified_bc_blue_carbon.csv')
    df = df[df['habitat_type'] == 'eelgrass']
    df = df.dropna(subset=['carbon_density_gCm2'])
    
    # Clean simple features
    features = ['latitude', 'longitude']
    X = df[features]
    y = df['carbon_density_gCm2']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Cross Validation
    loo = LeaveOneOut()
    y_pred = cross_val_predict(rf, X, y, cv=loo)
    
    # Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.6, edgecolors='w', s=60, color='#008ABC')
    
    # 1:1 Line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='1:1 Perfect Prediction')
    
    plt.title("Predicted vs Actual Blue Carbon Sequestration (LOOCV)", fontsize=14)
    plt.xlabel("Actual Carbon Density (gC/m²)")
    plt.ylabel("Predicted Carbon Density  (gC/m²)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig('model_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Residual Plot
    residuals = y - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.title("Distribution of Model Residuals", fontsize=14)
    plt.xlabel("Prediction Error (gC/m²)")
    plt.ylabel("Frequency")
    plt.axvline(x=0, color='k', linestyle='--', lw=2)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig('model_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    generate_map()
    generate_model_plots()
    print("Done!")
