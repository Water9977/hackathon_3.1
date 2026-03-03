from pystac_client import Client
import planetary_computer
import xarray as xr
from odc.stac import load
import numpy as np
import os
import matplotlib.pyplot as plt
import requests  # Added for geocoding
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd # Added for better data handling

# ==========================================================
# CONFIGURATION & DYNAMIC LOCATION
# ==========================================================

def get_bbox_from_name(location_name):
    print(f"Searching for location: {location_name}...")
    url = f"https://nominatim.openstreetmap.org/search?q={location_name}&format=json"
    headers = {'User-Agent': 'EcoEngine/1.0 (Hackathon Project)'}
    response = requests.get(url, headers=headers).json()
    
    if not response:
        print("Location not found. Using default coordinates.")
        return (78.00, 30.40, 78.15, 30.55)
    
    # Get bounding box [minlat, maxlat, minlon, maxlon] from OSM
    osm_bbox = response[0]['boundingbox']
    # Convert to STAC format (min_lon, min_lat, max_lon, max_lat)
    return (float(osm_bbox[2]), float(osm_bbox[0]), float(osm_bbox[3]), float(osm_bbox[1]))

loc_input = input("Enter a city or region to analyze (e.g., 'Dehradun', 'Amazon Rainforest'): ")
BBOX = get_bbox_from_name(loc_input)
print(f"Final BBOX: {BBOX}")

# Configuration
DATE_RANGE = "2024-01-01/2024-12-31" # Updated to a full year for better data coverage
MAX_CLOUD = 20
RESOLUTION = 20        

OUTPUT_FOLDER = "sentinel_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# UTILITIES
# ==========================================================
def normalize(arr):
    # Safe normalization to handle NaNs and zeros
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmax == vmin:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin + 1e-10)

# ==========================================================
# ANALYSIS MODELS
# ==========================================================

def vegetation_model(ndvi_array, landcover_array=None):
    ndvi_flat = ndvi_array.flatten()
    ndvi_flat = ndvi_flat[~np.isnan(ndvi_flat)]

    if len(ndvi_flat) == 0:
        return {"error": "No valid NDVI pixels"}

    mean_ndvi = np.mean(ndvi_flat)
    std_ndvi = np.std(ndvi_flat)
    high_veg_pct = np.sum(ndvi_flat > 0.5) / len(ndvi_flat)
    low_veg_pct = np.sum(ndvi_flat < 0.2) / len(ndvi_flat)

    vegetation_score = np.clip(
        (np.clip(mean_ndvi, 0, 1) * 50) +
        (high_veg_pct * 25) +
        ((1 - low_veg_pct) * 20) +
        ((1 - np.clip(std_ndvi, 0, 1)) * 20),
        0, 100
    )

    if vegetation_score < 30:
        vegetation_class = "Degraded"
    elif vegetation_score < 50:
        vegetation_class = "Low Vegetation"
    elif vegetation_score < 70:
        vegetation_class = "Moderate"
    elif vegetation_score < 85:
        vegetation_class = "Healthy"
    else:
        vegetation_class = "Dense Ecological Zone"

    report = {
        "Vegetation Score": round(float(vegetation_score), 2),
        "Vegetation Class": vegetation_class,
        "Mean NDVI": round(float(mean_ndvi), 3),
        "NDVI Std Dev": round(float(std_ndvi), 3),
        "High Vegetation %": round(float(high_veg_pct * 100), 2),
        "Low Vegetation %": round(float(low_veg_pct * 100), 2),
    }

    if landcover_array is not None:
        lc = landcover_array.flatten()
        report["Tree Cover %"] = round(float(np.sum(lc == 10) / lc.size * 100), 2)
        report["Shrubland %"] = round(float(np.sum(lc == 20) / lc.size * 100), 2)
        report["Grassland %"] = round(float(np.sum(lc == 30) / lc.size * 100), 2)
        report["Cropland %"] = round(float(np.sum(lc == 40) / lc.size * 100), 2)
        report["Built-up %"] = round(float(np.sum(lc == 50) / lc.size * 100), 2)

    return report

def moisture_model(ndwi_array):
    ndwi_flat = ndwi_array.flatten()
    ndwi_flat = ndwi_flat[~np.isnan(ndwi_flat)]

    if len(ndwi_flat) == 0:
        return {"error": "No valid NDWI pixels"}

    mean_ndwi = np.mean(ndwi_flat)
    std_ndwi = np.std(ndwi_flat)

    if mean_ndwi < 0:
        moisture_class = "Severe Moisture Stress"
    elif mean_ndwi < 0.1:
        moisture_class = "Moderate Stress"
    elif mean_ndwi < 0.3:
        moisture_class = "Healthy Moisture"
    else:
        moisture_class = "High Water Content"

    return {
        "Mean NDWI": round(float(mean_ndwi), 3),
        "NDWI Std Dev": round(float(std_ndwi), 3),
        "Moisture Class": moisture_class
    }

def thermal_stress_model(lst_array):
    lst_flat = lst_array.flatten()
    lst_flat = lst_flat[~np.isnan(lst_flat)]

    if len(lst_flat) == 0:
        return {"error": "No valid LST pixels"}

    mean_lst = np.mean(lst_flat)
    max_lst = np.max(lst_flat)
    tsi = np.clip((mean_lst - 25) / 15, 0, 1)

    if tsi < 0.2:
        stress_class = "No Thermal Stress"
    elif tsi < 0.4:
        stress_class = "Mild Stress"
    elif tsi < 0.6:
        stress_class = "Moderate Stress"
    elif tsi < 0.8:
        stress_class = "High Stress"
    else:
        stress_class = "Extreme Heat Stress"

    return {
        "Mean LST (°C)": round(float(mean_lst), 2),
        "Max LST (°C)": round(float(max_lst), 2),
        "Thermal Stress Index": round(float(tsi), 2),
        "Thermal Stress Class": stress_class
    }

def terrain_model(elevation_array):
    valid = elevation_array[~np.isnan(elevation_array)]

    if len(valid) == 0:
        return {"error": "No valid elevation pixels"}

    mean_elev = np.mean(valid)
    elev_range = np.max(valid) - np.min(valid)

    gy, gx = np.gradient(elevation_array)
    slope = np.sqrt(gx**2 + gy**2)
    mean_slope = np.nanmean(slope)

    return {
        "Mean Elevation (m)": round(float(mean_elev), 2),
        "Elevation Range (m)": round(float(elev_range), 2),
        "Mean Slope Index": round(float(mean_slope), 4)
    }

def carbon_model(carbon_array):
    valid = carbon_array[~np.isnan(carbon_array)]

    if len(valid) == 0:
        return {"error": "No valid carbon pixels"}

    mean_carbon = np.mean(valid)
    max_carbon = np.max(valid)

    if mean_carbon < 30:
        carbon_class = "Low Carbon Landscape"
    elif mean_carbon < 80:
        carbon_class = "Moderate Carbon Storage"
    elif mean_carbon < 150:
        carbon_class = "High Carbon Zone"
    else:
        carbon_class = "Critical Carbon Reservoir"

    return {
        "Mean Carbon (MgC/ha)": round(float(mean_carbon), 2),
        "Max Carbon (MgC/ha)": round(float(max_carbon), 2),
        "Carbon Class": carbon_class
    }

# ==========================================================
# DATA FETCHING
# ==========================================================

print("Connecting to Planetary Computer...")
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

# Search Sentinel-2
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=BBOX,
    datetime=DATE_RANGE,
    query={"eo:cloud_cover": {"lte": MAX_CLOUD}}
)
items = list(search.items())

if not items:
    print("No Sentinel-2 images found.")
    exit()

print(f"Found {len(items)} Sentinel images")

# Load Static Datasets
print("Loading static datasets...")
dem_search = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
dem_item = list(dem_search.items())[0]

wc_search = catalog.search(collections=["esa-worldcover"], bbox=BBOX)
wc_item = list(wc_search.items())[0]

landsat_search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=BBOX,
    datetime=DATE_RANGE,
    query={"eo:cloud_cover": {"lte": MAX_CLOUD}}
)
landsat_items = list(landsat_search.items())
landsat_item = landsat_items[0] if landsat_items else None

# Carbon Datasets (THE FIX IS HERE)
print("Searching Carbon datasets (Chloris + HGB)...")
carbon_search = catalog.search(collections=["chloris-biomass"], bbox=BBOX)
carbon_items = list(carbon_search.items())
carbon_item = carbon_items[0] if carbon_items else None

# hgb is the "Harmonized Global Biomass" dataset
hgb_search = catalog.search(collections=["hgb"], bbox=BBOX)
hgb_items = list(hgb_search.items())
hgb_item = hgb_items[0] if hgb_items else None

# ==========================================================
# PROCESSING LOOP
# ==========================================================

for item in items:
    print("\nProcessing:", item.id)

    # Load Base Image
    data = load(
        [item],
        bands=["B04", "B08", "B11"],
        bbox=BBOX,
        resolution=RESOLUTION,
        chunks={"x": 1024, "y": 1024}
    ).compute()

    # --- Carbon Stock Fix ---
    agb_carbon_stock = None
    bgb_carbon_stock = None
    
    # 1. Try Chloris (Aboveground Biomass)
    if carbon_item:
        try:
            # We explicitly ask for the 'biomass' asset
            cb_data = load([carbon_item], bands=["biomass"], like=data).compute()
            biomass = cb_data["biomass"].values[0].astype(float)
            biomass[biomass > 2000000] = np.nan # Clean extreme values
            agb_carbon_stock = biomass * 0.47
        except Exception as e:
            print(f"Note: Chloris loading failed: {e}")

    # 2. Try HGB (Harmonized Global Biomass for Above + Below ground)
    hgb_carbon = np.zeros_like(data.B04.values[0])
    if hgb_item:
        try:
            hgb_data = load([hgb_item], bands=["aboveground", "belowground"], like=data).compute()
            above = hgb_data["aboveground"].values[0].astype(float)
            below = hgb_data["belowground"].values[0].astype(float)
            hgb_carbon = np.nan_to_num(above) + np.nan_to_num(below)
        except Exception as e:
            print(f"Note: HGB loading failed: {e}")

    # Combine pools
    # If Chloris worked, we use it for Aboveground + HGB for Belowground
    # Otherwise we use HGB for everything
    if agb_carbon_stock is not None:
        total_carbon = np.nan_to_num(agb_carbon_stock) + hgb_carbon # simplistic sum
    else:
        total_carbon = hgb_carbon

    carbon_stock = total_carbon
    print(f"Carbon Stock Stats -> Min: {np.nanmin(carbon_stock):.2f}, Max: {np.nanmax(carbon_stock):.2f}")

    # Core Indices
    red = data.B04.values[0].astype(float)
    nir = data.B08.values[0].astype(float)
    swir = data.B11.values[0].astype(float)

    ndvi = (nir - red) / (nir + red + 1e-10)
    ndwi = (nir - swir) / (nir + swir + 1e-10)

    # Static Data
    dem_data = load([dem_item], bands=["data"], like=data).compute()
    elevation = dem_data["data"].values[0].astype(float)

    wc_data = load([wc_item], bands=["map"], like=data).compute()
    landcover = wc_data["map"].values[0].astype(float)

    # Thermal Data
    if landsat_item:
        lst_data = load([landsat_item], bands=["lwir11"], like=data).compute()
        # Scale factor for Landsat 8-9 Thermal
        kelvin = lst_data["lwir11"].values[0] * 0.00341802 + 149.0
        lst_celsius = kelvin - 273.15
    else:
        lst_celsius = np.zeros_like(ndvi)

    # ==========================================================
    # CESI CALCULATION (Restored)
    # ==========================================================
    gy, gx = np.gradient(elevation)
    slope = np.sqrt(gx**2 + gy**2)

    carbon_stress = 1 - normalize(carbon_stock) 
    ndvi_stress = 1 - normalize(ndvi)
    ndwi_stress = 1 - normalize(ndwi)
    lst_stress = normalize(lst_celsius)
    slope_stress = normalize(slope)

    CESI = (
        0.30 * ndvi_stress +
        0.20 * ndwi_stress +
        0.20 * lst_stress +
        0.10 * slope_stress +
        0.20 * carbon_stress
    )

    # ==========================================================
    # ML TRAINING & SIMULATION
    # ==========================================================
    print("\nPreparing AI Training Dataset...")
    
    # Reshape and stack features for ML
    # Features: NDVI, NDWI, LST, Elevation, Slope, Carbon
    features = np.stack([
        ndvi.flatten(), 
        ndwi.flatten(), 
        lst_celsius.flatten(), 
        elevation.flatten(), 
        slope.flatten(), 
        carbon_stock.flatten()
    ], axis=1)
    
    target = CESI.flatten()
    
    # Remove NaNs for training
    mask = ~np.isnan(features).any(axis=1) & ~np.isnan(target)
    X = features[mask]
    y = target[mask]
    
    # We'll sample 5000 points to keep it fast for the hackathon
    sample_idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
    X_sample = X[sample_idx]
    y_sample = y[sample_idx]
    
    print(f"Training RandomForest Model on {len(X_sample)} regional data points...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_sample, y_sample)
    
    # SAVE THE MODEL
    model_path = os.path.join(OUTPUT_FOLDER, f"eco_ai_model_{item.id}.joblib")
    joblib.dump(model, model_path)
    
    # FEATURE IMPORTANCE
    feature_names = ["Vegetation (NDVI)", "Moisture (NDWI)", "Temperature (LST)", "Elevation", "Slope", "Carbon Stock"]
    importances = model.feature_importances_
    
    print("\n--- AI INSIGHT: TOP STRESS DRIVERS ---")
    driver_data = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for name, imp in driver_data:
        print(f"{name}: {imp*100:.1f}% influence")

    # ==========================================================
    # AI CLIMATE SIMULATION: +2°C WARMING SCENARIO
    # ==========================================================
    print("\nRunning AI Climate Simulation (+2°C Warming Scenario)...")
    
    # Create a copy of traits for simulation
    sim_features = X.copy()
    # index 2 is LST
    sim_features[:, 2] += 2.0 
    
    sim_risk = model.predict(sim_features)
    
    # Reshape back to map
    CESI_SIM = np.full(CESI.shape, np.nan)
    CESI_SIM.flat[mask] = sim_risk
    
    # Calculate Impact
    avg_current_risk = np.mean(y)
    avg_sim_risk = np.mean(sim_risk)
    risk_increase = (avg_sim_risk - avg_current_risk) / avg_current_risk * 100
    
    print(f"SIMULATION RESULT: Projected Stress Increase: {risk_increase:.1f}%")

    # ==========================================================
    # OUTPUTS
    # ==========================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Current Map
    im1 = ax1.imshow(CESI, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax1.set_title(f"Current Ecological Stress\n{item.id}")
    plt.colorbar(im1, ax=ax1, label="Stress Index")
    
    # Simulation Map
    im2 = ax2.imshow(CESI_SIM, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax2.set_title(f"AI Projected Stress (+2°C Warming)\nIncrease: {risk_increase:.1f}%")
    plt.colorbar(im2, ax=ax2, label="Stress Index")

    output_path = os.path.join(OUTPUT_FOLDER, f"AI_PREDICTION_{item.id}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("AI Prediction comparison saved:", output_path)

    # Reports
    veg_report = vegetation_model(ndvi, landcover)
    moisture_report = moisture_model(ndwi)
    terrain_report = terrain_model(elevation)
    thermal_report = thermal_stress_model(lst_celsius)
    carbon_report = carbon_model(carbon_stock)

    print("\n" + "="*30)
    print("      ECOLOGICAL AI REPORT")
    print("="*30)
    
    print("\n--- VEGETATION ---")
    for k, v in veg_report.items(): print(f"{k}: {v}")

    print("\n--- MOISTURE ---")
    for k, v in moisture_report.items(): print(f"{k}: {v}")

    print("\n--- TERRAIN ---")
    for k, v in terrain_report.items(): print(f"{k}: {v}")

    print("\n--- THERMAL STRESS ---")
    for k, v in thermal_report.items(): print(f"{k}: {v}")

    print("\n--- CARBON STOCK ---")
    for k, v in carbon_report.items(): print(f"{k}: {v}")
    
    print(f"\nAI MODEL SAVED TO: {model_path}")

print("\n✅ FULL AI ECO ENGINE COMPLETE — WITH REAL MACHINE LEARNING")
