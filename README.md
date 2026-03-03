# 🛰️ EcoEngine AI — Hackathon 3.1

**EcoEngine AI** is a professional geospatial ecological analysis platform. It leverages satellite imagery from Microsoft's Planetary Computer to assess environmental health, predict climate risk, and monitor carbon stocks in real-time.

---

## 🚀 Features

### 📍 1. Dynamic Regional Analysis
Unlike standard tools with fixed coordinates, EcoEngine allows you to search for **any location** (e.g., "Amazon Rainforest", "Dehradun", "Kyoto"). It automatically geocodes the area and fetches relevant satellite data.

### 🌳 2. Advanced Carbon Stock Pipeline
We use a high-fidelity data pipeline combining **Chloris Biomass** (aboveground) and **HGB Harmonized Global Biomass** (above + belowground) to calculate accurate carbon density for any region.

### 🧠 3. regional AI Training (Random Forest)
The engine doesn't just use formulas; it trains a **RandomForestRegressor** specifically on the unique ecological fingerprint of your chosen region.
- **Feature Importance:** Identifies if Heat, Water, or Devegetation is the primary stress driver.
- **Regional Learning:** Adapts to different biomes (Deserts vs. Rainforests).

### 🌡️ 4. AI Climate Simulation (+2°C Warming)
A predictive module that simulates a global warming scenario.
- Uses the trained AI model to project how ecological stress will increase if the local temperature rises by 2°C.
- Generates side-by-side comparison heatmaps of current vs. projected risk.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Make sure you have Python 3.9+ installed.

### 2. Install Dependencies
Run the following command to install all necessary geospatial and AI libraries:

```bash
pip install pystac-client planetary-computer odc-stac numpy matplotlib scikit-learn joblib pandas requests xarray
```

---

## 📑 How to Use

1. **Run the Script:**
   ```bash
   python main.py
   ```
2. **Enter Location:**
   When prompted, type the name of the city or region you want to analyze.
3. **View Results:**
   - **Terminal:** Check for the "ECOLOGICAL AI REPORT" which includes Vegetation, Moisture, Thermal, and Carbon stats.
   - **AI Insight:** Look for the "TOP STRESS DRIVERS" to see what is hurting the local environment most.
   - **Visuals:** Open the `sentinel_output` folder to find your generated Stress Heatmaps and AI Predictions.

---

## 📊 Key Metrics Explained

| Metric | Description |
|---|---|
| **NDVI** | Vegetation health index (Greenness). |
| **NDWI** | Moisture/Water stress index. |
| **LST** | Land Surface Temperature (in Celsius). |
| **CESI** | Composite Ecological Stress Index (The "Heartbeat" of the ecosystem). |
| **Carbon Stock** | Total biomass carbon stored in the area (MgC/ha). |

---

## 🏗️ Tech Stack
- **Data Source:** Microsoft Planetary Computer (Sentinel-2, Landsat, ESA WorldCover).
- **AI Engine:** Scikit-Learn (Random Forest).
- **Geocoding:** OpenStreetMap (Nominatim).
- **Processing:** ODC-STAC, XArray, NumPy.

---
*Created for Hackathon 3.1*
