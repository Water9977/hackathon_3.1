# 🛰️ Eco-Engine v1.5 (Hackathon Edition)

An AI-powered Geospatial Ecological Analysis tool. This engine uses satellite imagery from the Microsoft Planetary Computer to assess environmental health, predict climate-related stress, and identify key ecological drivers.

## 🌟 Key Features
- **🌍 Dynamic Location Search**: Type any city or region name (e.g., "Dehradun", "Amazon Rainforest"), and the AI will automatically find the coordinates.
- **🌱 Fixed Carbon Stock Pipeline**: Accurately calculates Total Carbon Stock (Aboveground + Belowground) using the **Harmonized Global Biomass (HGB)** and Chloris datasets.
- **🌡️ AI Thermal Stress Index**: Analyzes Landsat thermal data to calculate surface heat stress in Celsius.
- **🤖 Integrated ML Model**: Trains a `RandomForestRegressor` on regional data to learn local ecological relationships.
- **🔮 Climate Simulation**: Predicts future ecological stress based on a simulated **+2°C warming scenario**.
- **📊 Detailed Eco-Reports**: Automatically prints structured reports on Vegetation, Moisture, Terrain, and Carbon health.

---

## 🛠️ Installation

Run these commands one-by-one in your terminal (Windows + R then type 'cmd'):

```bash
pip install pystac-client planetary-computer odc-stac numpy matplotlib requests
pip install scikit-learn joblib pandas xarray
```

---

## 🚀 How to Use

1. **Run the script**:
   ```bash
   python main.py
   ```
2. **Enter Location**: When prompted, type the name of the region you want to analyze.
3. **Wait for Processing**: The engine will connect to NASA/ESA satellites, download the latest cloud-free data from 2024, and train the AI model.
4. **Check Results**: 
   - View the **Text Reports** in your terminal.
   - Open the `sentinel_output/` folder to see the **AI_PREDICTION** heatmap!

---

## 📊 Technical Details (For Judges)
- **Data Sources**: Sentinel-2 (Optical), Landsat-C2-L2 (Thermal), Copernicus GLO-30 (DEM), ESA WorldCover (Land Use), HGB/Chloris (Carbon).
- **Core Formula (CESI)**: 
  `0.30 * NDVI + 0.20 * NDWI + 0.20 * LST + 0.10 * Slope + 0.20 * Carbon`
- **Machine Learning**: 100-tree Random Forest Regressor used for feature importance and predictive simulation.

---

## ⚠️ Notes
- **Carbon Fix**: The previously broken `soilgrids` pipeline has been replaced with the functional `hgb` (Harmonized Global Biomass) dataset.
- **AI Model**: The imports for `RandomForest` are now actively used to simulate warming impact and calculate feature importance.

**Good luck at the Hackathon! 🚀**
